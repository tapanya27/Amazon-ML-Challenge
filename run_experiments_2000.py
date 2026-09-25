"""Targeted 2000-S1 experiments. Read-only SQLite. Does not change production."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.blocking.sqlite_index import LEGALISH, MAX_POSTING, SqliteReferenceIndex, _tokens
from src.features.pair_features import PairFeatureGenerator
from src.models.decision import pairs_to_predictions
from src.models.evaluation import compute_entity_f05, evaluate_macro_f05
from src.preprocessing import preprocess_dataframe


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def parse_ids(s):
    if pd.isna(s) or not str(s).strip():
        return set()
    return {x.strip() for x in str(s).split(",") if x.strip()}


def load_gt(path):
    gt_df = pd.read_csv(path, sep="\t")
    return {s1: parse_ids(ms) for s1, ms in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"])}


def is_indic_name(name: str) -> bool:
    if not name:
        return False
    return re.search(r"[a-z]", name) is None and bool(re.search(r"[\u0900-\u0d7f]", name) or not name.isascii())


def retrieve_variant(index, name, addr, country, *, retrieve_cap=350, max_tokens=6,
                     always_addr=False, posting_limit=400, glued=False,
                     prefix_missing=False, indic_addr=False, typo_map=None):
    scores = {}
    country = (country or "").lower()

    def add_posting(posting, weight):
        for rid in posting:
            scores[int(rid)] = scores.get(int(rid), 0) + weight

    def add_tokens(text, which, min_len, stop, weight, ignore_fill=False):
        toks = list(_tokens(text, min_len, stop))
        toks.sort(key=lambda t: index.token_df(t, which) or 10**9)
        used = 0
        for t in toks:
            posting = index.posting(t, which, limit=posting_limit)
            if posting.size == 0:
                continue
            used += 1
            add_posting(posting, weight)
            if used >= max_tokens or (not ignore_fill and len(scores) >= retrieve_cap * 3):
                break

    for rid in index.exact_name_rows(name, country):
        scores[rid] = scores.get(rid, 0) + 50

    if glued and name and " " in name:
        g = name.replace(" ", "")
        for rid in index.exact_name_rows(g, country):
            scores[rid] = scores.get(rid, 0) + 40
        posting = index.posting(g, "name", limit=posting_limit)
        add_posting(posting, 8)

    skip_name = indic_addr and is_indic_name(name)
    if not skip_name:
        add_tokens(name, "name", 3, LEGALISH, 3, ignore_fill=False)

    if prefix_missing and name:
        ntoks = [t for t in _tokens(name, 3, LEGALISH) if index.token_df(t, "name") == 0 and len(t) >= 4]
        for t in ntoks[:2]:
            pat = t + "%"
            if country:
                rows = index.conn.execute(
                    "SELECT row_id FROM refs WHERE name LIKE ? AND country = ? LIMIT 200",
                    (pat, country),
                ).fetchall()
            else:
                rows = index.conn.execute(
                    "SELECT row_id FROM refs WHERE name LIKE ? LIMIT 200", (pat,)
                ).fetchall()
            add_posting([r[0] for r in rows], 6)

    if typo_map is not None and name:
        for t in list(_tokens(name, 3, LEGALISH)):
            if index.token_df(t, "name"):
                continue
            if len(t) < 5:
                continue
            key = t[:3]
            for cand, _c in typo_map.get(key, ()):
                if cand == t or abs(len(cand) - len(t)) > 1:
                    continue
                if sum(a != b for a, b in zip(cand, t)) + abs(len(cand) - len(t)) > 1:
                    continue
                posting = index.posting(cand, "name", limit=posting_limit)
                add_posting(posting, 4)

    add_tokens(addr, "addr", 2, set(), 2, ignore_fill=always_addr)

    if country and scores:
        recs = index.fetch_rows(list(scores.keys())[: retrieve_cap * 3])
        scores = {rid: scores[rid] for rid, _e, _n, _a, co in recs if co == country or not co}
    if not scores:
        return np.empty(0, dtype=np.int32), 0, False
    raw_n = len(scores)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:retrieve_cap]
    return np.asarray([r for r, _ in ranked], dtype=np.int32), raw_n, raw_n >= retrieve_cap


def eids_from_rows(index, rows):
    if rows is None or (hasattr(rows, "size") and rows.size == 0):
        return []
    recs = index.fetch_rows(rows.tolist() if hasattr(rows, "tolist") else list(rows))
    order = []
    idmap = {r[0]: r[1] for r in recs}
    for rid in (rows.tolist() if hasattr(rows, "tolist") else rows):
        eid = idmap.get(int(rid))
        if eid:
            order.append(eid)
    return order


def classify(s1_ids, true_map, pred_map, retrieve_eids, topk_eids):
    cats = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "OK": 0}
    for s1 in s1_ids:
        true = true_map.get(s1, set())
        pred = pred_map.get(s1, set())
        retr = set(retrieve_eids.get(s1, []))
        topk = set(topk_eids.get(s1, []))
        f05 = compute_entity_f05(pred, true)
        if f05 == 1.0:
            cats["OK"] += 1
            continue
        if not topk and true:
            cats["E"] += 1
            continue
        if not true and pred:
            cats["D"] += 1
            continue
        missing_retr = [t for t in true if t not in retr]
        missing_top = [t for t in true if t not in topk]
        if missing_retr:
            cats["A"] += 1
        elif missing_top:
            cats["B"] += 1
        else:
            cats["C"] += 1
    return cats


def link_recall(s1_ids, true_map, cand_map):
    found = tot = 0
    for s1 in s1_ids:
        true = true_map.get(s1, set())
        if not true:
            continue
        tot += len(true)
        found += len(true & set(cand_map.get(s1, [])))
    return found / tot if tot else 1.0, found, tot


def true_survive_frac(s1_ids, true_map, cand_map):
    return link_recall(s1_ids, true_map, cand_map)[0]


def score_topk(index, feat_gen, model, feature_cols, threshold, s1_ids, names, addrs, countries, top_rows):
    feat_df = feat_gen.generate_from_index(index, s1_ids, names, addrs, countries, top_rows)
    if len(feat_df):
        feat_df["score"] = model.predict_proba(feat_df[feature_cols])[:, 1]
        exact = (feat_df["name_exact"] >= 1.0) & (feat_df["addr_exact"] >= 1.0) & (feat_df["country_same"] >= 1.0)
        feat_df.loc[exact, "score"] = np.maximum(feat_df.loc[exact, "score"], threshold)
        preds = pairs_to_predictions(feat_df, threshold=threshold)
    else:
        preds = {}
    return feat_df, preds


def apply_guard(feat_df, threshold, min_name_tok=0.5):
    if feat_df is None or feat_df.empty:
        return {}
    df = feat_df.copy()
    mask = (df["addr_exact"] >= 1.0) & (df["name_exact"] < 1.0) & (df["name_token_set"] < min_name_tok)
    df.loc[mask, "score"] = 0.0
    return pairs_to_predictions(df, threshold=threshold)


def main():
    peak = [rss_mb()]
    stop = threading.Event()

    def loop():
        while not stop.wait(0.4):
            peak[0] = max(peak[0], rss_mb())

    threading.Thread(target=loop, daemon=True).start()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    threshold = float(meta.get("best_threshold", 0.92))
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_baseline.json")
    gt_all = load_gt(cfg["data"]["train"]["ground_truth"])
    sample = pd.read_csv("eval_train/_sample_s1.tsv", sep="\t")
    s1_p = preprocess_dataframe(sample, "exp")
    s1_ids = s1_p["entity_id"].tolist()
    assert len(s1_ids) == 2000
    names = s1_p["name_normalized"].fillna("").astype(str).replace("nan", "").tolist()
    addrs = s1_p["address_normalized"].fillna("").astype(str).replace("nan", "").tolist()
    countries = s1_p["country_normalized"].fillna("").astype(str).str.lower().replace("nan", "").tolist()
    true_map = {s: gt_all.get(s, set()) for s in s1_ids}

    index = SqliteReferenceIndex("data/indexes/s2s3.sqlite")
    print(f"index.n={index.n} rss={rss_mb():.1f}", flush=True)
    feat_gen = PairFeatureGenerator({})

    typo_map = defaultdict(list)
    for tok, c in index.conn.execute("SELECT token, c FROM tokc_name"):
        if tok and len(tok) >= 5:
            typo_map[tok[:3]].append((tok, int(c)))

    variants = [
        ("current", {}),
        ("prefix_for_missing_df", {"prefix_missing": True}),
        ("always_addr", {"always_addr": True}),
        ("posting_2000", {"posting_limit": 2000}),
        ("glued_name", {"glued": True}),
        ("indic_addr_only", {"indic_addr": True}),
        ("typo_edit1", {"typo_map": typo_map}),
    ]

    table_rows = []
    current_rows = None
    current_raw = None
    current_feat = None
    current_preds = None
    current_topk_eids = None
    current_retr_eids = None

    def run_from_retrieved(label, rows_list, raw_list, top_k, feat_df=None, preds=None, retrieve_s=0.0):
        nonlocal current_feat, current_preds, current_topk_eids
        t0 = time.perf_counter()
        top_rows = []
        topk_eids = {}
        retr_eids = {}
        n_exp = 0
        max_raw = 0
        sum_raw = 0
        for s1, rows, raw, nm, ad, co in zip(s1_ids, rows_list, raw_list, names, addrs, countries):
            sum_raw += int(raw)
            max_raw = max(max_raw, int(raw), int(rows.size))
            retr_eids[s1] = eids_from_rows(index, rows)
            top = index.cheap_prerank(nm, ad, co, rows, top_k)
            top_rows.append(top)
            topk_eids[s1] = eids_from_rows(index, top)
            n_exp += int(top.size)
        t_rank = time.perf_counter() - t0
        if feat_df is None:
            t1 = time.perf_counter()
            feat_df, preds = score_topk(
                index, feat_gen, model, feature_cols, threshold, s1_ids, names, addrs, countries, top_rows
            )
            t_feat = time.perf_counter() - t1
        else:
            t_feat = 0.0
        rec_pred = {s: set(preds.get(s, set())) for s in s1_ids}
        cats = classify(s1_ids, true_map, rec_pred, retr_eids, topk_eids)
        rec_r, _, _ = link_recall(s1_ids, true_map, retr_eids)
        rec_k, _, _ = link_recall(s1_ids, true_map, topk_eids)
        f05 = evaluate_macro_f05(rec_pred, true_map)["macro_f05"]
        avg_raw = sum_raw / 2000
        avg_exp = n_exp / 2000
        row = {
            "experiment": label,
            "A": cats["A"],
            "B": cats["B"],
            "C": cats["C"],
            "D": cats["D"],
            "E": cats["E"],
            "OK": cats["OK"],
            "cand_recall_retrieve": rec_r,
            "cand_recall_topk": rec_k,
            "macro_f05": f05,
            "avg_candidates": avg_raw,
            "max_candidates": max_raw,
            "avg_expensive": avg_exp,
            "retrieve_s": retrieve_s,
            "rank_s": t_rank,
            "feat_s": t_feat,
            "runtime_s": retrieve_s + t_rank + t_feat,
            "peak_rss_mb": round(max(peak[0], rss_mb()), 1),
            "true_survive_topk": rec_k,
        }
        print(json.dumps({k: row[k] for k in row}, default=float), flush=True)
        table_rows.append(row)
        return feat_df, preds, topk_eids, retr_eids, top_rows

    for name, kwargs in variants:
        print(f"\n=== BLOCK {name} ===", flush=True)
        t0 = time.perf_counter()
        rows_list, raw_list = [], []
        for nm, ad, co in zip(names, addrs, countries):
            rows, raw, _hit = retrieve_variant(index, nm, ad, co, **kwargs)
            rows_list.append(rows)
            raw_list.append(raw)
        retrieve_s = time.perf_counter() - t0
        print(f"retrieve_done {name} {retrieve_s:.1f}s rss={rss_mb():.1f}", flush=True)
        feat_df, preds, topk_eids, retr_eids, _ = run_from_retrieved(
            f"block:{name}", rows_list, raw_list, 30, retrieve_s=retrieve_s
        )
        if name == "current":
            current_rows, current_raw = rows_list, raw_list
            current_feat, current_preds = feat_df, preds
            current_topk_eids, current_retr_eids = topk_eids, retr_eids

    print("\n=== RANKING on current retrieve ===", flush=True)
    for k in (60, 100, 150):
        run_from_retrieved(f"rank:top-{k}", current_rows, current_raw, k, retrieve_s=0.0)

    # Feature analysis C
    print("\n=== FEATURE C/FP ===", flush=True)
    feat = current_feat
    rows_c = []
    if feat is not None and len(feat):
        feat = feat.copy()
        feat["is_true"] = [
            1 if cid in true_map.get(s1, set()) else 0
            for s1, cid in zip(feat["s1_id"], feat["candidate_id"])
        ]
        tp = feat[feat["is_true"] == 1]
        fp = feat[feat["is_true"] == 0]
        focus = fp[(fp["addr_exact"] >= 1.0) & (fp["name_token_set"].between(0.20, 0.34)) & (fp["score"] >= 0.99)]
        cols = [
            "name_exact", "name_token_set", "name_jaccard", "name_levenshtein", "name_token_sort",
            "addr_exact", "addr_token_set", "addr_jaccard", "addr_levenshtein", "country_same",
            "composite_score", "score",
        ]
        dist = {}
        for col in cols:
            if col not in feat.columns:
                continue
            dist[col] = {
                "tp_mean": float(tp[col].mean()) if len(tp) else None,
                "fp_mean": float(fp[col].mean()) if len(fp) else None,
                "tp_p10": float(tp[col].quantile(0.1)) if len(tp) else None,
                "fp_p90": float(fp[col].quantile(0.9)) if len(fp) else None,
                "focus_fp_mean": float(focus[col].mean()) if len(focus) and col in focus else None,
            }
        print(json.dumps({"n_tp_pairs": int(len(tp)), "n_fp_pairs": int(len(fp)), "n_focus_fp": int(len(focus)), "dist": dist}, indent=2), flush=True)
        os.makedirs("eval_train", exist_ok=True)
        with open("eval_train/exp_feature_dist.json", "w") as f:
            json.dump({"n_tp_pairs": int(len(tp)), "n_fp_pairs": int(len(fp)), "n_focus_fp": int(len(focus)), "dist": dist}, f, indent=2)

        for min_tok in (0.40, 0.50, 0.60, 0.70):
            gpreds = apply_guard(feat, threshold, min_name_tok=min_tok)
            rec_pred = {s: set(gpreds.get(s, set())) for s in s1_ids}
            cats = classify(s1_ids, true_map, rec_pred, current_retr_eids, current_topk_eids)
            f05 = evaluate_macro_f05(rec_pred, true_map)["macro_f05"]
            rec_k, _, _ = link_recall(s1_ids, true_map, current_topk_eids)
            row = {
                "experiment": f"guard:addr_exact_requires_name_tok>={min_tok:.2f}",
                "A": cats["A"],
                "B": cats["B"],
                "C": cats["C"],
                "D": cats["D"],
                "E": cats["E"],
                "OK": cats["OK"],
                "cand_recall_retrieve": link_recall(s1_ids, true_map, current_retr_eids)[0],
                "cand_recall_topk": rec_k,
                "macro_f05": f05,
                "avg_candidates": table_rows[0]["avg_candidates"],
                "max_candidates": table_rows[0]["max_candidates"],
                "avg_expensive": table_rows[0]["avg_expensive"],
                "retrieve_s": 0.0,
                "rank_s": 0.0,
                "feat_s": 0.0,
                "runtime_s": 0.0,
                "peak_rss_mb": round(max(peak[0], rss_mb()), 1),
                "true_survive_topk": rec_k,
            }
            print(json.dumps(row, default=float), flush=True)
            table_rows.append(row)
            rows_c.append(row)

    stop.set()
    out = {"peak_rss_mb": round(max(peak[0], rss_mb()), 1), "rows": table_rows}
    with open("eval_train/experiments_2000.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== TABLE ===", flush=True)
    print("Experiment | A | B | C | D | E | CandRecall_top | F0.5 | AvgCand | AvgExp | Runtime | PeakRSS", flush=True)
    for r in table_rows:
        print(
            f"{r['experiment']} | {r['A']} | {r['B']} | {r['C']} | {r['D']} | {r['E']} | "
            f"{r['cand_recall_topk']:.4f} | {r['macro_f05']:.4f} | {r['avg_candidates']:.1f} | "
            f"{r['avg_expensive']:.1f} | {r['runtime_s']:.1f} | {r['peak_rss_mb']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
