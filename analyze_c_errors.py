"""C-error feature analysis on the same 2000 S1. Isolated; not production."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiments_2000 import (
    classify,
    eids_from_rows,
    load_gt,
    retrieve_variant,
    score_topk,
)
from src.blocking.sqlite_index import SqliteReferenceIndex
from src.features.pair_features import PairFeatureGenerator
from src.preprocessing import preprocess_dataframe

FEAT_COLS = [
    "name_exact", "name_levenshtein", "name_token_sort", "name_token_set",
    "name_jaccard", "name_overlap", "name_ngram_sim", "name_first_token",
    "name_len_diff", "name_len_ratio", "addr_exact", "addr_levenshtein",
    "addr_token_set", "addr_jaccard", "addr_overlap", "addr_ngram_sim",
    "addr_len_diff", "addr_len_ratio", "addr_num_overlap", "addr_num_ratio",
    "addr_num_conflict", "country_same", "total_jaccard", "total_overlap",
    "name_addr_prod", "token_sort_prod", "composite_score", "score",
]


def cheap_score(q_name, q_addr, q_country, cname, caddr, cco):
    qn = set(q_name.split()) if q_name else set()
    qa = set(q_addr.split()) if q_addr else set()
    q_country = (q_country or "").lower()
    cname = cname or ""
    caddr = caddr or ""
    sc = len(qn & set(cname.split())) * 3 + len(qa & set(caddr.split())) * 4
    if q_name and cname == q_name:
        sc += 40
    if q_addr and caddr == q_addr:
        sc += 40
    if q_country and cco == q_country:
        sc += 5
    return sc


def summarize(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "p10": float(s.quantile(0.1)),
        "p50": float(s.median()),
        "p90": float(s.quantile(0.9)),
    }


def main():
    t0 = time.perf_counter()
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    threshold = float(meta.get("best_threshold", 0.92))
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_baseline.json")

    raw_s1 = pd.read_csv("eval_train/_sample_s1.tsv", sep="\t").set_index("entity_id")
    sample = pd.read_csv("eval_train/_sample_s1.tsv", sep="\t")
    s1_p = preprocess_dataframe(sample, "cana")
    s1_ids = s1_p["entity_id"].tolist()
    names = s1_p["name_normalized"].fillna("").astype(str).replace("nan", "").tolist()
    addrs = s1_p["address_normalized"].fillna("").astype(str).replace("nan", "").tolist()
    countries = s1_p["country_normalized"].fillna("").astype(str).str.lower().replace("nan", "").tolist()
    qmap = {s: (n, a, c) for s, n, a, c in zip(s1_ids, names, addrs, countries)}
    gt_all = load_gt(cfg["data"]["train"]["ground_truth"])
    true_map = {s: gt_all.get(s, set()) for s in s1_ids}

    index = SqliteReferenceIndex("data/indexes/s2s3.sqlite")
    feat_gen = PairFeatureGenerator({})
    print("retrieving (isolated targeted config)...", flush=True)
    rows_list = []
    for nm, ad, co in zip(names, addrs, countries):
        rows, _raw, _ = retrieve_variant(
            index, nm, ad, co, retrieve_cap=350, always_addr=True, glued=True, posting_limit=400
        )
        rows_list.append(rows)

    top_rows, retr_eids, topk_eids, cheap_by = [], {}, {}, {}
    for s1, rows, nm, ad, co in zip(s1_ids, rows_list, names, addrs, countries):
        retr_eids[s1] = eids_from_rows(index, rows)
        recs = {r[0]: r for r in index.fetch_rows(rows.tolist())} if rows.size else {}
        top = index.cheap_prerank(nm, ad, co, rows, 100)
        top_rows.append(top)
        topk_eids[s1] = eids_from_rows(index, top)
        for rid in (top.tolist() if top.size else []):
            rec = recs.get(int(rid))
            if not rec:
                continue
            _rid, eid, cname, caddr, cco = rec
            cheap_by[(s1, eid)] = cheap_score(nm, ad, co, cname or "", caddr or "", cco or "")

    feat_df, preds = score_topk(
        index, feat_gen, model, feature_cols, threshold, s1_ids, names, addrs, countries, top_rows
    )
    rec_pred = {s: set(preds.get(s, set())) for s in s1_ids}
    cats = classify(s1_ids, true_map, rec_pred, retr_eids, topk_eids)
    print(f"cats={cats} pairs={len(feat_df)} elapsed={time.perf_counter()-t0:.0f}s", flush=True)

    rec_lookup = {}
    needed = set()
    for s1 in s1_ids:
        needed |= true_map.get(s1, set())
        needed |= rec_pred.get(s1, set())
        needed |= set(topk_eids.get(s1, []))
    index.conn.execute("CREATE TEMP TABLE need_c (eid TEXT PRIMARY KEY)")
    index.conn.executemany("INSERT OR IGNORE INTO need_c VALUES (?)", [(e,) for e in needed])
    for eid, nm, ad, co in index.conn.execute(
        "SELECT r.entity_id, r.name, r.addr, r.country FROM refs r JOIN need_c n ON n.eid=r.entity_id"
    ):
        rec_lookup[eid] = {"name": nm or "", "addr": ad or "", "country": (co or "").lower()}

    feat_df = feat_df.copy()
    feat_df["is_true"] = [
        int(cid in true_map.get(s1, set())) for s1, cid in zip(feat_df["s1_id"], feat_df["candidate_id"])
    ]
    feat_df["selected"] = [
        int(cid in rec_pred.get(s1, set())) for s1, cid in zip(feat_df["s1_id"], feat_df["candidate_id"])
    ]
    feat_df["cheap"] = [
        cheap_by.get((s1, cid), None) for s1, cid in zip(feat_df["s1_id"], feat_df["candidate_id"])
    ]

    c_s1 = []
    for s1 in s1_ids:
        true = true_map.get(s1, set())
        pred = rec_pred.get(s1, set())
        topk = set(topk_eids.get(s1, []))
        if not true or not pred:
            continue
        fps = pred - true
        tps_in_top = true & topk
        if fps and tps_in_top:
            c_s1.append(s1)

    tp = feat_df[(feat_df["s1_id"].isin(c_s1)) & (feat_df["is_true"] == 1)]
    sel_fp = feat_df[(feat_df["s1_id"].isin(c_s1)) & (feat_df["is_true"] == 0) & (feat_df["selected"] == 1)]
    all_fp_c = feat_df[(feat_df["s1_id"].isin(c_s1)) & (feat_df["is_true"] == 0)]

    dist = {}
    sep = []
    for col in FEAT_COLS + ["cheap"]:
        if col not in tp.columns and col != "cheap":
            continue
        dist[col] = {"true_in_topk": summarize(tp[col]), "selected_fp": summarize(sel_fp[col])}
        tmed = dist[col]["true_in_topk"]["p50"] if dist[col]["true_in_topk"] else None
        fmed = dist[col]["selected_fp"]["p50"] if dist[col]["selected_fp"] else None
        if tmed is None or fmed is None:
            continue
        # overlap: share of selected FPs on the TP-majority side of midpoint
        mid = 0.5 * (tmed + fmed)
        tvals = pd.to_numeric(tp[col], errors="coerce").dropna()
        fvals = pd.to_numeric(sel_fp[col], errors="coerce").dropna()
        if tmed >= fmed:
            tp_ok = float((tvals >= mid).mean()) if len(tvals) else 0
            fp_ok = float((fvals < mid).mean()) if len(fvals) else 0
        else:
            tp_ok = float((tvals <= mid).mean()) if len(tvals) else 0
            fp_ok = float((fvals > mid).mean()) if len(fvals) else 0
        sep.append((abs(tmed - fmed), col, float(tmed), float(fmed), tp_ok, fp_ok, (tp_ok + fp_ok) / 2))
    sep.sort(reverse=True)

    # systematic patterns on selected FPs
    n_fp = max(len(sel_fp), 1)
    patterns = {
        "n_c_s1_with_selected_fp": len(c_s1),
        "n_true_pairs_in_topk": int(len(tp)),
        "n_selected_fp_pairs": int(len(sel_fp)),
        "fp_addr_exact_1": float((sel_fp["addr_exact"] >= 1).mean()) if len(sel_fp) else None,
        "tp_addr_exact_1": float((tp["addr_exact"] >= 1).mean()) if len(tp) else None,
        "fp_name_exact_1": float((sel_fp["name_exact"] >= 1).mean()) if len(sel_fp) else None,
        "tp_name_exact_1": float((tp["name_exact"] >= 1).mean()) if len(tp) else None,
        "fp_country_same": float((sel_fp["country_same"] >= 1).mean()) if len(sel_fp) else None,
        "fp_addr_exact_and_name_tokset_lt_0.5": float(
            ((sel_fp["addr_exact"] >= 1) & (sel_fp["name_token_set"] < 0.5)).mean()
        ) if len(sel_fp) else None,
        "tp_addr_exact_and_name_tokset_lt_0.5": float(
            ((tp["addr_exact"] >= 1) & (tp["name_token_set"] < 0.5)).mean()
        ) if len(tp) else None,
        "fp_name_first_token_0": float((sel_fp["name_first_token"] < 1).mean()) if len(sel_fp) else None,
        "fp_score_ge_0.99": float((sel_fp["score"] >= 0.99).mean()) if len(sel_fp) else None,
        "tp_score_ge_0.99": float((tp["score"] >= 0.99).mean()) if len(tp) else None,
        "fp_score_mean": float(sel_fp["score"].mean()) if len(sel_fp) else None,
        "tp_score_mean": float(tp["score"].mean()) if len(tp) else None,
    }

    # clean separation checks
    clean = {}
    if len(tp) and len(sel_fp):
        for col in ["name_token_set", "name_jaccard", "name_levenshtein", "name_exact",
                    "addr_exact", "composite_score", "score", "name_first_token"]:
            tmax, tmin = float(tp[col].max()), float(tp[col].min())
            fmax, fmin = float(sel_fp[col].max()), float(sel_fp[col].min())
            clean[col] = {
                "tp_range": [tmin, tmax],
                "fp_range": [fmin, fmax],
                "ranges_disjoint": bool(tmax < fmin or fmax < tmin),
            }

    examples = []
    # rank S1 by max FP score
    for s1 in c_s1:
        sub = feat_df[feat_df["s1_id"] == s1]
        fps = sub[(sub["is_true"] == 0) & (sub["selected"] == 1)].sort_values("score", ascending=False)
        tps = sub[sub["is_true"] == 1].sort_values("score", ascending=True)
        if fps.empty or tps.empty:
            continue
        fp_row = fps.iloc[0]
        # true: unselected true if any, else lowest-scoring true
        unsel = tps[tps["selected"] == 0]
        tp_row = unsel.iloc[0] if len(unsel) else tps.iloc[0]
        qn, qa, qc = qmap[s1]
        raw = raw_s1.loc[s1]
        tid, fid = tp_row["candidate_id"], fp_row["candidate_id"]
        trec = rec_lookup.get(tid, {})
        frec = rec_lookup.get(fid, {})
        examples.append({
            "s1_id": s1,
            "s1_raw_name": str(raw.get("business_name", "")),
            "s1_raw_addr": str(raw.get("business_address", "")),
            "s1_raw_country": str(raw.get("country", "")),
            "s1_norm_name": qn,
            "s1_norm_addr": qa,
            "true_id": tid,
            "true_selected": bool(tp_row["selected"]),
            "true_name": trec.get("name", ""),
            "true_addr": trec.get("addr", ""),
            "true_country": trec.get("country", ""),
            "false_id": fid,
            "false_name": frec.get("name", ""),
            "false_addr": frec.get("addr", ""),
            "false_country": frec.get("country", ""),
            "true_score": float(tp_row["score"]),
            "false_score": float(fp_row["score"]),
            "true_cheap": cheap_by.get((s1, tid)),
            "false_cheap": cheap_by.get((s1, fid)),
            "true_feats": {c: float(tp_row[c]) for c in FEAT_COLS if c in tp_row.index},
            "false_feats": {c: float(fp_row[c]) for c in FEAT_COLS if c in fp_row.index},
            "pattern": (
                "shared_address" if float(fp_row["addr_exact"]) >= 1 and float(fp_row["name_exact"]) < 1
                else "same_name_diff_addr" if float(fp_row["name_exact"]) >= 1 and float(fp_row["addr_exact"]) < 1
                else "weak_name_strong_addr" if float(fp_row["addr_token_set"]) >= 0.8 and float(fp_row["name_token_set"]) < 0.5
                else "other"
            ),
        })
    examples.sort(key=lambda x: x["false_score"] - x["true_score"], reverse=True)
    examples = examples[:30]

    pat_counts = {}
    for ex in examples:
        pat_counts[ex["pattern"]] = pat_counts.get(ex["pattern"], 0) + 1
    # full selected-FP pattern counts
    full_pat = {"shared_address": 0, "same_name_diff_addr": 0, "weak_name_strong_addr": 0, "other": 0}
    if len(sel_fp):
        full_pat["shared_address"] = int(((sel_fp["addr_exact"] >= 1) & (sel_fp["name_exact"] < 1)).sum())
        full_pat["same_name_diff_addr"] = int(((sel_fp["name_exact"] >= 1) & (sel_fp["addr_exact"] < 1)).sum())
        full_pat["weak_name_strong_addr"] = int(
            ((sel_fp["addr_token_set"] >= 0.8) & (sel_fp["name_token_set"] < 0.5) & (sel_fp["addr_exact"] < 1)).sum()
        )
        full_pat["other"] = int(len(sel_fp)) - full_pat["shared_address"] - full_pat["same_name_diff_addr"] - full_pat["weak_name_strong_addr"]

    out = {
        "n_s1": 2000,
        "classify": cats,
        "n_c_s1_true_in_top100_and_fp_selected": len(c_s1),
        "distributions": dist,
        "top_separators": [
            {"feature": c, "abs_median_gap": g, "tp_median": tm, "fp_median": fm,
             "frac_tp_on_own_side_of_mid": tpo, "frac_fp_on_own_side_of_mid": fpo, "mean_side_hit": m}
            for g, c, tm, fm, tpo, fpo, m in sep[:15]
        ],
        "clean_separation": clean,
        "patterns_selected_fp": full_pat,
        "pattern_counts_in_examples": pat_counts,
        "summary_rates": patterns,
        "examples": examples[:20],
        "elapsed_s": time.perf_counter() - t0,
    }
    os.makedirs("eval_train", exist_ok=True)
    with open("eval_train/c_error_feature_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in [
        "classify", "n_c_s1_true_in_top100_and_fp_selected", "top_separators",
        "clean_separation", "patterns_selected_fp", "summary_rates",
    ]}, indent=2), flush=True)
    print("wrote eval_train/c_error_feature_analysis.json examples", len(examples), flush=True)


if __name__ == "__main__":
    main()
