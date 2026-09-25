"""Isolated hard-negative XGBoost experiment. Does not overwrite production model/index."""
from __future__ import annotations

import json
import os
import random
import re
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
    link_recall,
    load_gt,
    retrieve_variant,
    score_topk,
)
from src.blocking.sqlite_index import SqliteReferenceIndex
from src.features.pair_features import PairFeatureGenerator
from src.models.decision import pairs_to_predictions
from src.models.evaluation import evaluate_macro_f05
from src.preprocessing import preprocess_dataframe

BRANCH_RE = re.compile(
    r"\b(center|southside|riverside|central|metro|uptown|downtown|group|hldgs|holdings|exports)\b",
    re.I,
)
INDIC_RE = re.compile(r"[\u0900-\u0d7f]")
LATIN_RE = re.compile(r"[a-z]")


def tag_fp(q_name, t_name, t_addr, feats) -> str:
    ne = float(feats.get("name_exact", 0) or 0)
    ae = float(feats.get("addr_exact", 0) or 0)
    nts = float(feats.get("name_token_set", 0) or 0)
    ats = float(feats.get("addr_token_set", 0) or 0)
    if ne >= 1 and not (t_addr or "").strip():
        return "empty_addr_exact_name"
    if ne >= 1 and ae < 1:
        return "exact_name_wrong_addr"
    if ae >= 1 and ne < 1:
        return "shared_address"
    if ats >= 0.8 and nts < 0.5:
        return "shared_building"
    blob = f"{q_name} {t_name}"
    if BRANCH_RE.search(blob or ""):
        return "branch"
    return "other_highscore"


def prefetch_eid_rows(index, eids):
    if not eids:
        return {}
    index.conn.execute("DROP TABLE IF EXISTS tmp_hn_eid")
    index.conn.execute("CREATE TEMP TABLE tmp_hn_eid (eid TEXT PRIMARY KEY)")
    index.conn.executemany("INSERT OR IGNORE INTO tmp_hn_eid VALUES (?)", [(e,) for e in eids])
    out = {}
    for eid, rid, nm, ad, co in index.conn.execute(
        "SELECT r.entity_id, r.row_id, r.name, r.addr, r.country "
        "FROM refs r JOIN tmp_hn_eid t ON t.eid=r.entity_id"
    ):
        out[eid] = (int(rid), nm or "", ad or "", (co or "").lower())
    return out


def retrieve_batch(index, s1_ids, names, addrs, countries, top_k=100):
    rows_list, top_rows, retr_eids, topk_eids = [], [], {}, {}
    for s1, nm, ad, co in zip(s1_ids, names, addrs, countries):
        rows, _raw, _ = retrieve_variant(
            index, nm, ad, co, retrieve_cap=350, always_addr=True, glued=True, posting_limit=400
        )
        rows_list.append(rows)
        top = index.cheap_prerank(nm, ad, co, rows, top_k)
        top_rows.append(top)
        retr_eids[s1] = eids_from_rows(index, rows)
        topk_eids[s1] = eids_from_rows(index, top)
    return rows_list, top_rows, retr_eids, topk_eids


def eval_model(model, feat_df, feature_cols, threshold, s1_ids, true_map, retr_eids, topk_eids, use_fast_path=True):
    df = feat_df.copy()
    df["score"] = model.predict_proba(df[feature_cols])[:, 1]
    if use_fast_path and {"name_exact", "addr_exact", "country_same"} <= set(df.columns):
        exact = (df["name_exact"] >= 1.0) & (df["addr_exact"] >= 1.0) & (df["country_same"] >= 1.0)
        df.loc[exact, "score"] = np.maximum(df.loc[exact, "score"], threshold)
    preds = pairs_to_predictions(df, threshold=threshold)
    rec_pred = {s: set(preds.get(s, set())) for s in s1_ids}
    cats = classify(s1_ids, true_map, rec_pred, retr_eids, topk_eids)
    ev = evaluate_macro_f05(rec_pred, true_map)
    rec_k, _, _ = link_recall(s1_ids, true_map, topk_eids)
    rec_r, _, _ = link_recall(s1_ids, true_map, retr_eids)
    n_pred = sum(len(v) for v in rec_pred.values())
    n_fp = 0
    n_tp = 0
    for s, pred in rec_pred.items():
        true = true_map.get(s, set())
        n_fp += len(pred - true)
        n_tp += len(pred & true)
    df["is_true"] = [int(cid in true_map.get(s, set())) for s, cid in zip(df["s1_id"], df["candidate_id"])]
    df["selected"] = [int(cid in rec_pred.get(s, set())) for s, cid in zip(df["s1_id"], df["candidate_id"])]
    tp_sc = df.loc[df["is_true"] == 1, "score"]
    fp_sel = df.loc[(df["is_true"] == 0) & (df["selected"] == 1), "score"]
    return {
        "macro_f05": ev["macro_f05"],
        "precision": ev["mean_precision"],
        "recall": ev["mean_recall"],
        "singleton_accuracy": ev["singleton_accuracy"],
        "A": cats["A"],
        "B": cats["B"],
        "C": cats["C"],
        "D": cats["D"],
        "E": cats["E"],
        "OK": cats["OK"],
        "top100_recall": rec_k,
        "retrieve_recall": rec_r,
        "pred_links": n_pred,
        "tp_links": n_tp,
        "fp_links": n_fp,
        "fp_rate": n_fp / n_pred if n_pred else 0.0,
        "score_true_mean": float(tp_sc.mean()) if len(tp_sc) else None,
        "score_true_p50": float(tp_sc.median()) if len(tp_sc) else None,
        "score_sel_fp_mean": float(fp_sel.mean()) if len(fp_sel) else None,
        "score_sel_fp_p50": float(fp_sel.median()) if len(fp_sel) else None,
        "score_sel_fp_ge_0.99": float((fp_sel >= 0.99).mean()) if len(fp_sel) else None,
    }, df, rec_pred


def pattern_fp_stats(feat_df, rec_pred, true_map, rec_lookup):
    out = {k: {"n_selected_fp": 0} for k in [
        "branch", "shared_address", "shared_building", "exact_name_wrong_addr",
        "empty_addr_exact_name", "other_highscore",
    ]}
    sel_fp = feat_df[(feat_df["is_true"] == 0) & (feat_df["selected"] == 1)]
    for row in sel_fp.itertuples():
        rec = rec_lookup.get(row.candidate_id, ("", "", "", ""))
        feats = row._asdict() if hasattr(row, "_asdict") else {}
        # pandas namedtuple
        qn = ""
        tag = tag_fp("", rec[1] if len(rec) > 1 else "", rec[2] if len(rec) > 2 else "", {
            "name_exact": getattr(row, "name_exact", 0),
            "addr_exact": getattr(row, "addr_exact", 0),
            "name_token_set": getattr(row, "name_token_set", 0),
            "addr_token_set": getattr(row, "addr_token_set", 0),
        })
        if BRANCH_RE.search(str(getattr(row, "candidate_id", "")) + " " + rec[1]):
            if tag == "other_highscore":
                tag = "branch"
        out[tag]["n_selected_fp"] = out.get(tag, {"n_selected_fp": 0})["n_selected_fp"] + 1
        if tag not in out:
            out[tag] = {"n_selected_fp": 1}
    n = max(len(sel_fp), 1)
    for k, v in list(out.items()):
        v["share_of_selected_fp"] = v["n_selected_fp"] / n
    out["n_selected_fp_total"] = int(len(sel_fp))
    return out


def main():
    t0 = time.perf_counter()
    os.makedirs("models/exp_hardneg", exist_ok=True)
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    prod_th = float(meta.get("best_threshold", 0.92))

    eval_ids = set(pd.read_csv("eval_train/_sample_s1.tsv", sep="\t")["entity_id"])
    gt_all = load_gt(cfg["data"]["train"]["ground_truth"])
    all_s1 = pd.read_csv(cfg["data"]["train"]["source1"], sep="\t", usecols=["entity_id"])
    pool = [s for s in all_s1["entity_id"].tolist() if s not in eval_ids]
    rng = np.random.RandomState(777)
    rng.shuffle(pool)
    n_mine, n_val = 4000, 1500
    mine_ids = pool[:n_mine]
    val_ids = pool[n_mine:n_mine + n_val]
    assert not (set(mine_ids) & set(val_ids) & eval_ids)
    assert not (set(mine_ids) & set(val_ids))
    print(f"excluded_eval2000={len(eval_ids)} mine={len(mine_ids)} val={len(val_ids)}", flush=True)

    s1_all = pd.read_csv(cfg["data"]["train"]["source1"], sep="\t")
    mine_raw = s1_all[s1_all.entity_id.isin(mine_ids)].copy()
    val_raw = s1_all[s1_all.entity_id.isin(val_ids)].copy()
    mine_p = preprocess_dataframe(mine_raw, "hn_mine")
    val_p = preprocess_dataframe(val_raw, "hn_val")

    prod = xgb.XGBClassifier()
    prod.load_model("models/xgb_baseline.json")
    index = SqliteReferenceIndex("data/indexes/s2s3.sqlite")
    feat_gen = PairFeatureGenerator({})
    print(f"index.n={index.n}", flush=True)

    def pack_df(dfp):
        ids = dfp["entity_id"].tolist()
        names = dfp["name_normalized"].fillna("").astype(str).replace("nan", "").tolist()
        addrs = dfp["address_normalized"].fillna("").astype(str).replace("nan", "").tolist()
        cos = dfp["country_normalized"].fillna("").astype(str).str.lower().replace("nan", "").tolist()
        return ids, names, addrs, cos

    mine_ids_o, mine_nm, mine_ad, mine_co = pack_df(mine_p)
    val_ids_o, val_nm, val_ad, val_co = pack_df(val_p)
    mine_true = {s: gt_all.get(s, set()) for s in mine_ids_o}
    val_true = {s: gt_all.get(s, set()) for s in val_ids_o}

    print("mining retrieve on train S1 only...", flush=True)
    t1 = time.perf_counter()
    _, mine_top, mine_retr, mine_topk = retrieve_batch(index, mine_ids_o, mine_nm, mine_ad, mine_co, 100)
    print(f"mine retrieve {time.perf_counter()-t1:.1f}s", flush=True)
    mine_feat, mine_preds = score_topk(
        index, feat_gen, prod, feature_cols, prod_th, mine_ids_o, mine_nm, mine_ad, mine_co, mine_top
    )
    mine_feat = mine_feat.copy()
    mine_feat["score"] = prod.predict_proba(mine_feat[feature_cols])[:, 1]
    mine_pred = {s: set(mine_preds.get(s, set())) for s in mine_ids_o}

    needed = set()
    for s in mine_ids_o:
        needed |= mine_true.get(s, set())
        needed |= set(mine_topk.get(s, []))
    rec_lookup = prefetch_eid_rows(index, needed)
    print(f"prefetched {len(rec_lookup)} ref rows", flush=True)

    rng2 = random.Random(42)
    pos_pairs, orig_neg, hard_neg = [], [], []
    hard_by_tag = {}
    qname = {s: n for s, n in zip(mine_ids_o, mine_nm)}
    feat_ix = mine_feat.set_index(["s1_id", "candidate_id"], drop=False)

    for s1, top in zip(mine_ids_o, mine_top):
        true = mine_true.get(s1, set())
        top_eids = mine_topk.get(s1, [])
        selected = mine_pred.get(s1, set())
        for tid in true:
            pos_pairs.append((s1, tid))
        fps = []
        for eid in top_eids:
            if eid in true:
                continue
            if eid in selected:
                rec = rec_lookup.get(eid, (None, "", "", ""))
                try:
                    row = feat_ix.loc[(s1, eid)]
                    feats = row if isinstance(row, pd.Series) else row.iloc[0]
                    fd = {c: float(feats[c]) for c in feature_cols if c in feats.index}
                except KeyError:
                    fd = {}
                tag = tag_fp(qname[s1], rec[1], rec[2], fd)
                if BRANCH_RE.search(f"{qname[s1]} {rec[1]}"):
                    tag = "branch"
                fps.append((eid, tag, float(fd.get("score", 1.0))))
        prio = {
            "empty_addr_exact_name": 0,
            "exact_name_wrong_addr": 1,
            "shared_address": 2,
            "shared_building": 3,
            "branch": 4,
            "other_highscore": 5,
        }
        fps.sort(key=lambda x: (prio.get(x[1], 9), -x[2]))
        for eid, tag, _sc in fps:
            hard_neg.append((s1, eid, tag))
            hard_by_tag[tag] = hard_by_tag.get(tag, 0) + 1
        easy = [e for e in top_eids if e not in true and e not in selected]
        if len(easy) > 40:
            easy = rng2.sample(easy, 40)
        for eid in easy:
            orig_neg.append((s1, eid))

    # positives not in top-k still included via pos_pairs
    seen = set()
    train_pairs = []
    pair_kind = []
    for p in pos_pairs:
        if p not in seen:
            seen.add(p)
            train_pairs.append(p)
            pair_kind.append("pos")
    n_pos = len(train_pairs)
    for p in orig_neg:
        if p not in seen:
            seen.add(p)
            train_pairs.append(p)
            pair_kind.append("orig_neg")
    n_orig = sum(1 for k in pair_kind if k == "orig_neg")
    for s1, eid, tag in hard_neg:
        p = (s1, eid)
        if p not in seen:
            seen.add(p)
            train_pairs.append(p)
            pair_kind.append("hard_neg")
    n_hard = sum(1 for k in pair_kind if k == "hard_neg")
    print(
        f"pairs pos={n_pos} orig_neg={n_orig} hard_neg={n_hard} ratio_hard/pos={n_hard/max(n_pos,1):.3f} tags={hard_by_tag}",
        flush=True,
    )

    # Build row lists aligned to generate_from_index: group by s1
    from collections import defaultdict as dd
    rows_by_s1 = dd(list)
    missing_gt = 0
    for s1, cid in train_pairs:
        rec = rec_lookup.get(cid)
        if rec is None:
            recs2 = prefetch_eid_rows(index, {cid})
            rec_lookup.update(recs2)
            rec = rec_lookup.get(cid)
        if rec is None:
            missing_gt += 1
            continue
        rows_by_s1[s1].append(rec[0])
    print(f"missing_ref_rows={missing_gt}", flush=True)

    s1_order, names_o, addrs_o, cos_o, row_lists = [], [], [], [], []
    qmap = {s: (n, a, c) for s, n, a, c in zip(mine_ids_o, mine_nm, mine_ad, mine_co)}
    for s1 in mine_ids_o:
        if s1 not in rows_by_s1:
            continue
        s1_order.append(s1)
        n, a, c = qmap[s1]
        names_o.append(n)
        addrs_o.append(a)
        cos_o.append(c)
        row_lists.append(np.asarray(rows_by_s1[s1], dtype=np.int32))

    train_feat = feat_gen.generate_from_index(index, s1_order, names_o, addrs_o, cos_o, row_lists)
    labels = [1 if cid in mine_true.get(s1, set()) else 0 for s1, cid in zip(train_feat["s1_id"], train_feat["candidate_id"])]
    train_feat = train_feat.copy()
    train_feat["label"] = labels
    print(f"train_feat_rows={len(train_feat)} pos={int(sum(labels))} rate={np.mean(labels):.4f}", flush=True)

    unique = np.array(sorted(set(train_feat["s1_id"])))
    rng.shuffle(unique)
    cut = int(len(unique) * 0.85)
    fit_s1 = set(unique[:cut])
    stop_s1 = set(unique[cut:])
    fit_m = train_feat["s1_id"].isin(fit_s1)
    stop_m = train_feat["s1_id"].isin(stop_s1)
    X_fit, y_fit = train_feat.loc[fit_m, feature_cols], train_feat.loc[fit_m, "label"]
    X_stop, y_stop = train_feat.loc[stop_m, feature_cols], train_feat.loc[stop_m, "label"]
    n_neg = int((train_feat["label"] == 0).sum())
    n_pos_f = int((train_feat["label"] == 1).sum())
    spw = n_neg / max(n_pos_f, 1)

    exp_model = xgb.XGBClassifier(
        n_estimators=450,
        max_depth=7,
        learning_rate=0.04,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.05,
        reg_alpha=0.3,
        reg_lambda=1.2,
        scale_pos_weight=min(spw, 8.0),
        tree_method="hist",
        early_stopping_rounds=35,
        eval_metric="aucpr",
        random_state=42,
    )
    print(f"fitting isolated xgb scale_pos_weight={min(spw,8):.2f} fit={len(X_fit)} stop={len(X_stop)}", flush=True)
    exp_model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], verbose=50)
    exp_model.save_model("models/exp_hardneg/xgb_hardneg.json")

    print("val retrieve (never used for hard negatives)...", flush=True)
    t1 = time.perf_counter()
    _, val_top, val_retr, val_topk = retrieve_batch(index, val_ids_o, val_nm, val_ad, val_co, 100)
    print(f"val retrieve {time.perf_counter()-t1:.1f}s", flush=True)
    val_feat, _ = score_topk(
        index, feat_gen, prod, feature_cols, prod_th, val_ids_o, val_nm, val_ad, val_co, val_top
    )
    val_needed = set()
    for s in val_ids_o:
        val_needed |= val_true.get(s, set())
        val_needed |= set(val_topk.get(s, []))
    val_lookup = prefetch_eid_rows(index, val_needed)

    prod_metrics, prod_df, _ = eval_model(prod, val_feat, feature_cols, prod_th, val_ids_o, val_true, val_retr, val_topk)
    exp_metrics_092, exp_df_092, _ = eval_model(exp_model, val_feat, feature_cols, prod_th, val_ids_o, val_true, val_retr, val_topk)

    best_th, best_f = prod_th, -1
    th_curve = []
    for th in np.arange(0.50, 0.96, 0.04):
        m, _, _ = eval_model(exp_model, val_feat, feature_cols, float(th), val_ids_o, val_true, val_retr, val_topk)
        th_curve.append({"th": float(th), "f05": m["macro_f05"], "p": m["precision"], "r": m["recall"]})
        if m["macro_f05"] > best_f:
            best_f, best_th = m["macro_f05"], float(th)
    exp_best, exp_df_best, _ = eval_model(exp_model, val_feat, feature_cols, best_th, val_ids_o, val_true, val_retr, val_topk)

    prod_pat = pattern_fp_stats(prod_df, None, val_true, {k: v for k, v in val_lookup.items()})
    exp_pat = pattern_fp_stats(exp_df_best, None, val_true, {k: v for k, v in val_lookup.items()})

    # Indic diagnostic on val true pairs in top-100
    n_true_topk = n_script = n_zero_name = n_script_zero = 0
    qmapv = {s: (n, a, c) for s, n, a, c in zip(val_ids_o, val_nm, val_ad, val_co)}
    for row in val_feat.itertuples():
        true = val_true.get(row.s1_id, set())
        if row.candidate_id not in true:
            continue
        n_true_topk += 1
        qn = qmapv[row.s1_id][0]
        tn = val_lookup.get(row.candidate_id, (0, "", "", ""))[1]
        script = bool((LATIN_RE.search(qn or "") and INDIC_RE.search(tn or "")) or (INDIC_RE.search(qn or "") and LATIN_RE.search(tn or "")))
        zero = float(row.name_token_set) == 0 and float(row.name_jaccard) == 0 and float(row.name_exact) == 0
        if script:
            n_script += 1
        if zero:
            n_zero_name += 1
        if script and zero:
            n_script_zero += 1
    n_true_all = sum(len(val_true[s]) for s in val_ids_o)
    indic = {
        "val_true_links": n_true_all,
        "true_pairs_scored_in_top100": n_true_topk,
        "script_mismatch_in_scored_true": n_script,
        "zero_name_sim_in_scored_true": n_zero_name,
        "script_mismatch_and_zero_name_sim": n_script_zero,
        "frac_scored_true_script_and_zero_name": n_script_zero / n_true_topk if n_true_topk else None,
    }

    out = {
        "note": "ISOLATED hard-neg experiment; production model/index unchanged",
        "excluded_eval2000": len(eval_ids),
        "n_mine_s1": len(mine_ids_o),
        "n_val_s1": len(val_ids_o),
        "class_balance": {
            "positives": n_pos,
            "original_negatives": n_orig,
            "hard_negatives": n_hard,
            "hard_neg_ratio_vs_pos": n_hard / max(n_pos, 1),
            "hard_neg_by_tag": hard_by_tag,
            "train_feat_rows": int(len(train_feat)),
            "train_feat_pos": n_pos_f,
        },
        "production_val_th0.92": prod_metrics,
        "hardneg_val_th0.92": exp_metrics_092,
        "hardneg_val_best_th": {"threshold": best_th, **exp_best},
        "threshold_curve": th_curve,
        "pattern_fp_production": prod_pat,
        "pattern_fp_hardneg_best": exp_pat,
        "indic_diagnostic": indic,
        "elapsed_s": time.perf_counter() - t0,
        "model_path": "models/exp_hardneg/xgb_hardneg.json",
    }
    with open("models/exp_hardneg/report.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
