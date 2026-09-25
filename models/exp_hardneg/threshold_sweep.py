"""Threshold sweep on saved hard-neg model + same 1500 val S1. Isolated."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train_hardneg_exp import (
    classify,
    pattern_fp_stats,
    prefetch_eid_rows,
    retrieve_batch,
)
from run_experiments_2000 import load_gt
from src.blocking.sqlite_index import SqliteReferenceIndex
from src.features.pair_features import PairFeatureGenerator
from src.models.decision import pairs_to_predictions
from src.models.evaluation import evaluate_macro_f05
from src.preprocessing import preprocess_dataframe

# pack_df is nested in main — recreate here
def _pack(dfp):
    ids = dfp["entity_id"].tolist()
    names = dfp["name_normalized"].fillna("").astype(str).replace("nan", "").tolist()
    addrs = dfp["address_normalized"].fillna("").astype(str).replace("nan", "").tolist()
    cos = dfp["country_normalized"].fillna("").astype(str).str.lower().replace("nan", "").tolist()
    return ids, names, addrs, cos


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    eval_ids = set(pd.read_csv("eval_train/_sample_s1.tsv", sep="\t")["entity_id"])
    gt_all = load_gt(cfg["data"]["train"]["ground_truth"])
    all_s1 = pd.read_csv(cfg["data"]["train"]["source1"], sep="\t", usecols=["entity_id"])
    pool = [s for s in all_s1["entity_id"].tolist() if s not in eval_ids]
    rng = np.random.RandomState(777)
    rng.shuffle(pool)
    val_ids = pool[4000:5500]
    assert len(val_ids) == 1500
    assert not (set(val_ids) & eval_ids)

    s1_all = pd.read_csv(cfg["data"]["train"]["source1"], sep="\t")
    val_p = preprocess_dataframe(s1_all[s1_all.entity_id.isin(val_ids)].copy(), "sweep_val")
    val_ids_o, val_nm, val_ad, val_co = _pack(val_p)
    val_true = {s: gt_all.get(s, set()) for s in val_ids_o}

    index = SqliteReferenceIndex("data/indexes/s2s3.sqlite")
    feat_gen = PairFeatureGenerator({})
    model = xgb.XGBClassifier()
    model.load_model("models/exp_hardneg/xgb_hardneg.json")
    print("rebuilding val top-100 scores once (not saved previously)...", flush=True)
    _, val_top, val_retr, val_topk = retrieve_batch(index, val_ids_o, val_nm, val_ad, val_co, 100)
    val_feat = feat_gen.generate_from_index(index, val_ids_o, val_nm, val_ad, val_co, val_top)
    scores = model.predict_proba(val_feat[feature_cols])[:, 1]
    val_feat = val_feat.copy()
    val_feat["_raw_score"] = scores

    needed = set()
    for s in val_ids_o:
        needed |= val_true.get(s, set())
        needed |= set(val_topk.get(s, []))
    rec_lookup = prefetch_eid_rows(index, needed)

    rows = []
    for th in [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]:
        df = val_feat.copy()
        df["score"] = df["_raw_score"]
        exact = (df["name_exact"] >= 1.0) & (df["addr_exact"] >= 1.0) & (df["country_same"] >= 1.0)
        df.loc[exact, "score"] = np.maximum(df.loc[exact, "score"], th)
        preds = pairs_to_predictions(df, threshold=th)
        rec_pred = {s: set(preds.get(s, set())) for s in val_ids_o}
        cats = classify(val_ids_o, val_true, rec_pred, val_retr, val_topk)
        ev = evaluate_macro_f05(rec_pred, val_true)
        n_pred = sum(len(v) for v in rec_pred.values())
        n_fp = sum(len(rec_pred[s] - val_true.get(s, set())) for s in val_ids_o)
        df["is_true"] = [int(cid in val_true.get(s, set())) for s, cid in zip(df["s1_id"], df["candidate_id"])]
        df["selected"] = [int(cid in rec_pred.get(s, set())) for s, cid in zip(df["s1_id"], df["candidate_id"])]
        pat = pattern_fp_stats(df, rec_pred, val_true, rec_lookup)
        rec_k = sum(
            len(val_true[s] & set(val_topk.get(s, []))) for s in val_ids_o if val_true.get(s)
        ) / max(sum(len(val_true[s]) for s in val_ids_o if val_true.get(s)), 1)
        row = {
            "threshold": th,
            "macro_f05": ev["macro_f05"],
            "precision": ev["mean_precision"],
            "recall": ev["mean_recall"],
            "A": cats["A"],
            "B": cats["B"],
            "C": cats["C"],
            "D": cats["D"],
            "E": cats["E"],
            "OK": cats["OK"],
            "fp_links": n_fp,
            "fp_rate": n_fp / n_pred if n_pred else 0.0,
            "top100_recall": rec_k,
            "pred_links": n_pred,
            "fp_branch": pat["branch"]["n_selected_fp"],
            "fp_shared_address": pat["shared_address"]["n_selected_fp"],
            "fp_shared_building": pat["shared_building"]["n_selected_fp"],
            "fp_exact_name_wrong_addr": pat["exact_name_wrong_addr"]["n_selected_fp"],
            "fp_empty_addr_exact_name": pat["empty_addr_exact_name"]["n_selected_fp"],
            "fp_other": pat["other_highscore"]["n_selected_fp"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    best = max(rows, key=lambda r: r["macro_f05"])
    out = {"n_val": 1500, "best_threshold": best["threshold"], "best_macro_f05": best["macro_f05"], "rows": rows}
    os.makedirs("models/exp_hardneg", exist_ok=True)
    with open("models/exp_hardneg/threshold_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("BEST", best["threshold"], best["macro_f05"], flush=True)


if __name__ == "__main__":
    main()
