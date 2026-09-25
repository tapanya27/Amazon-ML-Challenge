"""One isolated 2000-S1 eval: top-k=100, always-addr, glued-name, FP guards. Not production."""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiments_2000 import (
    apply_guard,
    classify,
    eids_from_rows,
    link_recall,
    load_gt,
    retrieve_variant,
    rss_mb,
    score_topk,
)
from src.blocking.sqlite_index import SqliteReferenceIndex
from src.features.pair_features import PairFeatureGenerator
from src.models.evaluation import evaluate_macro_f05
from src.preprocessing import preprocess_dataframe


def main():
    peak = [rss_mb()]
    stop = threading.Event()

    def loop():
        while not stop.wait(0.4):
            peak[0] = max(peak[0], rss_mb())

    threading.Thread(target=loop, daemon=True).start()
    t_all = time.perf_counter()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    threshold = float(meta.get("best_threshold", 0.92))
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_baseline.json")

    sample = pd.read_csv("eval_train/_sample_s1.tsv", sep="\t")
    s1_p = preprocess_dataframe(sample, "tgt")
    s1_ids = s1_p["entity_id"].tolist()
    assert len(s1_ids) == 2000
    names = s1_p["name_normalized"].fillna("").astype(str).replace("nan", "").tolist()
    addrs = s1_p["address_normalized"].fillna("").astype(str).replace("nan", "").tolist()
    countries = s1_p["country_normalized"].fillna("").astype(str).str.lower().replace("nan", "").tolist()
    gt_all = load_gt(cfg["data"]["train"]["ground_truth"])
    true_map = {s: gt_all.get(s, set()) for s in s1_ids}

    index = SqliteReferenceIndex("data/indexes/s2s3.sqlite")
    print(f"index.n={index.n} (reuse, no rebuild) rss={rss_mb():.1f}", flush=True)
    feat_gen = PairFeatureGenerator({})

    t0 = time.perf_counter()
    rows_list, raw_list = [], []
    sum_raw = max_raw = 0
    for nm, ad, co in zip(names, addrs, countries):
        rows, raw, _ = retrieve_variant(
            index, nm, ad, co,
            retrieve_cap=350,
            always_addr=True,
            glued=True,
            posting_limit=400,
        )
        rows_list.append(rows)
        raw_list.append(raw)
        sum_raw += int(raw)
        max_raw = max(max_raw, int(raw), int(rows.size))
    retrieve_s = time.perf_counter() - t0
    print(f"retrieve_done {retrieve_s:.1f}s avg_raw={sum_raw/2000:.1f}", flush=True)

    t0 = time.perf_counter()
    top_rows, retr_eids, topk_eids = [], {}, {}
    n_exp = 0
    for s1, rows, nm, ad, co in zip(s1_ids, rows_list, names, addrs, countries):
        retr_eids[s1] = eids_from_rows(index, rows)
        top = index.cheap_prerank(nm, ad, co, rows, 100)
        top_rows.append(top)
        topk_eids[s1] = eids_from_rows(index, top)
        n_exp += int(top.size)
    rank_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    feat_df, preds = score_topk(
        index, feat_gen, model, feature_cols, threshold, s1_ids, names, addrs, countries, top_rows
    )
    feat_s = time.perf_counter() - t0

    rec_r, _, _ = link_recall(s1_ids, true_map, retr_eids)
    rec_k, _, _ = link_recall(s1_ids, true_map, topk_eids)

    def pack(label, pred_map):
        rec_pred = {s: set(pred_map.get(s, set())) for s in s1_ids}
        cats = classify(s1_ids, true_map, rec_pred, retr_eids, topk_eids)
        ev = evaluate_macro_f05(rec_pred, true_map)
        return {
            "label": label,
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
            "cand_recall_retrieve": rec_r,
            "cand_recall_topk": rec_k,
            "avg_candidates": sum_raw / 2000,
            "max_candidates": max_raw,
            "avg_expensive": n_exp / 2000,
        }

    results = [pack("no_guard", preds)]
    for t in (0.40, 0.50, 0.55, 0.60, 0.70):
        results.append(pack(f"guard_name_tokset>={t:.2f}", apply_guard(feat_df, threshold, min_name_tok=t)))

    wall = time.perf_counter() - t_all
    stop.set()
    out = {
        "note": "ISOLATED experiment; production pipeline unchanged",
        "index_rows": index.n,
        "n_s1": 2000,
        "retrieve_cap": 350,
        "top_k": 100,
        "always_addr": True,
        "glued_name": True,
        "retrieve_s": retrieve_s,
        "rank_s": rank_s,
        "feat_s": feat_s,
        "runtime_s": wall,
        "peak_rss_mb": round(max(peak[0], rss_mb()), 1),
        "baseline": {
            "macro_f05": 0.6646,
            "cand_recall_retrieve": 0.7978,
            "cand_recall_topk": 0.7550,
            "avg_candidates": 356.4,
            "avg_expensive": 27.4,
            "A": 759, "B": 139, "C": 788, "D": 77, "E": 23,
        },
        "results": results,
    }
    os.makedirs("eval_train", exist_ok=True)
    with open("eval_train/targeted_2000.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
