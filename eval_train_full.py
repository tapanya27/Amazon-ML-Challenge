"""Evaluate fast pipeline on a reproducible train S1 sample vs FULL train S2+S3."""
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.evaluation import compute_entity_f05, evaluate_macro_f05
from src.pipeline import run_pipeline


def _rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def load_gt(path):
    gt_df = pd.read_csv(path, sep="\t")
    gt = {}
    for s1, ms in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if pd.isna(ms) or not str(ms).strip():
            gt[s1] = set()
        else:
            gt[s1] = {x.strip() for x in str(ms).split(",") if x.strip()}
    return gt


def main():
    n = int(os.environ.get("EVAL_N", "2000"))
    seed = 2026
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    gt_all = load_gt(cfg["data"]["train"]["ground_truth"])
    s1 = pd.read_csv(cfg["data"]["train"]["source1"], sep="\t", usecols=["entity_id"])
    rng = np.random.RandomState(seed)
    chosen = set(rng.choice(s1["entity_id"].values, size=min(n, len(s1)), replace=False))
    tmp_s1 = "eval_train/_sample_s1.tsv"
    os.makedirs("eval_train", exist_ok=True)
    full_s1 = pd.read_csv(cfg["data"]["train"]["source1"], sep="\t")
    full_s1[full_s1.entity_id.isin(chosen)].to_csv(tmp_s1, sep="\t", index=False)

    peak = [_rss_mb()]
    stop = threading.Event()

    def _rss_loop():
        while not stop.wait(0.4):
            peak[0] = max(peak[0], _rss_mb())

    th = threading.Thread(target=_rss_loop, daemon=True)
    th.start()
    t_eval = time.time()
    match_path, _, stats = run_pipeline(
        s1_path=tmp_s1,
        s2_path=cfg["data"]["train"]["source2"],
        s3_path=cfg["data"]["train"]["source3"],
        output_dir="eval_train",
        sample_size=None,
        top_k=30,
        write_candidates=True,
        index_path="data/indexes/s2s3.sqlite",
        rebuild_index=False,
    )
    pred_df = pd.read_csv(match_path, sep="\t")
    preds = {}
    for s1_id, ms in zip(pred_df["source1_entity_id"], pred_df["matched_entity_ids"]):
        if pd.isna(ms) or not str(ms).strip():
            preds[s1_id] = set()
        else:
            preds[s1_id] = {x.strip() for x in str(ms).split(",") if x.strip()}
    gt = {k: gt_all.get(k, set()) for k in preds}
    res = evaluate_macro_f05(preds, gt)
    tp = fp = fn = 0
    n0 = n1 = 0
    for sid, true in gt.items():
        pr = preds.get(sid, set())
        tp += len(pr & true)
        fp += len(pr - true)
        fn += len(true - pr)
        sc = compute_entity_f05(pr, true)
        if sc == 0:
            n0 += 1
        if sc == 1:
            n1 += 1
    cand_df = pd.read_csv("eval_train/candidate_pairs.tsv", sep="\t")
    found = tot = 0
    for s1_id, cs in zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"]):
        true = gt.get(s1_id, set())
        if not true:
            continue
        cset = set() if pd.isna(cs) or not str(cs).strip() else {x.strip() for x in str(cs).split(",") if x.strip()}
        tot += len(true)
        found += len(true & cset)
    out = {
        "note": "LOCAL TRAIN F0.5 — not official test",
        "n_s1": len(gt),
        "seed": seed,
        "macro_f05": res["macro_f05"],
        "mean_precision": res["mean_precision"],
        "mean_recall": res["mean_recall"],
        "singleton_accuracy": res["singleton_accuracy"],
        "tp_links": tp,
        "fp_links": fp,
        "fn_links": fn,
        "pct_f05_0": n0 / len(gt),
        "pct_f05_1": n1 / len(gt),
        "candidate_link_recall": found / tot if tot else 1.0,
        "pipeline_stats": stats,
        "peak_rss_mb": round(max(peak[0], _rss_mb()), 1),
        "eval_wall_s": round(time.time() - t_eval, 1),
    }
    stop.set()
    print(json.dumps(out, indent=2))
    with open("eval_train/local_train_eval_report.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
