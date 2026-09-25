"""
Validation evaluation: blocking recall, pair metrics, and macro F0.5 on held-out S1 entities.
"""

import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blocking.candidate_generation import (
    candidate_recall_stats,
    candidates_to_pairs,
    rank_and_cap_candidates,
    run_blocking,
)
from src.features.pair_features import PairFeatureGenerator
from src.models.decision import pairs_to_predictions
from src.models.evaluation import evaluate_macro_f05
from src.preprocessing import preprocess_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("validation")


def load_val_split(config, val_size=4000, seed=999):
    gt_df = pd.read_csv(config["data"]["train"]["ground_truth"], sep="\t")
    rng = np.random.RandomState(seed)
    val_s1 = set(rng.choice(gt_df["source1_entity_id"].values, size=val_size, replace=False))

    gt_map = {}
    needed_s2, needed_s3 = set(), set()
    gt_sub = gt_df[gt_df["source1_entity_id"].isin(val_s1)]
    for s1_id, match_str in zip(gt_sub["source1_entity_id"], gt_sub["matched_entity_ids"]):
        if pd.isna(match_str) or not str(match_str).strip():
            gt_map[s1_id] = set()
        else:
            matches = {m.strip() for m in str(match_str).split(",") if m.strip()}
            gt_map[s1_id] = matches
            for m in matches:
                if m.startswith("S2-"):
                    needed_s2.add(m)
                elif m.startswith("S3-"):
                    needed_s3.add(m)
    return gt_map, val_s1, needed_s2, needed_s3


def load_target_pool(config, needed_s2, needed_s3, background_rows=25000):
    s2_records, s3_records = [], []
    bg2 = bg3 = 0
    for chunk in pd.read_csv(config["data"]["train"]["source2"], sep="\t", chunksize=250000):
        hit = chunk[chunk["entity_id"].isin(needed_s2)]
        if len(hit):
            s2_records.append(hit)
        if bg2 < background_rows:
            bg = chunk[~chunk["entity_id"].isin(needed_s2)].head(background_rows - bg2)
            s2_records.append(bg)
            bg2 += len(bg)
    for chunk in pd.read_csv(config["data"]["train"]["source3"], sep="\t", chunksize=250000):
        hit = chunk[chunk["entity_id"].isin(needed_s3)]
        if len(hit):
            s3_records.append(hit)
        if bg3 < background_rows:
            bg = chunk[~chunk["entity_id"].isin(needed_s3)].head(background_rows - bg3)
            s3_records.append(bg)
            bg3 += len(bg)
    s2_df = pd.concat(s2_records, ignore_index=True).drop_duplicates(subset=["entity_id"])
    s3_df = pd.concat(s3_records, ignore_index=True).drop_duplicates(subset=["entity_id"])
    return pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)


def run_validation(val_size=4000, max_cands_per_query=300, threshold=None):
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = xgb.XGBClassifier()
    model.load_model("models/xgb_baseline.json")
    with open("models/model_metadata.json", "r") as f:
        meta = json.load(f)
    if threshold is None:
        threshold = float(meta.get("best_threshold", 0.56))
    feature_cols = meta["feature_cols"]

    gt_map, val_s1, needed_s2, needed_s3 = load_val_split(config, val_size=val_size)
    num_matched = sum(1 for v in gt_map.values() if v)
    num_singletons = len(gt_map) - num_matched
    logger.info("Held-out validation: %d S1 (%d matched, %d singletons)", len(gt_map), num_matched, num_singletons)

    s1_all = pd.read_csv(config["data"]["train"]["source1"], sep="\t")
    s1_df = s1_all[s1_all["entity_id"].isin(val_s1)].copy().reset_index(drop=True)
    target_df = load_target_pool(config, needed_s2, needed_s3)

    s1_p = preprocess_dataframe(s1_df, "Val_S1")
    target_p = preprocess_dataframe(target_df, "Val_Target")

    raw_cands = run_blocking(s1_p, target_p, config=config)
    recall_raw = candidate_recall_stats(raw_cands, gt_map)

    capped = rank_and_cap_candidates(
        s1_p, target_p, raw_cands, max_per_query=max_cands_per_query, inject_matches=None
    )
    recall_cap = candidate_recall_stats(capped, gt_map)

    candidate_pairs = candidates_to_pairs(capped)
    logger.info("Scoring %d candidate pairs (max %d / S1)", len(candidate_pairs), max_cands_per_query)

    feat_gen = PairFeatureGenerator(config={})
    feat_df = feat_gen.generate_features(s1_p, target_p, candidate_pairs)
    feat_df["score"] = model.predict_proba(feat_df[feature_cols])[:, 1]

    labels = [
        1 if cid in gt_map.get(sid, set()) else 0
        for sid, cid in zip(feat_df["s1_id"], feat_df["candidate_id"])
    ]
    feat_df["label"] = labels
    y_true = np.array(labels)
    y_score = feat_df["score"].values
    y_pred = (y_score >= threshold).astype(int)

    pair_prec = precision_score(y_true, y_pred, zero_division=0)
    pair_rec = recall_score(y_true, y_pred, zero_division=0)
    pair_f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_score) if y_true.sum() else 0.0

    predictions = pairs_to_predictions(feat_df, threshold=threshold)
    results = evaluate_macro_f05(predictions, gt_map)

    pred_links = sum(len(v) for v in predictions.values())
    true_links = sum(len(v) for v in gt_map.values() if v)
    singleton_gt = {k for k, v in gt_map.items() if not v}
    correct_singleton = sum(1 for s in singleton_gt if not predictions.get(s, set()))
    false_singleton = sum(1 for s in singleton_gt if predictions.get(s, set()))

    print("\nCurrent Validation Results")
    print("--------------------------")
    print(f"Precision:              {results['mean_precision']:.4f}")
    print(f"Recall:                 {results['mean_recall']:.4f}")
    print(f"F0.5:                   {results['macro_f05']:.4f}")
    print(f"Number of S1 entities:  {results['num_entities']}")
    print(f"Number of predicted matches (links): {pred_links}")
    print(f"Number of true matches (links):      {true_links}")
    print(f"Correct singleton predictions:       {correct_singleton}")
    print(f"False singleton matches:             {false_singleton}")

    print("\nPair-level metrics (candidate pairs)")
    print(f"Precision: {pair_prec:.4f}  Recall: {pair_rec:.4f}  F1: {pair_f1:.4f}  PR-AUC: {pr_auc:.4f}")

    print("\nCandidate Recall (union, before cap)")
    print(f"Link recall:            {recall_raw['link_recall']:.4f}")
    print(f"Avg candidates per S1:  {recall_raw['avg_candidates']:.1f}")
    print(f"Max candidates per S1:  {recall_raw['max_candidates']}")

    print(f"\nCandidate Recall (after ranked cap={max_cands_per_query})")
    print(f"Link recall:            {recall_cap['link_recall']:.4f}")
    print(f"Avg candidates per S1:  {recall_cap['avg_candidates']:.1f}")
    print(f"Max candidates per S1:  {recall_cap['max_candidates']}")

    print(f"\nDecision threshold: {threshold:.2f}")
    print(f"Singleton accuracy: {results['singleton_accuracy']:.4f}")

    return {
        "results": results,
        "recall_raw": recall_raw,
        "recall_cap": recall_cap,
        "pair_metrics": {"precision": pair_prec, "recall": pair_rec, "f1": pair_f1, "pr_auc": pr_auc},
        "feat_df": feat_df,
        "predictions": predictions,
        "gt_map": gt_map,
    }


if __name__ == "__main__":
    run_validation()
