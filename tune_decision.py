"""Threshold + singleton cutoff tuning on held-out validation (reuses one feature pass)."""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from src.blocking.candidate_generation import (
    candidates_to_pairs,
    rank_and_cap_candidates,
    run_blocking,
)
from src.features.pair_features import PairFeatureGenerator
from src.models.evaluation import evaluate_macro_f05
from src.preprocessing import preprocess_dataframe
from validate_model import load_target_pool, load_val_split


def predict_with_singleton_cutoff(
    feat_df: pd.DataFrame,
    gt_map: dict,
    match_threshold: float,
    singleton_max_score: float | None = None,
) -> dict:
    """
    Apply match_threshold per pair; if singleton_max_score is set, S1 entities with
    ground-truth unknown at inference we use only scores: clear all preds when max score < cutoff.
    For tuning we simulate inference: singleton_max_score applies to ALL S1 (no GT leak).
    """
    from collections import defaultdict

    preds = defaultdict(set)
    passing = feat_df[feat_df["score"] >= match_threshold]
    for _, row in passing.iterrows():
        preds[row["s1_id"]].add(row["candidate_id"])

    if singleton_max_score is not None:
        max_by_s1 = feat_df.groupby("s1_id")["score"].max()
        for s1_id, mx in max_by_s1.items():
            if mx < singleton_max_score:
                preds[s1_id] = set()
    return dict(preds)


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_baseline.json")
    meta = json.load(open("models/model_metadata.json"))
    cols = meta["feature_cols"]

    gt, val_s1, ns2, ns3 = load_val_split(config)
    s1 = pd.read_csv(config["data"]["train"]["source1"], sep="\t")
    s1 = s1[s1.entity_id.isin(val_s1)]
    target = load_target_pool(config, ns2, ns3)
    s1p = preprocess_dataframe(s1, "v")
    tp = preprocess_dataframe(target, "t")
    raw = run_blocking(s1p, tp, config=config)
    capped = rank_and_cap_candidates(s1p, tp, raw, max_per_query=300)
    pairs = candidates_to_pairs(capped)
    feat = PairFeatureGenerator({}).generate_features(s1p, tp, pairs)
    feat["score"] = model.predict_proba(feat[cols])[:, 1]

    best = (-1.0, None)
    for th in np.arange(0.78, 0.96, 0.01):
        for sing_cut in [None, 0.5, 0.55, 0.6, 0.65, 0.7]:
            preds = predict_with_singleton_cutoff(feat, gt, float(th), sing_cut)
            r = evaluate_macro_f05(preds, gt)
            if r["macro_f05"] > best[0]:
                best = (r["macro_f05"], (float(th), sing_cut, r))

    th, sing, r = best[1]
    print("BEST", "th=", th, "singleton_max_score=", sing)
    print(
        f"F05={r['macro_f05']:.4f} prec={r['mean_precision']:.4f} rec={r['mean_recall']:.4f} "
        f"singleton_acc={r['singleton_accuracy']:.4f}"
    )

    # Finer search around best th
    base_th, base_sing, _ = th, sing
    for th in np.arange(max(0.75, base_th - 0.03), min(0.96, base_th + 0.03), 0.005):
        preds = predict_with_singleton_cutoff(feat, gt, float(th), base_sing)
        r = evaluate_macro_f05(preds, gt)
        if r["macro_f05"] > best[0]:
            best = (r["macro_f05"], (float(th), base_sing, r))
    th, sing, r = best[1]
    print("FINE", "th=", th, "singleton_max_score=", sing, f"F05={r['macro_f05']:.4f}")


if __name__ == "__main__":
    main()
