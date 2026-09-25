"""Export false positives, false negatives, singleton and ambiguous errors."""
import json
import os

import pandas as pd
import xgboost as xgb
import yaml

from src.blocking.candidate_generation import (
    candidates_to_pairs,
    rank_and_cap_candidates,
    run_blocking,
)
from src.features.pair_features import PairFeatureGenerator
from src.models.decision import pairs_to_predictions
from src.preprocessing import preprocess_dataframe
from validate_model import load_target_pool, load_val_split

OUT = "output/error_analysis"
THRESHOLD = 0.92
CAP = 300


def main():
    os.makedirs(OUT, exist_ok=True)
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_baseline.json")
    cols = json.load(open("models/model_metadata.json"))["feature_cols"]

    gt, val_s1, ns2, ns3 = load_val_split(config)
    s1 = pd.read_csv(config["data"]["train"]["source1"], sep="\t")
    s1 = s1[s1.entity_id.isin(val_s1)].reset_index(drop=True)
    target = load_target_pool(config, ns2, ns3)
    s1p = preprocess_dataframe(s1, "v")
    tp = preprocess_dataframe(target, "t")
    raw = run_blocking(s1p, tp, config=config)
    capped = rank_and_cap_candidates(s1p, tp, raw, max_per_query=CAP)
    pairs = candidates_to_pairs(capped)
    feat = PairFeatureGenerator({}).generate_features(s1p, tp, pairs)
    feat["score"] = model.predict_proba(feat[cols])[:, 1]
    preds = pairs_to_predictions(feat, threshold=THRESHOLD)

    s1_idx = s1.set_index("entity_id")
    t_idx = target.set_index("entity_id")

    fp_rows, fn_rows, sing_rows, amb_rows = [], [], [], []

    for s1_id, true_set in gt.items():
        pred_set = preds.get(s1_id, set())
        q = s1_idx.loc[s1_id]

        if not true_set:
            if pred_set:
                for pid in pred_set:
                    c = t_idx.loc[pid]
                    pr = feat[(feat.s1_id == s1_id) & (feat.candidate_id == pid)].iloc[0]
                    sing_rows.append(_row(q, c, pr, 0))
            continue

        missed = true_set - pred_set
        extra = pred_set - true_set
        if len(pred_set) > 1 and len(true_set) > 1:
            for pid in pred_set:
                pr = feat[(feat.s1_id == s1_id) & (feat.candidate_id == pid)].iloc[0]
                amb_rows.append(_row(q, t_idx.loc[pid], pr, 1 if pid in true_set else 0))

        for pid in extra:
            c = t_idx.loc[pid]
            pr = feat[(feat.s1_id == s1_id) & (feat.candidate_id == pid)].iloc[0]
            fp_rows.append(_row(q, c, pr, 0))

        for pid in missed:
            sub = feat[(feat.s1_id == s1_id) & (feat.candidate_id == pid)]
            if len(sub):
                pr = sub.iloc[0]
            else:
                pr = {"score": -1.0, "name_token_set": 0, "addr_token_set": 0, "total_overlap": 0}
            c = t_idx.loc[pid]
            fn_rows.append(_row(q, c, pr, 1))

    _save(fp_rows, f"{OUT}/false_positive_examples.csv")
    _save(fn_rows, f"{OUT}/false_negative_examples.csv")
    _save(sing_rows, f"{OUT}/singleton_errors.csv")
    _save(amb_rows[:500], f"{OUT}/ambiguous_examples.csv")
    print(f"Wrote {len(fp_rows)} FP, {len(fn_rows)} FN, {len(sing_rows)} singleton errs to {OUT}/")


def _row(q, c, pr, label):
    if hasattr(pr, "to_dict"):
        pr = pr.to_dict()
    return {
        "s1_id": q.name if hasattr(q, "name") else q.get("entity_id"),
        "candidate_id": c.name if hasattr(c, "name") else c.get("entity_id"),
        "name_s1": q.business_name,
        "name_cand": c.business_name,
        "addr_s1": q.business_address,
        "addr_cand": c.business_address,
        "country": q.country,
        "score": pr.get("score", pr.get("pred_score")),
        "true_label": label,
        "name_token_set": pr.get("name_token_set"),
        "addr_token_set": pr.get("addr_token_set"),
        "total_overlap": pr.get("total_overlap"),
    }


def _save(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
