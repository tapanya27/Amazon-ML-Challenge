"""Quick cap / threshold grid on held-out validation (seed 999)."""
import json

import xgboost as xgb
import yaml

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
from validate_model import load_target_pool, load_val_split


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_baseline.json")
    meta = json.load(open("models/model_metadata.json"))
    cols = meta["feature_cols"]

    gt, val_s1, ns2, ns3 = load_val_split(config)
    import pandas as pd

    s1 = pd.read_csv(config["data"]["train"]["source1"], sep="\t")
    s1 = s1[s1.entity_id.isin(val_s1)]
    target = load_target_pool(config, ns2, ns3)
    s1p = preprocess_dataframe(s1, "v")
    tp = preprocess_dataframe(target, "t")
    raw = run_blocking(s1p, tp, config=config)

    for cap in [300, 400, 500]:
        capped = rank_and_cap_candidates(s1p, tp, raw, max_per_query=cap)
        rec = candidate_recall_stats(capped, gt)["link_recall"]
        pairs = candidates_to_pairs(capped)
        feat = PairFeatureGenerator({}).generate_features(s1p, tp, pairs)
        feat["score"] = model.predict_proba(feat[cols])[:, 1]
        for th in [0.64, 0.68, 0.72, 0.76, 0.80]:
            r = evaluate_macro_f05(pairs_to_predictions(feat, th), gt)
            print(
                f"cap={cap} th={th:.2f} link_rec={rec:.4f} "
                f"F05={r['macro_f05']:.4f} singleton={r['singleton_accuracy']:.3f} "
                f"prec={r['mean_precision']:.3f} rec={r['mean_recall']:.3f}"
            )


if __name__ == "__main__":
    main()
