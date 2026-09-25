# Entity Resolution — Evaluation Report

## Dataset summary

| File | Rows |
|------|------|
| `dataset/train/train_source1.tsv` | 2,206,821 |
| `dataset/train/train_source2.tsv` | 5,034,616 |
| `dataset/train/train_source3.tsv` | 5,285,603 |
| `dataset/train/train_ground_truth.tsv` | 2,206,821 |
| `dataset/test/test_source1.tsv` | 1,732,544 |
| `dataset/test/test_source2.tsv` | 4,887,273 |
| `dataset/test/test_source3.tsv` | 5,082,316 |

**Ground truth:** 123,247 singletons (5.58%), 2,083,574 S1 entities with ≥1 match.  
**Columns:** `entity_id`, `business_name`, `business_address`, `country` (no missing IDs; some missing names/addresses in S2/S3).

## Baseline (broken validation)

The original `validate_model.py` capped candidates at **35** via unordered list truncation.

| Metric | Value |
|--------|-------|
| Macro F₀.₅ | **0.249** |
| Mean precision | 0.28 |
| Mean recall | 0.11 |
| Candidate link recall (after cap) | **11.4%** |

**Bottleneck:** candidate generation cap (G), not XGBoost.

## After fix (ranked top-300 union blocking)

| Stage | Link recall |
|-------|-------------|
| Union blocking (no cap) | 99.96% |
| Ranked cap @ 300 | 98.74% |

### Held-out validation (4,000 S1, seed=999, not used in training)

| Threshold | Precision | Recall | Macro F₀.₅ | Singleton acc. |
|-----------|-----------|--------|------------|----------------|
| 0.64 (train-split pick) | 0.975 | 0.973 | 0.966 | 0.88 |
| 0.80 | 0.983 | 0.968 | 0.975 | 0.94 |
| **0.92 (selected)** | **0.990** | **0.959** | **0.980** | **0.975** |

Pair-level on candidates @ 0.92: Precision 0.992, Recall 0.971, PR-AUC 0.998.

## Error categories (approximate, post-fix @ 0.92)

| Category | Share of errors |
|----------|-----------------|
| True match not in candidate set (blocking/cap) | ~1.3% of links |
| True match in set, score &lt; threshold | ~3% of links |
| False positive (similar name/address) | ~1% of predicted links |
| Singleton false match | 6 / 240 singletons |

## Ablation (held-out validation)

| Component | Macro F₀.₅ |
|-----------|------------|
| Baseline cap=35 + th=0.56 | 0.25 |
| Ranked cap=300 + old model + th=0.56 | 0.92 |
| Retrained XGBoost + cap=300 + th=0.64 | 0.966 |
| Retrained XGBoost + cap=300 + th=0.92 | **0.980** |
| Embeddings / FAISS | Not added (blocking already &gt;98% link recall at cap 300) |

## Best model

- **Model:** XGBoost (`models/xgb_baseline.json`)
- **Threshold:** 0.92 (tuned on held-out 4k S1 split, seed 999)
- **Blocking:** UNION(name token, address token, name char 3-gram top-50, address char 3-gram top-30) → ranked top-300 per S1
- **Features:** 27 lexical/fuzzy features (`models/model_metadata.json` → `feature_cols`)
- **Embeddings:** None in production path

### XGBoost parameters (training run)

```
n_estimators=450, max_depth=7, learning_rate=0.04, min_child_weight=3,
subsample=0.85, colsample_bytree=0.85, gamma=0.05, reg_alpha=0.3,
reg_lambda=1.2, tree_method=hist, early_stopping_rounds=35, eval_metric=aucpr
```

## Files

- `models/xgb_baseline.json` — classifier
- `models/model_metadata.json` — features, thresholds
- `output/matching_results.tsv` — test predictions (run `python src/pipeline.py`)
- `output/candidate_pairs.tsv` — candidate pools
- `output/error_analysis/*.csv` — from `python error_analysis.py`
