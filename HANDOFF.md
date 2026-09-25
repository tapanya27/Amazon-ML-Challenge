# Team handoff — business entity resolution (ML Challenge 2026)

**Status:** Experimentation stopped. Use this document to continue from the current best configuration. **Do not treat any local validation number as the official competition score** — only a Portal submission produces that.

---

## 1. Current architecture

End-to-end flow for each Source 1 (S1) test record:

1. **Preprocessing** (`src/preprocessing/`) — normalize business name, address, and country (handles US, India, France labels; no hard-coded country filter).
2. **Reference index** (`src/blocking/sqlite_index.py`) — disk-backed SQLite index over all Source 2 + Source 3 rows. Token-based blocking with posting caps; **no full in-RAM inverted index**.
3. **Retrieve + prerank** — `retrieve(..., retrieve_cap=350)` then `cheap_prerank(..., top_k=100)` to keep at most 100 candidates per S1 row before scoring.
4. **Pair features** (`src/features/pair_features.py`) — 27 similarity features (name/address/country; RapidFuzz-based string metrics). Column list is in `models/exp_hardneg/infer_metadata.json`.
5. **Classifier** — XGBoost binary match model (`models/exp_hardneg/xgb_hardneg.json`), trained with hard-negative mining (`train_hardneg_exp.py`; **do not retrain** unless intentionally starting a new experiment).
6. **Decision** (`src/models/decision.py`) — entity-level thresholding via `pairs_to_predictions` at **0.90**; exact name+address+country fast-path bumps scores to at least the threshold during inference.

**Production baseline (older):** `models/xgb_baseline.json`, `models/model_metadata.json`, and `src/pipeline.py` defaults still point at the baseline model and `top_k=30`. The **current best path for test inference** is the isolated runner below, not unmodified `pipeline.py`.

---

## 2. Current best model and threshold

| Setting | Value |
|--------|--------|
| Model | `models/exp_hardneg/xgb_hardneg.json` |
| Threshold | **0.90** |
| Candidate top-k (after prerank) | **100** |
| Retrieve cap (before prerank) | **350** |
| Inference metadata | `models/exp_hardneg/infer_metadata.json` |
| Training / val report | `models/exp_hardneg/report.json`, `models/exp_hardneg/threshold_sweep.json` |

---

## 3. Validation result (local only — not official test score)

On a **held-out training slice** (1,500 S1 entities; see `train_hardneg_exp.py` / `models/exp_hardneg/report.json`):

- **Macro F₀.₅ ≈ 0.821** at threshold **0.90** (`hardneg_val_best_th` in `report.json`).
- This measures local train holdout quality only. **It is not the competition leaderboard score.**

To reproduce validation metrics (optional, not required for test inference):

```bash
python train_hardneg_exp.py
```

Uses existing train index `data/indexes/s2s3.sqlite` and does **not** replace the saved model unless the script is changed to do so.

---

## 4. SQLite index locations (pre-built — do not rebuild)

| Index | Path | Purpose |
|-------|------|---------|
| Train S2+S3 | `data/indexes/s2s3.sqlite` | Training experiments, hard-neg mining, local eval (~3.0 GB) |
| Test S2+S3 | `data/indexes/test_s2s3.sqlite` | **Test inference** (~3.0 GB) |

**Do not delete, move, or rebuild** these files unless you explicitly intend to regenerate them (hours of work, large disk). Scripts that *would* rebuild (avoid running on full data):

- `build_sqlite_index.py` — test index builder
- `resume_train_sqlite_index.py` — train index ingest resume
- `src/pipeline.py --rebuild-index`

If `test_s2s3.sqlite-wal` / `.sqlite-shm` exist, leave them; SQLite may be using WAL mode.

---

## 5. Full test inference status

- **Full test inference has NOT completed successfully.**
- A **`MemoryError`** occurred during an earlier full test run using **`src/pipeline.py`** with default **S1 batch size 4000** (failure around the **4000-row batch boundary** — only a partial `output/matching_results.partial.tsv` exists, ~44k S1 rows of ~1.73M total).
- A separate attempt via `run_full_test_infer.py` also left **incomplete** `output/matching_results.tsv` (~13k rows). Treat all current `output/*.tsv` as **partial / not submittable**.

---

## 6. What the next developer should run

**Primary test inference entrypoint:** `run_full_test_infer.py` (hard-neg model, threshold 0.90, top-k 100, reuses `data/indexes/test_s2s3.sqlite`).

From repo root:

```bash
python run_full_test_infer.py
```

Built-in memory settings (edit constants at top of `run_full_test_infer.py` if needed):

- `START_BATCH = 250` — start here for S1 chunk processing
- `FALLBACK_BATCH = 100` — automatic retry on `MemoryError`
- `THRESHOLD = 0.90`, `TOP_K = 100`, `RETRIEVE_CAP = 350`
- `INDEX_PATH = data/indexes/test_s2s3.sqlite`
- `MODEL_PATH = models/exp_hardneg/xgb_hardneg.json`

**Important:** This script **restarts from S1 row 0** and **overwrites** `output/matching_results.tsv`. For a ~1.73M-row test set, plan for **resume/checkpointing** or long runtime before relying on a single run.

After a **complete** run, validate format (requires `candidate_pairs.tsv` too — `run_full_test_infer.py` does **not** write candidates; use `src/pipeline.py` with hard-neg paths or add candidate export if the submission package needs it):

```bash
python utils/validate_submission.py ^
  --matching output/matching_results.tsv ^
  --candidate output/candidate_pairs.tsv ^
  --test-dir dataset/test
```

Alternative (baseline defaults — **not** current best without CLI overrides):

```bash
python -m src.pipeline --index-path data/indexes/test_s2s3.sqlite --batch-size 250 --top-k 100
```

You must pass `--model-path models/exp_hardneg/xgb_hardneg.json` and matching metadata if extending `pipeline.py` CLI (today it does not expose model path; prefer `run_full_test_infer.py`).

---

## 7. Known bottleneck

The main practical ceiling is **candidate recall / ranking** (blocking + top-k prerank), not SQLite index construction. Indexes are already built; improving match quality likely means better candidates or model features — not re-indexing.

---

## 8. Key files and directories

| Area | Location |
|------|----------|
| Core library | `src/` (blocking, features, models, preprocessing) |
| Config (baseline pipeline) | `configs/config.yaml` |
| Hard-neg train script | `train_hardneg_exp.py` |
| Test inference (current best) | `run_full_test_infer.py` |
| Local train eval | `eval_train_full.py`, `run_experiments_2000.py`, `eval_train/` |
| Data | `dataset/train/`, `dataset/test/` |
| Submission validator | `utils/validate_submission.py` |
| Challenge README | `README.md` |

---

## 9. Environment

See `requirements.txt`. Verified locally with Python 3.x, `xgboost` 3.3.0, `pandas` 2.3.3, `numpy` 2.3.4. Optional: `psutil` for RSS logging in inference scripts.

---

## 10. Explicit do-nots (unless starting a new experiment)

- Do not retrain or overwrite `models/exp_hardneg/xgb_hardneg.json`.
- Do not rebuild or delete `data/indexes/s2s3.sqlite` or `data/indexes/test_s2s3.sqlite`.
- Do not claim an **official test score** without a competition submission.
- Do not assume partial `output/` TSVs are complete or leaderboard-ready.
