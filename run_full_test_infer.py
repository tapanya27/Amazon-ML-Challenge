"""
Memory-safe full test inference. Isolated runner (does not edit src/pipeline.py).
Hard-neg model, threshold 0.90, top-k 100, existing test SQLite index.
Writes matching_results.tsv + candidate_pairs.tsv incrementally with resume support.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.blocking.sqlite_index import SqliteReferenceIndex
from src.features.pair_features import PairFeatureGenerator
from src.models.decision import pairs_to_predictions
from src.preprocessing import preprocess_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("full_test")
logging.getLogger("src.preprocessing.normalize").setLevel(logging.ERROR)
logging.getLogger("src.features.pair_features").setLevel(logging.WARNING)

THRESHOLD = 0.90
TOP_K = 100
RETRIEVE_CAP = 350
INDEX_PATH = "data/indexes/test_s2s3.sqlite"
MODEL_PATH = "models/exp_hardneg/xgb_hardneg.json"
META_PATH = "models/exp_hardneg/infer_metadata.json"
S1_PATH = "dataset/test/test_source1.tsv"
OUT_MATCH = os.path.join("output", "matching_results.tsv")
OUT_CAND = os.path.join("output", "candidate_pairs.tsv")
CHECKPOINT_PATH = os.path.join("output", ".infer_checkpoint.json")
START_BATCH = 250
FALLBACK_BATCH = 100
MIN_BATCH = 50


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def release(*objs):
    for o in objs:
        del o
    gc.collect()


def count_data_rows(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def load_resume() -> tuple[int, int, int]:
    """Return (rows_done, matched_s1, pred_links) if outputs are consistent; else restart."""
    n_m = count_data_rows(OUT_MATCH)
    n_c = count_data_rows(OUT_CAND)
    if n_m == 0 and n_c == 0:
        return 0, 0, 0
    if n_m != n_c:
        logger.warning(
            "Resume mismatch matching=%d candidate=%d — restarting both outputs from zero",
            n_m, n_c,
        )
        return 0, 0, 0
    if os.path.isfile(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            ck = json.load(f)
        if int(ck.get("done", -1)) == n_m:
            return n_m, int(ck.get("matched", 0)), int(ck.get("links", 0))
    logger.warning("Checkpoint missing or stale; trusting row count %d only (stats reset)", n_m)
    return n_m, 0, 0


def save_checkpoint(done: int, matched: int, links: int) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"done": done, "matched": matched, "links": links, "ts": time.time()}, f)


def iter_s1_chunks(chunksize: int, skip_rows: int):
    remaining = skip_rows
    for chunk in pd.read_csv(S1_PATH, sep="\t", chunksize=chunksize):
        if remaining >= len(chunk):
            remaining -= len(chunk)
            del chunk
            gc.collect()
            continue
        if remaining > 0:
            chunk = chunk.iloc[remaining:].copy()
            remaining = 0
        yield chunk
        del chunk


def score_batch(index, feat_gen, model, feature_cols, s1_p, use_fast_path=True):
    s1_ids = s1_p["entity_id"].tolist()
    names = s1_p["name_normalized"].fillna("").astype(str).tolist()
    addrs = s1_p["address_normalized"].fillna("").astype(str).tolist()
    countries = s1_p["country_normalized"].fillna("").astype(str).str.lower().tolist()
    row_lists = []
    n_zero = n_cap = n_pre = n_exp = 0
    premax = 0
    for nm, ad, co in zip(names, addrs, countries):
        nm = "" if nm == "nan" else nm
        ad = "" if ad == "nan" else ad
        rows, raw_n, hit = index.retrieve(nm, ad, co, retrieve_cap=RETRIEVE_CAP)
        n_pre += int(raw_n)
        premax = max(premax, int(raw_n) if raw_n else int(rows.size))
        if hit:
            n_cap += 1
        if rows.size == 0:
            n_zero += 1
        top = index.cheap_prerank(nm, ad, co, rows, TOP_K)
        row_lists.append(top)
        n_exp += int(top.size)
        del rows
    feat_df = feat_gen.generate_from_index(index, s1_ids, names, addrs, countries, row_lists)
    batch_preds = defaultdict(list)
    if len(feat_df):
        scores = model.predict_proba(feat_df[feature_cols])[:, 1]
        feat_df = feat_df.copy()
        feat_df["score"] = scores
        if use_fast_path:
            exact = (feat_df["name_exact"] >= 1.0) & (feat_df["addr_exact"] >= 1.0) & (feat_df["country_same"] >= 1.0)
            feat_df.loc[exact, "score"] = np.maximum(feat_df.loc[exact, "score"], THRESHOLD)
            del exact
        pred_sets = pairs_to_predictions(feat_df, threshold=THRESHOLD, score_col="score")
        for k, v in pred_sets.items():
            batch_preds[k] = sorted(v)
        del scores, pred_sets
    stats = (n_zero, n_cap, n_pre, n_exp, premax)
    match_lines = []
    cand_lines = []
    n_matched = n_links = 0
    for i, s1_id in enumerate(s1_ids):
        m_list = batch_preds.get(s1_id, [])
        if m_list:
            n_matched += 1
            n_links += len(m_list)
        match_lines.append(f"{s1_id}\t{','.join(m_list)}\n")
        cids = [index.entity_id(int(r)) for r in row_lists[i]]
        cand_lines.append(f"{s1_id}\t{','.join(cids)}\n")
    release(feat_df, row_lists, batch_preds, names, addrs, countries, s1_ids)
    return match_lines, cand_lines, n_matched, n_links, stats


def score_batch_with_fallback(index, feat_gen, model, feature_cols, s1_p, batch_size: int):
    sizes = []
    for s in (batch_size, FALLBACK_BATCH, MIN_BATCH):
        if s <= batch_size and (not sizes or s < sizes[-1]):
            sizes.append(s)
    last_err = None
    for sz in sizes:
        try:
            return _run_sub_batches(index, feat_gen, model, feature_cols, s1_p, sz)
        except MemoryError as e:
            last_err = e
            logger.warning("MemoryError at sub-batch size=%d; gc and retry smaller", sz)
            gc.collect()
    raise last_err  # type: ignore[misc]


def _run_sub_batches(index, feat_gen, model, feature_cols, s1_p, batch_size: int):
    n = len(s1_p)
    i = 0
    all_m, all_c = [], []
    tot_m = tot_l = 0
    z = c = pre = exp = pmax = 0
    while i < n:
        j = min(i + batch_size, n)
        sub = s1_p.iloc[i:j]
        m_lines, c_lines, nm, nl, st = score_batch(index, feat_gen, model, feature_cols, sub)
        all_m.extend(m_lines)
        all_c.extend(c_lines)
        tot_m += nm
        tot_l += nl
        z += st[0]
        c += st[1]
        pre += st[2]
        exp += st[3]
        pmax = max(pmax, st[4])
        i = j
        release(sub, m_lines, c_lines)
    return all_m, all_c, tot_m, tot_l, (z, c, pre, exp, pmax)


def process_chunk(index, feat_gen, model, feature_cols, raw_chunk, batch_size: int):
    s1_p = preprocess_dataframe(raw_chunk, "S1_Inference")
    result = score_batch_with_fallback(index, feat_gen, model, feature_cols, s1_p, batch_size)
    release(s1_p)
    return result


def open_outputs(resume_rows: int):
    mode = "a" if resume_rows > 0 else "w"
    fm = open(OUT_MATCH, mode, encoding="utf-8")
    fc = open(OUT_CAND, mode, encoding="utf-8")
    if resume_rows == 0:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.flush()
        fc.flush()
    return fm, fc


def main():
    assert os.path.isfile(INDEX_PATH), INDEX_PATH
    assert os.path.isfile(MODEL_PATH), MODEL_PATH
    os.makedirs("output", exist_ok=True)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    assert float(meta["best_threshold"]) == THRESHOLD

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    index = SqliteReferenceIndex(INDEX_PATH)
    feat_gen = PairFeatureGenerator({})

    n_s1 = sum(1 for _ in open(S1_PATH, "rb")) - 1
    resume_rows, matched, links = load_resume()
    if resume_rows >= n_s1:
        logger.info("Already complete: %d/%d rows", resume_rows, n_s1)
    else:
        logger.info(
            "model=%s threshold=%.2f top_k=%d batch=%d resume=%d/%d index_rows=%d rss=%.1f",
            MODEL_PATH, THRESHOLD, TOP_K, START_BATCH, resume_rows, n_s1, index.n, rss_mb(),
        )

    t0 = time.time()
    done = resume_rows
    zero = cap = pretot = exptot = pretmax = 0
    fm, fc = open_outputs(resume_rows)

    try:
        for raw in iter_s1_chunks(START_BATCH, resume_rows):
            try:
                m_lines, c_lines, nm, nl, st = process_chunk(
                    index, feat_gen, model, feature_cols, raw, START_BATCH,
                )
            except MemoryError:
                logger.warning("MemoryError on outer chunk; retry chunk with batch=%d", FALLBACK_BATCH)
                gc.collect()
                m_lines, c_lines, nm, nl, st = process_chunk(
                    index, feat_gen, model, feature_cols, raw, FALLBACK_BATCH,
                )
            for ml, cl in zip(m_lines, c_lines):
                fm.write(ml)
                fc.write(cl)
            fm.flush()
            fc.flush()
            n_chunk = len(m_lines)
            done += n_chunk
            matched += nm
            links += nl
            zero += st[0]
            cap += st[1]
            pretot += st[2]
            exptot += st[3]
            pretmax = max(pretmax, st[4])
            save_checkpoint(done, matched, links)
            if done % 5000 < START_BATCH or done == resume_rows + n_chunk:
                elapsed = time.time() - t0
                logger.info(
                    "progress s1=%d/%d matched=%d links=%d rss=%.1f elapsed=%.0fs (%.1fs/1k)",
                    done, n_s1, matched, links, rss_mb(), elapsed, elapsed / max(done - resume_rows, 1) * 1000,
                )
            release(m_lines, c_lines, raw)
    finally:
        fm.close()
        fc.close()

    elapsed = time.time() - t0
    logger.info(
        "Saved %s and %s n_s1=%d matched=%d links=%d zero_cand=%d cap_hits=%d avg_pre=%.1f max_pre=%d avg_exp=%.1f rss=%.1f elapsed=%.0fs",
        OUT_MATCH, OUT_CAND, done, matched, links, zero, cap,
        pretot / max(done - resume_rows, 1), pretmax, exptot / max(done - resume_rows, 1), rss_mb(), elapsed,
    )

    out_ids = pd.read_csv(OUT_MATCH, sep="\t", usecols=["source1_entity_id"])
    cand_ids = pd.read_csv(OUT_CAND, sep="\t", usecols=["source1_entity_id"])
    src_ids = pd.read_csv(S1_PATH, sep="\t", usecols=["entity_id"])
    missing = set(src_ids["entity_id"]) - set(out_ids["source1_entity_id"])
    extra = set(out_ids["source1_entity_id"]) - set(src_ids["entity_id"])
    dups = int(out_ids["source1_entity_id"].duplicated().sum())
    logger.info(
        "verify match=%d cand=%d src=%d missing=%d extra=%d dups=%d",
        len(out_ids), len(cand_ids), len(src_ids), len(missing), len(extra), dups,
    )
    if len(cand_ids) != len(out_ids):
        raise SystemExit(f"candidate row count {len(cand_ids)} != matching {len(out_ids)}")
    if missing or extra or dups or len(out_ids) != len(src_ids):
        raise SystemExit(f"output incomplete missing={len(missing)} extra={len(extra)} dups={dups}")
    save_checkpoint(done, matched, links)
    print("VERIFY_OK", len(out_ids), "S1", "links", links, "matched_s1", matched, "elapsed_s", round(elapsed, 1))


if __name__ == "__main__":
    main()
