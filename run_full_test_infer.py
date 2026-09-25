"""
Memory-safe full test inference. Isolated runner (does not edit src/pipeline.py).
Hard-neg model, threshold 0.90, top-k 100, existing test SQLite index.
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
OUT_PATH = os.path.join("output", "matching_results.tsv")
START_BATCH = 250
FALLBACK_BATCH = 100


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
    lines = []
    n_matched = n_links = 0
    for s1_id in s1_ids:
        m_list = batch_preds.get(s1_id, [])
        if m_list:
            n_matched += 1
            n_links += len(m_list)
        lines.append(f"{s1_id}\t{','.join(m_list)}\n")
    release(feat_df, row_lists, batch_preds, names, addrs, countries, s1_ids)
    return lines, n_matched, n_links, stats


def process_chunk(index, feat_gen, model, feature_cols, raw_chunk, batch_size):
    s1_p = preprocess_dataframe(raw_chunk, "S1_Inference")
    out_lines = []
    tot_m = tot_l = 0
    z = c = pre = exp = pmax = 0
    i = 0
    n = len(s1_p)
    while i < n:
        j = min(i + batch_size, n)
        sub = s1_p.iloc[i:j]
        try:
            lines, nm, nl, st = score_batch(index, feat_gen, model, feature_cols, sub)
        except MemoryError:
            if batch_size <= FALLBACK_BATCH:
                raise
            logger.warning("MemoryError on sub-batch %d-%d size=%d; retry size=%d", i, j, batch_size, FALLBACK_BATCH)
            gc.collect()
            lines, nm, nl, st = [], 0, 0, (0, 0, 0, 0, 0)
            k = i
            while k < j:
                k2 = min(k + FALLBACK_BATCH, j)
                sl, a, b, st2 = score_batch(index, feat_gen, model, feature_cols, s1_p.iloc[k:k2])
                lines.extend(sl)
                nm += a
                nl += b
                z += st2[0]
                c += st2[1]
                pre += st2[2]
                exp += st2[3]
                pmax = max(pmax, st2[4])
                k = k2
            out_lines.extend(lines)
            tot_m += nm
            tot_l += nl
            i = j
            release(sub, lines)
            continue
        out_lines.extend(lines)
        tot_m += nm
        tot_l += nl
        z += st[0]
        c += st[1]
        pre += st[2]
        exp += st[3]
        pmax = max(pmax, st[4])
        i = j
        release(sub, lines)
    release(s1_p)
    return out_lines, tot_m, tot_l, (z, c, pre, exp, pmax)


def main():
    assert os.path.isfile(INDEX_PATH), INDEX_PATH
    assert os.path.isfile(MODEL_PATH), MODEL_PATH
    os.makedirs("output", exist_ok=True)
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    assert float(meta["best_threshold"]) == THRESHOLD

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    logger.info("model=%s threshold=%.2f top_k=%d batch=%d rss=%.1f", MODEL_PATH, THRESHOLD, TOP_K, START_BATCH, rss_mb())
    index = SqliteReferenceIndex(INDEX_PATH)
    logger.info("Reusing index %s rows=%d rss=%.1f", INDEX_PATH, index.n, rss_mb())
    feat_gen = PairFeatureGenerator({})

    n_s1 = sum(1 for _ in open(S1_PATH, "rb")) - 1
    logger.info("test S1 rows=%d; restarting from batch 0", n_s1)

    t0 = time.time()
    done = matched = links = zero = cap = pretot = exptot = pretmax = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        for raw in pd.read_csv(S1_PATH, sep="\t", chunksize=START_BATCH):
            try:
                lines, nm, nl, st = process_chunk(index, feat_gen, model, feature_cols, raw, START_BATCH)
            except MemoryError:
                logger.warning("MemoryError on chunk of %d; splitting to %d", START_BATCH, FALLBACK_BATCH)
                gc.collect()
                lines, nm, nl, st = process_chunk(index, feat_gen, model, feature_cols, raw, FALLBACK_BATCH)
            for line in lines:
                out.write(line)
            out.flush()
            n_chunk = len(lines)
            done += n_chunk
            matched += nm
            links += nl
            zero += st[0]
            cap += st[1]
            pretot += st[2]
            exptot += st[3]
            pretmax = max(pretmax, st[4])
            if done % 5000 < START_BATCH or done == n_chunk:
                elapsed = time.time() - t0
                logger.info(
                    "progress s1=%d/%d matched=%d links=%d rss=%.1f elapsed=%.0fs (%.1fs/1k)",
                    done, n_s1, matched, links, rss_mb(), elapsed, elapsed / max(done, 1) * 1000,
                )
            release(lines, raw)

    logger.info(
        "Saved %s n_s1=%d matched=%d links=%d zero_cand=%d cap_hits=%d avg_pre=%.1f max_pre=%d avg_exp=%.1f rss=%.1f",
        OUT_PATH, done, matched, links, zero, cap, pretot / max(done, 1), pretmax, exptot / max(done, 1), rss_mb(),
    )
    out_ids = pd.read_csv(OUT_PATH, sep="\t", usecols=["source1_entity_id"])
    src_ids = pd.read_csv(S1_PATH, sep="\t", usecols=["entity_id"])
    missing = set(src_ids["entity_id"]) - set(out_ids["source1_entity_id"])
    extra = set(out_ids["source1_entity_id"]) - set(src_ids["entity_id"])
    dups = int(out_ids["source1_entity_id"].duplicated().sum())
    logger.info("verify n_out=%d n_src=%d missing=%d extra=%d dups=%d", len(out_ids), len(src_ids), len(missing), len(extra), dups)
    if missing or extra or dups or len(out_ids) != len(src_ids):
        raise SystemExit(f"output incomplete missing={len(missing)} extra={len(extra)} dups={dups}")
    print("VERIFY_OK", len(out_ids), "S1", "links", links, "matched_s1", matched)


if __name__ == "__main__":
    main()
