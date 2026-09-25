"""
Memory-safe batched inference using a one-time integer ReferenceIndex.
Writes matching_results.partial.tsv incrementally; renames on success.
"""

import os
import sys
import yaml
import json
import logging
import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_dataframe
from src.blocking.sqlite_index import SqliteReferenceIndex
from src.features.pair_features import PairFeatureGenerator
from src.models.decision import pairs_to_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def run_pipeline(
    s1_path: str = "dataset/test/test_source1.tsv",
    s2_path: str = "dataset/test/test_source2.tsv",
    s3_path: str = "dataset/test/test_source3.tsv",
    output_dir: str = "output",
    model_path: str = "models/xgb_baseline.json",
    metadata_path: str = "models/model_metadata.json",
    sample_size: int = None,
    top_k: int = 30,
    retrieve_cap: int = 350,
    s1_batch_size: int = 4000,
    use_fast_path: bool = True,
    write_candidates: bool = True,
    index_path: str = "data/indexes/s2s3.sqlite",
    rebuild_index: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    with open(metadata_path) as f:
        meta = json.load(f)
    threshold = float(meta.get("best_threshold", 0.92))
    feature_cols = meta["feature_cols"]
    logger.info("Threshold=%.3f top_k=%d retrieve_cap=%d", threshold, top_k, retrieve_cap)

    logger.info("Loading Source 1 from %s...", s1_path)
    s1_df = pd.read_csv(s1_path, sep="\t", nrows=sample_size)
    logger.info("Loaded %d Source 1 records.", len(s1_df))
    s1_p = preprocess_dataframe(s1_df, "S1_Inference")

    if rebuild_index or not os.path.isfile(index_path):
        logger.info("Building disk-backed index at %s from FULL S2+S3 (chunked, no 10M RAM concat)...", index_path)
        index = SqliteReferenceIndex.build_from_files([s2_path, s3_path], index_path)
    else:
        logger.info("Reusing SQLite index %s", index_path)
        index = SqliteReferenceIndex(index_path)
    logger.info("Index rows: %d", index.n)
    feat_gen = PairFeatureGenerator({})

    match_partial = os.path.join(output_dir, "matching_results.partial.tsv")
    cand_partial = os.path.join(output_dir, "candidate_pairs.partial.tsv")
    match_final = os.path.join(output_dir, "matching_results.tsv")
    cand_final = os.path.join(output_dir, "candidate_pairs.tsv")

    n_s1 = len(s1_p)
    matched_count = 0
    total_links = 0
    zero_cand = 0
    cap_hits = 0
    pretot = 0
    exptot = 0
    pretmax = 0

    with open(match_partial, "w", encoding="utf-8") as fm, open(cand_partial, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for start in range(0, n_s1, s1_batch_size):
            end = min(start + s1_batch_size, n_s1)
            batch = s1_p.iloc[start:end]
            logger.info("S1 batch %d-%d / %d", start, end, n_s1)

            s1_ids = batch["entity_id"].tolist()
            names = batch["name_normalized"].fillna("").astype(str).tolist()
            addrs = batch["address_normalized"].fillna("").astype(str).tolist()
            countries = batch["country_normalized"].fillna("").astype(str).str.lower().tolist()

            retrieved = []
            row_lists = []
            for nm, ad, co in zip(names, addrs, countries):
                nm = "" if nm == "nan" else nm
                ad = "" if ad == "nan" else ad
                rows, raw_n, hit = index.retrieve(nm, ad, co, retrieve_cap=retrieve_cap)
                pretot += raw_n
                pretmax = max(pretmax, int(raw_n) if raw_n else rows.size)
                if hit:
                    cap_hits += 1
                if rows.size == 0:
                    zero_cand += 1
                top = index.cheap_prerank(nm, ad, co, rows, top_k)
                retrieved.append(rows)
                row_lists.append(top)
                exptot += len(top)

            feat_df = feat_gen.generate_from_index(index, s1_ids, names, addrs, countries, row_lists)
            batch_preds = defaultdict(list)
            if len(feat_df):
                scores = model.predict_proba(feat_df[feature_cols])[:, 1]
                feat_df["score"] = scores
                if use_fast_path:
                    # unambiguous: exact name+address already in features
                    exact = (feat_df["name_exact"] >= 1.0) & (feat_df["addr_exact"] >= 1.0) & (feat_df["country_same"] >= 1.0)
                    feat_df.loc[exact, "score"] = np.maximum(feat_df.loc[exact, "score"], threshold)
                pred_sets = pairs_to_predictions(feat_df, threshold=threshold, score_col="score")
                for k, v in pred_sets.items():
                    batch_preds[k] = sorted(v)

            for i, s1_id in enumerate(s1_ids):
                m_list = batch_preds.get(s1_id, [])
                if m_list:
                    matched_count += 1
                    total_links += len(m_list)
                fm.write(f"{s1_id}\t{','.join(m_list)}\n")
                if write_candidates:
                    cids = [index.entity_id(int(r)) for r in row_lists[i]]
                    fc.write(f"{s1_id}\t{','.join(cids)}\n")

            fm.flush()
            fc.flush()
            del feat_df, retrieved, row_lists, batch_preds

    os.replace(match_partial, match_final)
    if write_candidates:
        os.replace(cand_partial, cand_final)

    elapsed = time.time() - t0
    logger.info("Saved %s", match_final)
    logger.info(
        "Done: %d/%d S1 with matches (%d links). zero_cand=%d cap_hits=%d "
        "avg_pre=%.1f max_pre=%d avg_expensive=%.1f time=%.1fs (%.1fs/1k S1)",
        matched_count, n_s1, total_links, zero_cand, cap_hits,
        pretot / max(n_s1, 1), pretmax, exptot / max(n_s1, 1),
        elapsed, elapsed / max(n_s1, 1) * 1000,
    )
    stats = {
        "n_s1": n_s1,
        "matched_s1": matched_count,
        "pred_links": total_links,
        "zero_candidate_s1": zero_cand,
        "cap_hits": cap_hits,
        "avg_retrieved": pretot / max(n_s1, 1),
        "max_retrieved": pretmax,
        "avg_expensive": exptot / max(n_s1, 1),
        "seconds": elapsed,
        "sec_per_1000": elapsed / max(n_s1, 1) * 1000,
    }
    return match_final, cand_final, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", default="dataset/test/test_source1.tsv")
    parser.add_argument("--s2", default="dataset/test/test_source2.tsv")
    parser.add_argument("--s3", default="dataset/test/test_source3.tsv")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--retrieve-cap", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--no-fast-path", action="store_true")
    parser.add_argument("--no-candidates", action="store_true")
    parser.add_argument("--index-path", default="data/indexes/test_s2s3.sqlite")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()
    run_pipeline(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        top_k=args.top_k,
        retrieve_cap=args.retrieve_cap,
        s1_batch_size=args.batch_size,
        use_fast_path=not args.no_fast_path,
        write_candidates=not args.no_candidates,
        index_path=args.index_path,
        rebuild_index=args.rebuild_index,
    )
