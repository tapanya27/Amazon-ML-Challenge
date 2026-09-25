"""
Unified candidate generation: multi-strategy blocking, union, and ranked top-K capping.

Arbitrary truncation (e.g. first 35 IDs) destroys recall; we rank by cheap lexical
signals before applying a per-query cap.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .ngram_block import NgramBlocker
from .token_block import TokenBlocker

logger = logging.getLogger(__name__)

NAME_STOPWORDS = {
    "limited", "ltd", "company", "co", "inc", "incorporated", "corp", "corporation",
    "pvt", "private", "llc", "llp", "the", "and", "of", "for", "in", "at", "dba", "group",
}


def default_blockers(config: dict, n_target: int = 0) -> list:
    """Blockers aligned with configs/config.yaml (country blocker is a no-op Cartesian guard)."""
    blk = config.get("blocking", {})
    name_cfg = blk.get("name_token", {})
    addr_cfg = blk.get("address_token", {})
    ngram_cfg = blk.get("ngram", {})
    large = n_target >= 1_000_000

    blockers = [
        TokenBlocker(
            config={},
            column="name_normalized",
            min_token_len=name_cfg.get("min_token_length", 3),
            max_freq_ratio=name_cfg.get("max_token_frequency_ratio", 0.01),
            stopwords=set(name_cfg.get("stopwords", [])) or NAME_STOPWORDS,
            max_candidates_per_query=400 if large else 2000,
            max_posting_len=5000 if large else 50_000,
        ),
        TokenBlocker(
            config={},
            column="address_normalized",
            min_token_len=addr_cfg.get("min_token_length", 2),
            max_freq_ratio=addr_cfg.get("max_token_frequency_ratio", 0.01),
            max_candidates_per_query=400 if large else 2000,
            max_posting_len=5000 if large else 50_000,
        ),
    ]
    # Character n-gram TF-IDF on ~10M rows does not fit in memory.
    if not large:
        blockers.extend(
            [
                NgramBlocker(
                    config={},
                    column="name_normalized",
                    n=ngram_cfg.get("n", 3),
                    top_k=ngram_cfg.get("top_k", 50),
                    batch_size=250,
                ),
                NgramBlocker(
                    config={},
                    column="address_normalized",
                    n=ngram_cfg.get("n", 3),
                    top_k=min(30, ngram_cfg.get("top_k", 50)),
                    batch_size=250,
                ),
            ]
        )
    return blockers


def fit_blockers(target_df: pd.DataFrame, config: Optional[dict] = None) -> list:
    config = config or {}
    blockers = default_blockers(config, n_target=len(target_df))
    for blocker in blockers:
        blocker.fit(target_df)
    return blockers


def transform_union(blockers: list, s1_df: pd.DataFrame) -> Dict[str, Set[str]]:
    candidates: Dict[str, Set[str]] = defaultdict(set)
    for blocker in blockers:
        batch = blocker.transform(s1_df)
        for s1_id, c_set in batch.items():
            candidates[s1_id].update(c_set)
    logger.info(
        "Blocking union: %d / %d S1 queries with >=1 candidate",
        sum(1 for v in candidates.values() if v),
        len(s1_df),
    )
    return dict(candidates)


def run_blocking(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    blockers: Optional[list] = None,
    config: Optional[dict] = None,
) -> Dict[str, Set[str]]:
    """Union candidates from all blockers."""
    config = config or {}
    if blockers is None:
        blockers = fit_blockers(target_df, config)
        return transform_union(blockers, s1_df)
    for blocker in blockers:
        if getattr(blocker, "inverted_index", None) is None and getattr(blocker, "target_matrix", None) is None:
            blocker.fit(target_df)
    return transform_union(blockers, s1_df)


def _token_set(text: str) -> Set[str]:
    if not text or (isinstance(text, float) and np.isnan(text)):
        return set()
    return set(str(text).split())


def _cheap_rank_score(
    q_name: str,
    q_addr: str,
    c_name: str,
    c_addr: str,
    q_name_t: Set[str],
    q_addr_t: Set[str],
    c_name_t: Set[str],
    c_addr_t: Set[str],
) -> float:
    name_ov = len(q_name_t & c_name_t)
    addr_ov = len(q_addr_t & c_addr_t)
    if name_ov == 0 and addr_ov == 0:
        return 0.0
    # Weight address overlap (often more discriminative for duplicates)
    base = name_ov * 2.0 + addr_ov * 3.0
    if q_name and c_name:
        base += 0.01 * fuzz.token_set_ratio(q_name, c_name)
    if q_addr and c_addr:
        base += 0.01 * fuzz.token_set_ratio(q_addr, c_addr)
    return base


def rank_and_cap_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidates: Dict[str, Set[str]],
    max_per_query: Optional[int] = 300,
    inject_matches: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Set[str]]:
    """
    Rank union candidates and keep top max_per_query per S1.
    inject_matches: always include these IDs (training / recall safety net).
    """
    inject_matches = inject_matches or {}
    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    target_dict = target_df.set_index("entity_id").to_dict("index")

    out: Dict[str, Set[str]] = {}
    for s1_id in s1_df["entity_id"]:
        c_set = set(candidates.get(s1_id, set()))
        must = inject_matches.get(s1_id, set())
        c_set.update(must)

        if max_per_query is None or len(c_set) <= max_per_query:
            out[s1_id] = c_set
            continue

        q = s1_dict.get(s1_id, {})
        q_name = str(q.get("name_normalized", "") or "")
        q_addr = str(q.get("address_normalized", "") or "")
        q_name_t = _token_set(q_name)
        q_addr_t = _token_set(q_addr)

        scored: List[Tuple[float, str]] = []
        for cid in c_set:
            if cid in must:
                scored.append((1e9, cid))
                continue
            c = target_dict.get(cid, {})
            c_name = str(c.get("name_normalized", "") or "")
            c_addr = str(c.get("address_normalized", "") or "")
            sc = _cheap_rank_score(
                q_name, q_addr, c_name, c_addr,
                q_name_t, q_addr_t, _token_set(c_name), _token_set(c_addr),
            )
            scored.append((sc, cid))

        scored.sort(key=lambda x: (-x[0], x[1]))
        kept = {cid for _, cid in scored[:max_per_query]}
        kept.update(must)
        out[s1_id] = kept

    return out


def candidates_to_pairs(candidates: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for s1_id, c_set in candidates.items():
        for cid in c_set:
            pairs.append((s1_id, cid))
    return pairs


def candidate_recall_stats(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> dict:
    """Entity-level and link-level recall for true matches in candidate sets."""
    total_links = 0
    found_links = 0
    entities_miss = 0
    matched_entities = 0
    cand_counts = []

    for s1_id, true_set in ground_truth.items():
        if not true_set:
            continue
        matched_entities += 1
        c_set = candidates.get(s1_id, set())
        cand_counts.append(len(c_set))
        hit = true_set & c_set
        total_links += len(true_set)
        found_links += len(hit)
        if len(hit) < len(true_set):
            entities_miss += 1

    return {
        "link_recall": found_links / total_links if total_links else 1.0,
        "entities_with_any_miss": entities_miss,
        "matched_entities": matched_entities,
        "total_true_links": total_links,
        "found_true_links": found_links,
        "avg_candidates": float(np.mean(cand_counts)) if cand_counts else 0.0,
        "max_candidates": int(max(cand_counts)) if cand_counts else 0,
        "p95_candidates": float(np.percentile(cand_counts, 95)) if cand_counts else 0.0,
    }
