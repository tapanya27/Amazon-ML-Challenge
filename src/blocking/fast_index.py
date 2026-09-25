"""
Integer-index blocking + cheap pre-rank for large S2/S3 universes.

Stores row indices (int32) in inverted indexes. Does not rebuild a 10M-row
string dict per S1 batch. Caps posting traversal and candidate set size.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Re-export stopwords if token_block doesn't have NAME_STOPWORDS - it doesn't
# I'll define locally
LEGALISH = {
    "limited", "ltd", "company", "co", "inc", "incorporated", "corp", "corporation",
    "pvt", "private", "llc", "llp", "the", "and", "of", "for", "in", "at", "dba", "group",
}


class ReferenceIndex:
    """Read-only S2/S3 index built once."""

    def __init__(
        self,
        df: pd.DataFrame,
        max_freq_ratio: float = 0.01,
        max_posting_len: int = 4000,
        min_name_tok: int = 3,
        min_addr_tok: int = 2,
    ):
        n = len(df)
        self.n = n
        self.entity_ids = df["entity_id"].astype(str).to_numpy()
        self.names = df["name_normalized"].fillna("").astype(str).replace("nan", "").to_numpy()
        self.addrs = df["address_normalized"].fillna("").astype(str).replace("nan", "").to_numpy()
        countries = df["country_normalized"].fillna("").astype(str).str.lower().replace("nan", "")
        self.countries = countries.to_numpy()

        self.max_posting_len = max_posting_len
        max_freq = max(1, int(n * max_freq_ratio))

        logger.info("Building name token index on %d records...", n)
        self.name_index, self.name_df = self._build_index(self.names, min_name_tok, max_freq, LEGALISH)
        logger.info("  name tokens kept: %d", len(self.name_index))

        logger.info("Building address token index on %d records...", n)
        self.addr_index, self.addr_df = self._build_index(self.addrs, min_addr_tok, max_freq, set())
        logger.info("  address tokens kept: %d", len(self.addr_index))

        # exact-name buckets for fast path / retrieval (cap huge names)
        self.exact_name: Dict[str, np.ndarray] = {}
        buckets: Dict[str, List[int]] = defaultdict(list)
        for i, nm in enumerate(self.names):
            if nm:
                buckets[nm].append(i)
        for k, rows in buckets.items():
            if 1 <= len(rows) <= 200:
                self.exact_name[k] = np.asarray(rows, dtype=np.int32)
        logger.info("  exact-name keys (size<=200): %d", len(self.exact_name))

    def _build_index(
        self,
        texts: np.ndarray,
        min_len: int,
        max_freq: int,
        stop: Set[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        counts: Dict[str, int] = defaultdict(int)
        for text in texts:
            if not text:
                continue
            for tok in set(text.split()):
                if tok in stop:
                    continue
                if len(tok) < min_len and not tok.isdigit():
                    continue
                counts[tok] += 1
        valid = {t for t, c in counts.items() if c <= max_freq}
        postings: Dict[str, List[int]] = defaultdict(list)
        for i, text in enumerate(texts):
            if not text:
                continue
            for tok in set(text.split()):
                if tok in valid:
                    if len(postings[tok]) < self.max_posting_len:
                        postings[tok].append(i)
        arr_index = {t: np.asarray(v, dtype=np.int32) for t, v in postings.items()}
        dfreq = {t: counts[t] for t in arr_index}
        return arr_index, dfreq

    def retrieve(
        self,
        name: str,
        addr: str,
        country: str,
        retrieve_cap: int = 350,
        max_tokens: int = 8,
        require_country: bool = True,
    ) -> Tuple[np.ndarray, int, bool]:
        """
        Returns (row_indices, raw_unique_seen, hit_cap).
        Country filter applied while accumulating, not via Cartesian product.
        """
        scores: Dict[int, int] = {}
        country = (country or "").lower()

        def add_postings(tokens_sorted: List[str], index: Dict[str, np.ndarray], weight: int):
            used = 0
            for tok in tokens_sorted:
                posting = index.get(tok)
                if posting is None:
                    continue
                used += 1
                for rid in posting:
                    if require_country and country and self.countries[rid] != country:
                        continue
                    scores[rid] = scores.get(rid, 0) + weight
                if used >= max_tokens:
                    break
                if len(scores) >= retrieve_cap * 4:
                    break

        name_toks = []
        if name:
            name_toks = [
                t for t in set(name.split())
                if t in self.name_index and (len(t) >= 3 or t.isdigit()) and t not in LEGALISH
            ]
            name_toks.sort(key=lambda t: self.name_df.get(t, 10**9))
        addr_toks = []
        if addr:
            addr_toks = [
                t for t in set(addr.split())
                if t in self.addr_index and (len(t) >= 2 or t.isdigit())
            ]
            addr_toks.sort(key=lambda t: self.addr_df.get(t, 10**9))

        if name and name in self.exact_name:
            for rid in self.exact_name[name]:
                if require_country and country and self.countries[rid] != country:
                    continue
                scores[rid] = scores.get(rid, 0) + 50

        add_postings(name_toks, self.name_index, 3)
        add_postings(addr_toks, self.addr_index, 2)

        if not scores:
            return np.empty(0, dtype=np.int32), 0, False

        raw_n = len(scores)
        hit_cap = raw_n >= retrieve_cap
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:retrieve_cap]
        return np.asarray([r for r, _ in ranked], dtype=np.int32), raw_n, hit_cap

    def cheap_prerank(
        self,
        q_name: str,
        q_addr: str,
        q_country: str,
        rows: np.ndarray,
        top_k: int,
    ) -> np.ndarray:
        if rows.size == 0:
            return rows
        qn = set(q_name.split()) if q_name else set()
        qa = set(q_addr.split()) if q_addr else set()
        q_country = (q_country or "").lower()
        scored = []
        for rid in rows:
            cname = self.names[rid]
            caddr = self.addrs[rid]
            cn = set(cname.split()) if cname else set()
            ca = set(caddr.split()) if caddr else set()
            name_ov = len(qn & cn)
            addr_ov = len(qa & ca)
            sc = name_ov * 3 + addr_ov * 4
            if q_name and cname == q_name:
                sc += 40
            if q_addr and caddr == q_addr:
                sc += 40
            if q_country and self.countries[rid] == q_country:
                sc += 5
            scored.append((sc, int(rid)))
        scored.sort(key=lambda x: (-x[0], x[1]))
        keep = [r for _, r in scored[:top_k]]
        return np.asarray(keep, dtype=np.int32)


def cheap_prerank_batch(
    index: ReferenceIndex,
    s1_names: Sequence[str],
    s1_addrs: Sequence[str],
    s1_countries: Sequence[str],
    retrieved: Sequence[np.ndarray],
    top_k: int,
) -> List[np.ndarray]:
    return [
        index.cheap_prerank(n, a, c, rows, top_k)
        for n, a, c, rows in zip(s1_names, s1_addrs, s1_countries, retrieved)
    ]
