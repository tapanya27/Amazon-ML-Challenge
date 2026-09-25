"""
Optimized pair-wise feature engineering for Entity Resolution.
Features designed to achieve extreme precision (F_0.5 > 0.90) by capturing
lexical, syntactic, address-number, token-set, and cross-field signals.
"""

import logging
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


def get_ngrams(text: str, n: int = 3) -> set:
    if not text:
        return set()
    text = f" {text} "
    return set(text[i:i+n] for i in range(len(text)-n+1))

def ngram_sim(text1: str, text2: str, n: int = 3) -> float:
    s1 = get_ngrams(text1, n)
    s2 = get_ngrams(text2, n)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / min(len(s1), len(s2))

def token_features(t1: set, t2: set) -> Tuple[float, float]:
    if not t1 or not t2:
        return 0.0, 0.0
    intersection = len(t1 & t2)
    jaccard = intersection / len(t1 | t2)
    overlap = intersection / min(len(t1), len(t2))
    return jaccard, overlap

def extract_numbers(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r'\d+', text))


class PairFeatureGenerator:
    """Generates features for a set of candidate pairs."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self._s1_lookup = None
        self._tgt_lookup = None

    def bind_lookups(self, s1_lookup: dict = None, tgt_lookup: dict = None):
        """Optional small S1 lookup; do not pass a 10M-row target dict."""
        if s1_lookup is not None:
            self._s1_lookup = s1_lookup
        if tgt_lookup is not None:
            self._tgt_lookup = tgt_lookup

    def generate_from_index(self, index, s1_ids, s1_names, s1_addrs, s1_countries, row_lists) -> pd.DataFrame:
        """Features for TOP_K integer rows using cached reference arrays (no 10M dict)."""
        features = []
        n_pairs = sum(len(r) for r in row_lists)
        logger.info("Generating features for %d candidate pairs (index path)...", n_pairs)
        for s1_id, qn, qa, qc, rows in zip(s1_ids, s1_names, s1_addrs, s1_countries, row_lists):
            qn = qn if qn and qn != "nan" else ""
            qa = qa if qa and qa != "nan" else ""
            qc = (qc or "").lower()
            q_name_words = qn.split()
            qt_name = set(q_name_words)
            qt_addr = set(qa.split()) if qa else set()
            q_nums = extract_numbers(qa)
            recs = {r[0]: r for r in index.fetch_rows([int(x) for x in rows])}
            for rid in rows:
                rid = int(rid)
                rec = recs.get(rid)
                if not rec:
                    continue
                _i, eid, cn, ca, cc = rec
                features.append(
                    self._one_pair(s1_id, eid, qn, cn or "", qa, ca or "", qc, (cc or "").lower(),
                                   q_name_words, qt_name, qt_addr, q_nums)
                )
        return pd.DataFrame(features)

    def _one_pair(self, q_id, c_id, qn_name, cn_name, qn_addr, cn_addr, q_country, c_country,
                  q_name_words, qt_name, qt_addr, q_nums):
        cn_name = cn_name if cn_name and cn_name != "nan" else ""
        cn_addr = cn_addr if cn_addr and cn_addr != "nan" else ""
        c_name_words = cn_name.split()
        ct_name = set(c_name_words)
        ct_addr = set(cn_addr.split()) if cn_addr else set()

        n_exact = 1.0 if qn_name == cn_name and qn_name else 0.0
        n_lev = fuzz.ratio(qn_name, cn_name) / 100.0 if qn_name and cn_name else 0.0
        n_token_sort = fuzz.token_sort_ratio(qn_name, cn_name) / 100.0 if qn_name and cn_name else 0.0
        n_token_set = fuzz.token_set_ratio(qn_name, cn_name) / 100.0 if qn_name and cn_name else 0.0
        n_jaccard, n_overlap = token_features(qt_name, ct_name)
        n_ngram = ngram_sim(qn_name, cn_name, n=3)
        first_token_match = 1.0 if (q_name_words and c_name_words and q_name_words[0] == c_name_words[0]) else 0.0
        n_len_diff = abs(len(qn_name) - len(cn_name))
        n_len_ratio = n_len_diff / max(len(qn_name), len(cn_name), 1)

        a_exact = 1.0 if qn_addr == cn_addr and qn_addr else 0.0
        a_lev = fuzz.ratio(qn_addr, cn_addr) / 100.0 if qn_addr and cn_addr else 0.0
        a_token_set = fuzz.token_set_ratio(qn_addr, cn_addr) / 100.0 if qn_addr and cn_addr else 0.0
        a_jaccard, a_overlap = token_features(qt_addr, ct_addr)
        a_ngram = ngram_sim(qn_addr, cn_addr, n=3)
        a_len_diff = abs(len(qn_addr) - len(cn_addr))
        a_len_ratio = a_len_diff / max(len(qn_addr), len(cn_addr), 1)
        c_nums = extract_numbers(cn_addr)
        num_overlap_count = len(q_nums & c_nums)
        num_overlap_ratio = num_overlap_count / max(len(q_nums), len(c_nums), 1) if (q_nums or c_nums) else 1.0
        num_conflict = 1.0 if (q_nums and c_nums and not (q_nums & c_nums)) else 0.0
        country_same = 1.0 if (q_country == c_country and q_country) else 0.0
        tot_jaccard, tot_overlap = token_features(qt_name | qt_addr, ct_name | ct_addr)
        name_addr_prod = n_lev * a_lev
        token_sort_prod = n_token_sort * a_token_set
        composite_score = (0.5 * n_token_set) + (0.5 * a_token_set)
        return {
            "s1_id": q_id,
            "candidate_id": c_id,
            "name_exact": n_exact,
            "name_levenshtein": n_lev,
            "name_token_sort": n_token_sort,
            "name_token_set": n_token_set,
            "name_jaccard": n_jaccard,
            "name_overlap": n_overlap,
            "name_ngram_sim": n_ngram,
            "name_first_token": first_token_match,
            "name_len_diff": n_len_diff,
            "name_len_ratio": n_len_ratio,
            "addr_exact": a_exact,
            "addr_levenshtein": a_lev,
            "addr_token_set": a_token_set,
            "addr_jaccard": a_jaccard,
            "addr_overlap": a_overlap,
            "addr_ngram_sim": a_ngram,
            "addr_len_diff": a_len_diff,
            "addr_len_ratio": a_len_ratio,
            "addr_num_overlap": num_overlap_count,
            "addr_num_ratio": num_overlap_ratio,
            "addr_num_conflict": num_conflict,
            "country_same": country_same,
            "total_jaccard": tot_jaccard,
            "total_overlap": tot_overlap,
            "name_addr_prod": name_addr_prod,
            "token_sort_prod": token_sort_prod,
            "composite_score": composite_score,
        }

    def generate_features(self, s1_df: pd.DataFrame, target_df: pd.DataFrame, candidate_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Generates feature matrix for the given candidate pairs.
        
        Args:
            s1_df: Query dataframe (Source 1)
            target_df: Target dataframe (Source 2 or Source 3 or combined)
            candidate_pairs: List of (s1_id, candidate_id) tuples
            
        Returns:
            DataFrame containing feature vectors for each pair.
        """
        total_pairs = len(candidate_pairs)
        if total_pairs == 0:
            return pd.DataFrame()
            
        logger.info(f"Generating features for {total_pairs} candidate pairs...")
        s1_lookup = self._s1_lookup
        tgt_lookup = self._tgt_lookup
        if s1_lookup is None:
            s1_lookup = s1_df.set_index("entity_id").to_dict("index")
        if tgt_lookup is None:
            tgt_lookup = target_df.set_index("entity_id").to_dict("index")
        
        features = []
        
        for idx, (q_id, c_id) in enumerate(candidate_pairs):
            q = s1_lookup.get(q_id)
            c = tgt_lookup.get(c_id)
            
            if not q or not c:
                continue
                
            qn_name = str(q.get('name_normalized', ''))
            cn_name = str(c.get('name_normalized', ''))
            qn_addr = str(q.get('address_normalized', ''))
            cn_addr = str(c.get('address_normalized', ''))
            
            if qn_name == 'nan': qn_name = ''
            if cn_name == 'nan': cn_name = ''
            if qn_addr == 'nan': qn_addr = ''
            if cn_addr == 'nan': cn_addr = ''
            
            # Words
            q_name_words = qn_name.split()
            c_name_words = cn_name.split()
            qt_name = set(q_name_words)
            ct_name = set(c_name_words)
            
            qt_addr = set(qn_addr.split())
            ct_addr = set(cn_addr.split())
            
            # --- 1. NAME FEATURES ---
            n_exact = 1.0 if qn_name == cn_name and qn_name else 0.0
            n_lev = fuzz.ratio(qn_name, cn_name) / 100.0 if qn_name and cn_name else 0.0
            n_token_sort = fuzz.token_sort_ratio(qn_name, cn_name) / 100.0 if qn_name and cn_name else 0.0
            n_token_set = fuzz.token_set_ratio(qn_name, cn_name) / 100.0 if qn_name and cn_name else 0.0
            n_jaccard, n_overlap = token_features(qt_name, ct_name)
            n_ngram = ngram_sim(qn_name, cn_name, n=3)
            
            # First token match
            first_token_match = 1.0 if (q_name_words and c_name_words and q_name_words[0] == c_name_words[0]) else 0.0
            
            # Length dynamics
            n_len_diff = abs(len(qn_name) - len(cn_name))
            n_len_ratio = n_len_diff / max(len(qn_name), len(cn_name), 1)
            
            # --- 2. ADDRESS FEATURES ---
            a_exact = 1.0 if qn_addr == cn_addr and qn_addr else 0.0
            a_lev = fuzz.ratio(qn_addr, cn_addr) / 100.0 if qn_addr and cn_addr else 0.0
            a_token_set = fuzz.token_set_ratio(qn_addr, cn_addr) / 100.0 if qn_addr and cn_addr else 0.0
            a_jaccard, a_overlap = token_features(qt_addr, ct_addr)
            a_ngram = ngram_sim(qn_addr, cn_addr, n=3)
            
            a_len_diff = abs(len(qn_addr) - len(cn_addr))
            a_len_ratio = a_len_diff / max(len(qn_addr), len(cn_addr), 1)
            
            # Number match / mismatch (street & house numbers)
            q_nums = extract_numbers(qn_addr)
            c_nums = extract_numbers(cn_addr)
            num_overlap_count = len(q_nums & c_nums)
            num_overlap_ratio = num_overlap_count / max(len(q_nums), len(c_nums), 1) if (q_nums or c_nums) else 1.0
            num_conflict = 1.0 if (q_nums and c_nums and not (q_nums & c_nums)) else 0.0
            
            # --- 3. CROSS-FIELD & COUNTRY FEATURES ---
            q_country = str(q.get('country_normalized', '')).lower()
            c_country = str(c.get('country_normalized', '')).lower()
            country_same = 1.0 if (q_country == c_country and q_country) else 0.0
            
            tot_jaccard, tot_overlap = token_features(qt_name | qt_addr, ct_name | ct_addr)
            
            name_addr_prod = n_lev * a_lev
            token_sort_prod = n_token_sort * a_token_set
            composite_score = (0.5 * n_token_set) + (0.5 * a_token_set)
            
            features.append({
                's1_id': q_id,
                'candidate_id': c_id,
                'name_exact': n_exact,
                'name_levenshtein': n_lev,
                'name_token_sort': n_token_sort,
                'name_token_set': n_token_set,
                'name_jaccard': n_jaccard,
                'name_overlap': n_overlap,
                'name_ngram_sim': n_ngram,
                'name_first_token': first_token_match,
                'name_len_diff': n_len_diff,
                'name_len_ratio': n_len_ratio,
                'addr_exact': a_exact,
                'addr_levenshtein': a_lev,
                'addr_token_set': a_token_set,
                'addr_jaccard': a_jaccard,
                'addr_overlap': a_overlap,
                'addr_ngram_sim': a_ngram,
                'addr_len_diff': a_len_diff,
                'addr_len_ratio': a_len_ratio,
                'addr_num_overlap': num_overlap_count,
                'addr_num_ratio': num_overlap_ratio,
                'addr_num_conflict': num_conflict,
                'country_same': country_same,
                'total_jaccard': tot_jaccard,
                'total_overlap': tot_overlap,
                'name_addr_prod': name_addr_prod,
                'token_sort_prod': token_sort_prod,
                'composite_score': composite_score
            })
            
            if (idx + 1) % 50000 == 0 or (idx + 1) == total_pairs:
                logger.info(f"  Generated features for {idx + 1}/{total_pairs} pairs...")
                
        logger.info(f"Generated {len(features)} feature rows successfully.")
        return pd.DataFrame(features)
