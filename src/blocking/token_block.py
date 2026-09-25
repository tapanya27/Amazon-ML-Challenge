"""
Token-based blocking using inverted indexes.
"""

from collections import defaultdict
from typing import Dict, Set, List
import logging
import pandas as pd

from .base import BaseBlocker

logger = logging.getLogger(__name__)


class TokenBlocker(BaseBlocker):
    """Generates candidates based on shared informative tokens."""
    
    def __init__(
        self,
        config: dict,
        column: str,
        min_token_len: int = 3,
        max_freq_ratio: float = 0.01,
        stopwords: set = None,
        max_candidates_per_query: int = 400,
        max_posting_len: int = 8000,
        max_tokens_used: int = 10,
    ):
        super().__init__(config)
        self.column = column
        self.min_token_len = min_token_len
        self.max_freq_ratio = max_freq_ratio
        self.stopwords = stopwords or set()
        self.max_candidates_per_query = max_candidates_per_query
        self.max_posting_len = max_posting_len
        self.max_tokens_used = max_tokens_used

        self.inverted_index = defaultdict(list)
        self.target_size = 0
        self.token_counts = defaultdict(int)
        self.valid_tokens = set()
    
    def _tokenize(self, text: str) -> List[str]:
        if not text or pd.isna(text):
            return []
        tokens = str(text).split()
        # Keep tokens >= min length, or numeric tokens if they are shorter (for addresses)
        return [
            t for t in tokens 
            if (len(t) >= self.min_token_len or t.isdigit()) 
            and t not in self.stopwords
        ]
        
    def fit(self, source_df: pd.DataFrame):
        """Build the inverted index on the target dataset."""
        logger.info(f"Building TokenBlocker index on {self.column} for {len(source_df)} records...")
        self.target_size = len(source_df)
        self.inverted_index.clear()
        self.token_counts.clear()
        self.valid_tokens.clear()
        
        max_allowed_freq = max(1, int(self.target_size * self.max_freq_ratio))
        
        # First pass: count frequencies
        # We can optimize this by combining steps if memory allows, but two passes is safer
        logger.info("  Counting token frequencies...")
        for text in source_df[self.column]:
            for token in set(self._tokenize(text)):
                self.token_counts[token] += 1
                
        # Filter tokens
        for token, count in self.token_counts.items():
            if count <= max_allowed_freq:
                self.valid_tokens.add(token)
                
        logger.info(f"  Kept {len(self.valid_tokens)} valid tokens (filtered {len(self.token_counts) - len(self.valid_tokens)} overly common/stopwords)")
        
        # Second pass: build index
        logger.info("  Populating inverted index...")
        for entity_id, text in zip(source_df['entity_id'], source_df[self.column]):
            for token in set(self._tokenize(text)):
                if token in self.valid_tokens:
                    self.inverted_index[token].append(entity_id)
                    
        logger.info("  TokenBlocker index built successfully.")
    
    def transform(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Retrieve candidates for query dataset."""
        logger.info(f"Retrieving candidates using TokenBlocker on {self.column} for {len(query_df)} queries...")
        candidates = defaultdict(set)
        
        found_queries = 0
        total_candidates = 0
        
        for q_id, text in zip(query_df['entity_id'], query_df[self.column]):
            tokens = [t for t in set(self._tokenize(text)) if t in self.valid_tokens]
            tokens.sort(key=lambda t: self.token_counts.get(t, 10**9))
            overlap = defaultdict(int)
            used = 0
            for token in tokens:
                posting = self.inverted_index.get(token)
                if not posting:
                    continue
                if len(posting) > self.max_posting_len:
                    continue
                used += 1
                for eid in posting:
                    overlap[eid] += 1
                if used >= self.max_tokens_used:
                    break
            if overlap:
                ranked = sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0]))
                query_candidates = {eid for eid, _ in ranked[: self.max_candidates_per_query]}
                candidates[q_id] = query_candidates
                found_queries += 1
                total_candidates += len(query_candidates)
                
        logger.info(f"  TokenBlocker found candidates for {found_queries}/{len(query_df)} queries.")
        logger.info(f"  Total candidate pairs generated: {total_candidates}")
        
        return dict(candidates)
