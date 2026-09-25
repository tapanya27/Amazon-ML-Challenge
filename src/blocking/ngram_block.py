"""
Character n-gram blocking using TF-IDF and chunked cosine similarity.
Memory-safe implementation with small chunking to prevent OOM errors on large corpora.
"""

import logging
from typing import Dict, Set
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseBlocker

logger = logging.getLogger(__name__)


class NgramBlocker(BaseBlocker):
    """Retrieves top-K candidates based on character n-gram TF-IDF cosine similarity."""
    
    def __init__(self, config: dict = None, column: str = 'name_normalized', n: int = 3, top_k: int = 20, batch_size: int = 250):
        super().__init__(config or {})
        self.column = column
        self.n = n
        self.top_k = top_k
        self.batch_size = batch_size
        
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(self.n, self.n),
            min_df=2,
            max_df=0.5,
            lowercase=True
        )
        self.target_matrix = None
        self.target_ids = None
        
    def fit(self, source_df: pd.DataFrame):
        """Fit the TF-IDF vectorizer and transform the target dataset."""
        logger.info(f"Fitting NgramBlocker (char_{self.n}gram) on {self.column} for {len(source_df)} records...")
        
        texts = source_df[self.column].fillna('').astype(str)
        self.target_ids = source_df['entity_id'].values
        
        # Fit and transform target matrix
        self.target_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"  Target matrix shape: {self.target_matrix.shape}")
        
    def transform(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Retrieve top-K candidates for each query using chunked matrix multiplication."""
        logger.info(f"Retrieving top-{self.top_k} candidates using NgramBlocker on {self.column} for {len(query_df)} queries...")
        
        candidates = {}
        query_texts = query_df[self.column].fillna('').astype(str)
        query_ids = query_df['entity_id'].values
        
        total_queries = len(query_df)
        
        for start_idx in range(0, total_queries, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_queries)
            
            chunk_texts = query_texts.iloc[start_idx:end_idx]
            query_matrix = self.vectorizer.transform(chunk_texts)
            
            # Cosine similarity in small chunks (chunk_size x num_targets)
            sim_matrix = query_matrix.dot(self.target_matrix.T).tocsr()
            
            for i in range(sim_matrix.shape[0]):
                q_id = query_ids[start_idx + i]
                row_start = sim_matrix.indptr[i]
                row_end = sim_matrix.indptr[i+1]
                
                if row_start == row_end:
                    continue
                    
                row_indices = sim_matrix.indices[row_start:row_end]
                row_data = sim_matrix.data[row_start:row_end]
                
                # Filter out negligible similarity < 0.20
                valid_mask = row_data >= 0.20
                if not np.any(valid_mask):
                    continue
                row_indices = row_indices[valid_mask]
                row_data = row_data[valid_mask]
                
                if len(row_indices) > self.top_k:
                    top_k_idx = np.argpartition(row_data, -self.top_k)[-self.top_k:]
                    best_indices = row_indices[top_k_idx]
                else:
                    best_indices = row_indices
                
                candidates[q_id] = {self.target_ids[idx] for idx in best_indices}
                
            if end_idx % (self.batch_size * 20) == 0 or end_idx == total_queries:
                logger.info(f"  Processed {end_idx}/{total_queries} queries...")
                
        logger.info(f"  NgramBlocker finished. Generated candidates for {len(candidates)} queries.")
        return candidates
