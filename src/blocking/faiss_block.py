"""
FAISS-based embedding blocker using BGE-M3.
"""

import logging
from typing import Dict, Set
import numpy as np
import pandas as pd
import faiss
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .base import BaseBlocker

logger = logging.getLogger(__name__)


class FaissBlocker(BaseBlocker):
    """Retrieves candidates using dense embeddings and FAISS nearest neighbor search."""
    
    def __init__(self, config: dict, column: str = 'name_normalized', top_k: int = 20, batch_size: int = 512):
        super().__init__(config)
        self.column = column
        self.top_k = top_k
        self.batch_size = batch_size
        self.index = None
        self.target_ids = None
        
        self.model_name = config.get('embedding_model', 'BAAI/bge-m3')
        if SentenceTransformer:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        else:
            self.model = None
            logger.warning("sentence-transformers not installed. FaissBlocker will fail.")
            
    def _get_embeddings(self, texts: list) -> np.ndarray:
        if not self.model:
            raise RuntimeError("SentenceTransformer not initialized.")
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size, 
            show_progress_bar=False,
            normalize_embeddings=True # Crucial for cosine similarity with Inner Product index
        )
        return embeddings.astype('float32')
        
    def fit(self, source_df: pd.DataFrame):
        """Fit the FAISS index on target embeddings."""
        logger.info(f"Fitting FaissBlocker on {self.column} for {len(source_df)} records...")
        texts = source_df[self.column].fillna('').tolist()
        self.target_ids = source_df['entity_id'].values
        
        # In a real distributed system, we'd pre-compute these and load them.
        logger.info(f"Computing embeddings for {len(texts)} targets (this may take a while)...")
        target_embeddings = self._get_embeddings(texts)
        
        d = target_embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity since embeddings are normalized
        self.index = faiss.IndexFlatIP(d)
        
        # If we have GPU, we could move index to GPU:
        # res = faiss.StandardGpuResources()
        # self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        logger.info("Adding embeddings to FAISS index...")
        self.index.add(target_embeddings)
        logger.info("FAISS index built.")
        
    def transform(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Retrieve top-K candidates."""
        logger.info(f"Retrieving top-{self.top_k} candidates using FaissBlocker...")
        
        candidates = {}
        texts = query_df[self.column].fillna('').tolist()
        query_ids = query_df['entity_id'].values
        
        query_embeddings = self._get_embeddings(texts)
        
        logger.info("Searching FAISS index...")
        distances, indices = self.index.search(query_embeddings, self.top_k)
        
        for i, q_id in enumerate(query_ids):
            q_cands = set()
            for j in range(self.top_k):
                idx = indices[i, j]
                if idx != -1: # -1 means no neighbor found (e.g. if k > n_targets)
                    q_cands.add(self.target_ids[idx])
            candidates[q_id] = q_cands
            
        logger.info("FAISS search complete.")
        return candidates
