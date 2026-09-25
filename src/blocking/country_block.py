"""
Country blocking strategy.
"""

import logging
from typing import Dict, Set
import pandas as pd

from .base import BaseBlocker

logger = logging.getLogger(__name__)


class CountryBlocker(BaseBlocker):
    """Generates candidates based solely on country equality.
    
    WARNING: Generating the full Cartesian product for identical countries
    (e.g., all 1.3M US records × 3M S2 US records) is O(N*M) and infeasible.
    This blocker will limit candidate generation or serve as a filter.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.target_countries = {}
        
    def fit(self, source_df: pd.DataFrame):
        logger.info(f"Fitting CountryBlocker on {len(source_df)} records...")
        # To avoid O(N*M) explosions, we do not build a full inverted index here.
        # It's physically impossible to materialize the Cartesian product.
        pass
        
    def transform(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        logger.info("CountryBlocker transform called. Skipping full Cartesian product generation to avoid O(N*M) memory explosion.")
        return {}
