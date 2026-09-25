"""
Base blocker interface for Entity Resolution.
"""

from typing import Dict, Set
import pandas as pd


class BaseBlocker:
    """Base class for all blocking strategies."""
    
    def __init__(self, config: dict):
        self.config = config
    
    def fit(self, source_df: pd.DataFrame):
        """Fit the blocker on the target database (e.g. S2 or S3).
        
        Args:
            source_df: DataFrame containing the target records.
        """
        raise NotImplementedError
    
    def transform(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Find candidates for the query records.
        
        Args:
            query_df: DataFrame containing the query records (e.g. S1).
            
        Returns:
            Dictionary mapping query entity_id to a set of candidate entity_ids.
        """
        raise NotImplementedError

