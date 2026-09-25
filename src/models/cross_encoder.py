"""
Cross-encoder re-ranking module.
"""

import logging
import pandas as pd
import numpy as np
import torch
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoModelForSequenceClassification, AutoTokenizer = None, None

logger = logging.getLogger(__name__)

class CrossEncoderReRanker:
    """Re-ranks candidates using a cross-encoder model."""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', batch_size: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if AutoModelForSequenceClassification is None:
            logger.warning("transformers not installed. Cross-encoder will fail.")
            self.model = None
            self.tokenizer = None
        else:
            logger.info(f"Loading cross-encoder: {self.model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
    def rerank(self, query_texts: list, candidate_texts: list) -> np.ndarray:
        """
        Scores pairs of query and candidate texts.
        
        Returns:
            np.ndarray of scores
        """
        if not self.model:
            raise RuntimeError("Cross-encoder model not loaded.")
            
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(query_texts), self.batch_size):
                q_batch = query_texts[i:i+self.batch_size]
                c_batch = candidate_texts[i:i+self.batch_size]
                
                features = self.tokenizer(
                    q_batch, 
                    c_batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                )
                
                features = {k: v.to(self.device) for k, v in features.items()}
                
                scores = self.model(**features).logits.squeeze(-1)
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)
                
                all_scores.append(scores.cpu().numpy())
                
        return np.concatenate(all_scores)

