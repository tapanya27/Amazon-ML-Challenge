"""
Official Macro F-0.5 Evaluation module for Entity Resolution.
Strictly adheres to ML Challenge 2026 Problem Statement.
"""

from typing import Dict, Set, Any
import numpy as np
import pandas as pd


def compute_entity_f05(pred_set: Set[str], true_set: Set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    
    Formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
        
    Singletons:
        - True singleton (empty true_set) and empty pred_set -> 1.0
        - True singleton (empty true_set) and non-empty pred_set -> 0.0
        - Non-singleton (non-empty true_set) and empty pred_set -> 0.0
    """
    if not true_set:
        # Ground truth is singleton
        return 1.0 if not pred_set else 0.0
        
    if not pred_set:
        # Missed all matches
        return 0.0
        
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
        
    prec = tp / len(pred_set)
    rec = tp / len(true_set)
    
    denom = (0.25 * prec) + rec
    if denom == 0:
        return 0.0
        
    return (1.25 * prec * rec) / denom


def evaluate_macro_f05(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Computes Macro-Averaged F_0.5 score across all Source 1 entities in ground_truth.
    
    Args:
        predictions: Dict mapping source1_entity_id -> set of predicted matching entity IDs
        ground_truth: Dict mapping source1_entity_id -> set of true matching entity IDs (empty set for singletons)
        
    Returns:
        Dict with 'macro_f05', 'mean_precision', 'mean_recall', 'singleton_accuracy'
    """
    scores = []
    precisions = []
    recalls = []
    singleton_scores = []
    matched_scores = []
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(pred_set, true_set)
        scores.append(score)
        
        if not true_set:
            singleton_scores.append(score)
        else:
            matched_scores.append(score)
            if pred_set:
                tp = len(pred_set & true_set)
                precisions.append(tp / len(pred_set))
                recalls.append(tp / len(true_set))
            else:
                precisions.append(0.0)
                recalls.append(0.0)
                
    return {
        'macro_f05': float(np.mean(scores)) if scores else 0.0,
        'mean_precision': float(np.mean(precisions)) if precisions else 0.0,
        'mean_recall': float(np.mean(recalls)) if recalls else 0.0,
        'singleton_accuracy': float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        'matched_f05': float(np.mean(matched_scores)) if matched_scores else 0.0,
        'num_entities': len(scores),
        'num_singletons': len(singleton_scores),
        'num_matched': len(matched_scores)
    }
