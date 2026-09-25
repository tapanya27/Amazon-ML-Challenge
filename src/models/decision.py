"""
Entity-level match decisions from pair-level classifier scores.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Set

import pandas as pd


def pairs_to_predictions(
    scored_pairs: pd.DataFrame,
    threshold: float,
    s1_col: str = "s1_id",
    cand_col: str = "candidate_id",
    score_col: str = "score",
    min_margin: float = 0.0,
    max_matches: int = 20,
    no_match_if_max_below: Optional[float] = None,
) -> Dict[str, Set[str]]:
    """
    For each S1, keep candidates with score >= threshold.
    If min_margin > 0 and exactly one match would pass for a singleton-like score
    distribution, require gap between best and second-best (optional conservative mode).
    """
    preds: Dict[str, Set[str]] = defaultdict(set)
    if scored_pairs.empty:
        return {}

    passing = scored_pairs[scored_pairs[score_col] >= threshold]
    for s1_id, grp in passing.groupby(s1_col):
        ordered = grp.sort_values(score_col, ascending=False)
        ids = ordered[cand_col].tolist()
        scores = ordered[score_col].tolist()
        selected = []
        for i, cid in enumerate(ids):
            if len(selected) >= max_matches:
                break
            if i == 0:
                selected.append(cid)
            elif min_margin <= 0 or scores[0] - scores[i] >= min_margin or scores[i] >= threshold:
                selected.append(cid)
        preds[s1_id] = set(selected)

    if no_match_if_max_below is not None and not scored_pairs.empty:
        max_by_s1 = scored_pairs.groupby(s1_col)[score_col].max()
        for s1_id, mx in max_by_s1.items():
            if mx < no_match_if_max_below:
                preds[s1_id] = set()

    return dict(preds)
