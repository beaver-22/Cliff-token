"""
Cliff token detection.

A token at position t is a cliff token if the drop in estimated success
probability, p_{t-1} - p_t, exceeds the adaptive threshold
tau + z_alpha * SE_t of a one-sided two-proportion z-test
(N = 64 rollouts, tau = 0.1, z_alpha = 1.645; see cliff_threshold.py).

Scores are compared between consecutive non-null positions.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class CliffTokenInfo:
    """Information about a detected cliff token."""
    position: int              # 1-indexed position in the response
    token_str: Optional[str]
    token_id: Optional[int]
    prev_score: float
    curr_score: float
    drop_magnitude: float
    drop_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_valid_score_pairs(
    scores: List[Optional[float]],
) -> List[Tuple[int, float, int, float]]:
    """Extract consecutive non-None score pairs from a (possibly sparse) score array.

    With window=16, scores are [0.5, None, ..., None, 0.3, None, ...].
    This function yields pairs like (0, 0.5, 16, 0.3) — comparing scores
    at their actual token positions regardless of intervening Nones.

    Returns:
        List of (prev_idx, prev_score, curr_idx, curr_score) tuples.
    """
    valid = [(i, s) for i, s in enumerate(scores) if s is not None]
    pairs = []
    for j in range(1, len(valid)):
        prev_idx, prev_score = valid[j - 1]
        curr_idx, curr_score = valid[j]
        pairs.append((prev_idx, prev_score, curr_idx, curr_score))
    return pairs


def find_all_cliff_tokens_statistical(
    scores: List[Optional[float]],
    tokens: Optional[List[str]] = None,
    token_ids: Optional[List[int]] = None,
    N: int = 64,
) -> List[CliffTokenInfo]:
    """Statistical cliff detection via two-proportion z-test.

    Cliff at position t iff:
        Δ̂ > δ_0 + z_α · SE
        with N=64 rollouts, δ_0=0.1, α=0.05 (z_α≈1.645)

    Uses precomputed (N+1)×(N+1) lookup matrix for O(1) per-position checks.
    Recovers k from stored float score via round(score * N).
    """
    from src.analysis.cliff_threshold import get_default_matrix, score_to_k

    matrix = get_default_matrix()
    drops: List[CliffTokenInfo] = []

    for prev_idx, prev_score, curr_idx, curr_score in _get_valid_score_pairs(scores):
        k_prev = score_to_k(prev_score, N)
        k_curr = score_to_k(curr_score, N)
        if 0 <= k_prev < matrix.shape[0] and 0 <= k_curr < matrix.shape[1] \
                and matrix[k_prev, k_curr]:
            drops.append(CliffTokenInfo(
                position=curr_idx + 1,
                token_str=tokens[curr_idx] if tokens and curr_idx < len(tokens) else None,
                token_id=token_ids[curr_idx] if token_ids and curr_idx < len(token_ids) else None,
                prev_score=prev_score,
                curr_score=curr_score,
                drop_magnitude=prev_score - curr_score,
                drop_type="statistical_cliff",
            ))
    return drops


def find_first_cliff_token_statistical(
    scores: List[Optional[float]],
    tokens: Optional[List[str]] = None,
    token_ids: Optional[List[int]] = None,
    N: int = 64,
) -> Optional[CliffTokenInfo]:
    """First statistical cliff token (z-test based)."""
    from src.analysis.cliff_threshold import get_default_matrix, score_to_k

    matrix = get_default_matrix()
    for prev_idx, prev_score, curr_idx, curr_score in _get_valid_score_pairs(scores):
        k_prev = score_to_k(prev_score, N)
        k_curr = score_to_k(curr_score, N)
        if 0 <= k_prev < matrix.shape[0] and 0 <= k_curr < matrix.shape[1] \
                and matrix[k_prev, k_curr]:
            return CliffTokenInfo(
                position=curr_idx + 1,
                token_str=tokens[curr_idx] if tokens and curr_idx < len(tokens) else None,
                token_id=token_ids[curr_idx] if token_ids and curr_idx < len(token_ids) else None,
                prev_score=prev_score,
                curr_score=curr_score,
                drop_magnitude=prev_score - curr_score,
                drop_type="statistical_cliff",
            )
    return None
