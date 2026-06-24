"""
Cliff Token Detection Module

Provides functions to detect Cliff Tokens and Critical Tokens in reasoning paths.

Window-aware: All functions compare consecutive NON-NULL scores,
correctly handling sparse score arrays from windowed rollout (e.g. window=16
produces scores at positions 0, 16, 32, ... with None elsewhere).

Cliff Token Definition:
- Token t where tokenwise-potential drops by >= threshold (default 0.20)
- i.e., score(prev) - score(t) >= threshold

Critical Token Definition (from paper):
- Token t where score(t) = 0 and all subsequent score(t+k) <= 0.05
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from src import config


@dataclass
class CliffTokenInfo:
    """Information about a detected Cliff Token."""
    position: int
    token_str: Optional[str]
    token_id: Optional[int]
    prev_score: float
    curr_score: float
    drop_magnitude: float
    drop_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias for backward compatibility
DropTokenInfo = CliffTokenInfo


@dataclass
class CriticalTokenInfo:
    """Information about a detected Critical Token (paper definition)."""
    position: int
    token_str: Optional[str]
    token_id: Optional[int]
    correctness_score: float
    scores_after: List[float]

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


def find_first_cliff_token(
    scores: List[Optional[float]],
    threshold: float = config.DEFAULT_CLIFF_THRESHOLD,
    tokens: Optional[List[str]] = None,
    token_ids: Optional[List[int]] = None,
) -> Optional[CliffTokenInfo]:
    """Find the first token where score drops by >= threshold."""
    for prev_idx, prev_score, curr_idx, curr_score in _get_valid_score_pairs(scores):
        drop = prev_score - curr_score
        if drop >= threshold:
            return CliffTokenInfo(
                position=curr_idx + 1,
                token_str=tokens[curr_idx] if tokens and curr_idx < len(tokens) else None,
                token_id=token_ids[curr_idx] if token_ids and curr_idx < len(token_ids) else None,
                prev_score=prev_score,
                curr_score=curr_score,
                drop_magnitude=drop,
                drop_type="first_cliff",
            )

    return None


# Backward-compatible alias
find_first_significant_drop = find_first_cliff_token


def find_max_cliff(
    scores: List[Optional[float]],
    tokens: Optional[List[str]] = None,
    token_ids: Optional[List[int]] = None,
) -> Optional[CliffTokenInfo]:
    """Find the token with the maximum single-step score drop."""
    max_drop = 0
    max_drop_info = None

    for prev_idx, prev_score, curr_idx, curr_score in _get_valid_score_pairs(scores):
        drop = prev_score - curr_score
        if drop > max_drop:
            max_drop = drop
            max_drop_info = CliffTokenInfo(
                position=curr_idx + 1,
                token_str=tokens[curr_idx] if tokens and curr_idx < len(tokens) else None,
                token_id=token_ids[curr_idx] if token_ids and curr_idx < len(token_ids) else None,
                prev_score=prev_score,
                curr_score=curr_score,
                drop_magnitude=drop,
                drop_type="max_cliff",
            )

    return max_drop_info


# Backward-compatible alias
find_max_drop = find_max_cliff


def find_all_cliff_tokens(
    scores: List[Optional[float]],
    threshold: float,
    tokens: Optional[List[str]] = None,
    token_ids: Optional[List[int]] = None,
) -> List[CliffTokenInfo]:
    """Find all tokens where score drops by >= threshold."""
    drops = []

    for prev_idx, prev_score, curr_idx, curr_score in _get_valid_score_pairs(scores):
        drop = prev_score - curr_score
        if drop >= threshold:
            drops.append(CliffTokenInfo(
                position=curr_idx + 1,
                token_str=tokens[curr_idx] if tokens and curr_idx < len(tokens) else None,
                token_id=token_ids[curr_idx] if token_ids and curr_idx < len(token_ids) else None,
                prev_score=prev_score,
                curr_score=curr_score,
                drop_magnitude=drop,
                drop_type=f"cliff_threshold_{threshold}",
            ))

    return drops


# Backward-compatible alias
find_all_drops_above_threshold = find_all_cliff_tokens


# ============================================================
# Statistical cliff detection (default for new code)
# ============================================================

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


def find_critical_token(
    scores: List[Optional[float]],
    threshold: float = config.CRITICAL_TOKEN_THRESHOLD,
    tokens: Optional[List[str]] = None,
    token_ids: Optional[List[int]] = None,
) -> Optional[CriticalTokenInfo]:
    """
    Find the Critical Token using the paper's definition.

    Critical Token is the FIRST scored position where:
    1. score = 0 (never leads to correct answer)
    2. All subsequent scored positions have score < threshold (default 5%)
    """
    if not scores:
        return None

    for i, score in enumerate(scores):
        if score is None:
            continue

        if score == 0:
            subsequent_valid = [s for s in scores[i + 1:] if s is not None]

            if not subsequent_valid or all(s < threshold for s in subsequent_valid):
                return CriticalTokenInfo(
                    position=i + 1,
                    token_str=tokens[i] if tokens and i < len(tokens) else None,
                    token_id=token_ids[i] if token_ids and i < len(token_ids) else None,
                    correctness_score=0.0,
                    scores_after=subsequent_valid[:10],
                )

    return None


def count_drops_by_threshold(
    scores: List[Optional[float]],
    thresholds: List[float] = None,
) -> Dict[float, int]:
    """Count the number of cliff drops at various thresholds."""
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    counts = {t: 0 for t in thresholds}

    for _, prev_score, _, curr_score in _get_valid_score_pairs(scores):
        drop = prev_score - curr_score
        for t in thresholds:
            if drop >= t:
                counts[t] += 1

    return counts
