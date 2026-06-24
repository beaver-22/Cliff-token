"""Statistical cliff token definition via two-proportion z-test.

Cliff token = position where the estimated potential drop is statistically
significant (one-sided two-proportion z-test against H0: drop ≤ δ_0).

Test statistic:
    Z = (Δ̂ − δ_0) / SE
    SE = sqrt(p̂_{t-1}(1-p̂_{t-1})/N + p̂_t(1-p̂_t)/N)   (unpooled)

Reject H0 (cliff) iff:
    Δ̂ > δ_0 + z_α · SE

Defaults:
    N = 64        rollout samples
    δ_0 = 0.1     minimum effect size
    α = 0.05      significance level → z_α ≈ 1.645 (one-sided 95%)

Score storage convention: floats in [0,1] (k/N). Recover k via round(score * N).
"""

import json
import math
from typing import Optional

import numpy as np


# ============================================================
# Defaults (match the protocol in the plan)
# ============================================================
DEFAULT_N = 64
DEFAULT_DELTA_0 = 0.1
DEFAULT_ALPHA = 0.05
DEFAULT_Z_ALPHA = 1.645   # one-sided 95%


# ============================================================
# Core test
# ============================================================

def is_cliff(
    k_prev: int,
    k_curr: int,
    N: int = DEFAULT_N,
    delta_0: float = DEFAULT_DELTA_0,
    z_alpha: float = DEFAULT_Z_ALPHA,
) -> bool:
    """Two-proportion z-test for cliff detection.

    Returns True iff Δ̂ = (k_prev/N - k_curr/N) > δ_0 + z_α · SE
    """
    if k_prev <= 0 or k_curr < 0 or k_prev > N or k_curr > N:
        return False
    p_prev = k_prev / N
    p_curr = k_curr / N
    delta = p_prev - p_curr
    if delta <= delta_0:
        return False  # not even larger than min effect
    var = p_prev * (1.0 - p_prev) / N + p_curr * (1.0 - p_curr) / N
    se = math.sqrt(var) if var > 0 else 0.0
    threshold = delta_0 + z_alpha * se
    return delta > threshold


def build_cliff_matrix(
    N: int = DEFAULT_N,
    delta_0: float = DEFAULT_DELTA_0,
    z_alpha: float = DEFAULT_Z_ALPHA,
) -> np.ndarray:
    """Pre-compute (N+1)×(N+1) boolean matrix M[k_prev, k_curr] = is_cliff."""
    size = N + 1
    M = np.zeros((size, size), dtype=bool)
    for k_prev in range(size):
        for k_curr in range(size):
            M[k_prev, k_curr] = is_cliff(k_prev, k_curr, N, delta_0, z_alpha)
    return M


def _min_drop_for_kp(
    k_prev: int, N: int, delta_0: float, z_alpha: float
) -> Optional[float]:
    """For a given k_prev, return the minimum (p_prev - p_curr) drop that triggers cliff."""
    if k_prev <= 0:
        return None
    for k_curr in range(k_prev, -1, -1):
        if is_cliff(k_prev, k_curr, N, delta_0, z_alpha):
            return round((k_prev - k_curr) / N, 4)
    return None


def save_cliff_matrix(
    M: np.ndarray,
    output_path: str,
    N: int = DEFAULT_N,
    delta_0: float = DEFAULT_DELTA_0,
    z_alpha: float = DEFAULT_Z_ALPHA,
):
    """Save matrix as JSON for inspection."""
    data = {
        "config": {
            "N": N,
            "delta_0": delta_0,
            "z_alpha": z_alpha,
            "alpha": DEFAULT_ALPHA,
            "matrix_shape": list(M.shape),
        },
        "summary": {
            "total_cells": int(M.size),
            "cliff_cells": int(M.sum()),
            "min_k_prev_with_cliff": int(np.min(np.where(M.any(axis=1))[0]))
            if M.any() else None,
        },
        "boundary_per_k_prev": [
            {
                "k_prev": kp,
                "p_prev": round(kp / N, 4),
                "min_drop_required": _min_drop_for_kp(kp, N, delta_0, z_alpha),
            }
            for kp in range(N + 1)
        ],
        "is_cliff_matrix": M.astype(int).tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================
# Module-level cached default matrix (built on first use)
# ============================================================

_DEFAULT_MATRIX: Optional[np.ndarray] = None


def get_default_matrix() -> np.ndarray:
    """Return cached default matrix (N=64, δ_0=0.1, z_α=1.645)."""
    global _DEFAULT_MATRIX
    if _DEFAULT_MATRIX is None:
        _DEFAULT_MATRIX = build_cliff_matrix()
    return _DEFAULT_MATRIX


def is_cliff_lookup(k_prev: int, k_curr: int) -> bool:
    """O(1) cliff check using cached default matrix."""
    M = get_default_matrix()
    if not (0 <= k_prev < M.shape[0] and 0 <= k_curr < M.shape[1]):
        return False
    return bool(M[k_prev, k_curr])


def score_to_k(score: Optional[float], N: int = DEFAULT_N) -> int:
    """Recover integer count k ∈ [0, N] from probability score (k/N)."""
    if score is None:
        return -1
    k = int(round(score * N))
    if k < 0:
        return 0
    if k > N:
        return N
    return k
