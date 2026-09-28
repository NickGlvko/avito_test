"""Six fixed, training-free combinations of frozen complete-pool rankers."""
# Fusion consumes complete ranked lists; top-50-only input would lose recoverable items.
from __future__ import annotations

import numpy as np

from .semantic_blend import normalize_scores, rank_all


ALPHAS = (0.25, 0.50, 0.75)
BETAS = (0.50, 0.75, 0.90)
RRF_K = 60
CONTROL = "D_full_unweighted"
VARIANTS = (CONTROL, *(f"d_blend_{alpha:.2f}" for alpha in ALPHAS),
            *(f"rrf_bd_{beta:.2f}" for beta in BETAS))


def blend_d(unweighted: np.ndarray, balanced: np.ndarray,
            alpha: float) -> np.ndarray:
    """Alpha is the weight of the submitted, unweighted D ranker."""
    if alpha not in ALPHAS and alpha not in (0.0, 1.0):
        raise ValueError(f"Unexpected D blend weight: {alpha}")
    if np.shape(unweighted) != np.shape(balanced):
        raise ValueError("D score vectors are not aligned")
    return (alpha * normalize_scores(unweighted) +
            (1.0 - alpha) * normalize_scores(balanced))


def full_ranks(ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Return one-based ranks aligned with ascending corpus row IDs."""
    ids = np.asarray(ids)
    scores = np.asarray(scores, dtype=np.float64)
    if (ids.ndim != 1 or scores.shape != ids.shape or not len(ids) or
            not np.all(ids[1:] > ids[:-1]) or not np.isfinite(scores).all()):
        raise ValueError("A complete rank requires sorted unique IDs and finite scores")
    order = np.lexsort((ids, -scores))
    ranks = np.empty(len(ids), dtype=np.int32)
    ranks[order] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return ranks


def rrf_bd(b_ids: np.ndarray, b_scores: np.ndarray,
           d_ids: np.ndarray, d_scores: np.ndarray,
           beta: float) -> np.ndarray:
    """RRF on the union of complete lists, aligned with sorted D item rows."""
    if beta not in BETAS and beta not in (0.0, 1.0):
        raise ValueError(f"Unexpected B/D RRF weight: {beta}")
    b_rank = full_ranks(b_ids, b_scores)
    d_rank = full_ranks(d_ids, d_scores)
    positions = np.searchsorted(d_ids, b_ids)
    if np.any(positions == len(d_ids)) or not np.array_equal(d_ids[positions], b_ids):
        raise ValueError("The frozen B candidate pool is not a subset of D")
    fused = beta / (RRF_K + d_rank.astype(np.float64))
    fused[positions] += (1.0 - beta) / (RRF_K + b_rank.astype(np.float64))
    return fused


def select(ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return rank_all(ids, scores)[:50]
