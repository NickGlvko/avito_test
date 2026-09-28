"""Fixed, query-wise blends of the frozen B3000 and B15000 rankers."""
# Normalize scores within each complete query pool before mixing models.
from __future__ import annotations

import numpy as np


ALPHAS = (0.25, 0.5, 0.75)
CONTROL = "B15000"


def variant(alpha: float) -> str:
    if alpha not in ALPHAS:
        raise ValueError(f"Unapproved blend weight: {alpha}")
    return f"blend_{alpha:.2f}"


def normalize_scores(values: np.ndarray) -> np.ndarray:
    """Standardize a model's scores over one complete query candidate pool."""
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("A query needs a nonempty vector of finite model scores")
    centered = scores - scores.mean(dtype=np.float64)
    deviation = np.sqrt(np.mean(centered * centered, dtype=np.float64))
    return centered / deviation if deviation > 0 else np.zeros_like(scores)


def blended_scores(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    if alpha not in ALPHAS or np.shape(old) != np.shape(new):
        raise ValueError("Blend weight or aligned score vector is invalid")
    return (1.0 - alpha) * normalize_scores(old) + alpha * normalize_scores(new)


def rank_all(ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Sort all candidate rows by score, with deterministic item-row tie breaks."""
    if len(ids) != len(scores):
        raise ValueError("Candidate rows and scores are misaligned")
    return ids[np.lexsort((ids, -scores))]


def score_candidate_pool(retriever, semantic, models: dict, query: dict,
                         relevant: np.ndarray, augmented: bool):
    """Score every old-pool candidate once with each frozen B ranker."""
    from .ranker import TEXT_FEATURES, pool_ids
    from .semantic_search import semantic_features

    sources, text_scores, allowed = retriever.retrieve_with_scores(
        query, relevant, augmented)
    ids = pool_ids(sources)
    text_features = retriever.features(query, ids, sources, text_scores, allowed)
    e5_features = semantic_features(semantic, query, ids, sources, False)
    features = np.column_stack((text_features[:, :len(TEXT_FEATURES)], e5_features))
    old = np.asarray(models["B3000"].predict(features, thread_count=1), dtype=np.float64)
    new = np.asarray(models["B15000"].predict(features, thread_count=1), dtype=np.float64)
    return ids, sources, old, new
