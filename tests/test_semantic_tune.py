"""Small checks for the fixed score mixtures used by the capacity experiment."""
# Tiny deterministic arrays isolate score mixing from model training.
from __future__ import annotations

import unittest

import numpy as np

from avito_test.semantic_blend import blended_scores, normalize_scores
from avito_test.semantic_tune_evaluate import controls
from avito_test.semantic_tune_final import final_scores


class _FixedModel:
    def __init__(self, scores: list[float]):
        self.scores = np.asarray(scores, dtype=np.float64)

    def predict(self, features: np.ndarray, thread_count: int = 6) -> np.ndarray:
        if len(features) != len(self.scores):
            raise ValueError("Fixture scores do not align with the full candidate pool")
        return self.scores


class SemanticTuneTests(unittest.TestCase):
    def test_each_query_pool_is_normalized_separately(self) -> None:
        arrays = {"features": np.zeros((6, 44), dtype=np.float32),
                  "offsets": np.asarray([0, 3, 6]),
                  "B3000": np.asarray([1., 2., 3., 100., 200., 300.]),
                  "B15000": np.asarray([3., 2., 1., 300., 200., 100.])}
        existing = controls(arrays)["submitted_blend"]
        self.assertTrue(np.allclose(existing[:3], existing[3:]))
        model = _FixedModel([0., 4., 8., 10., 20., 30.])
        scored = final_scores(arrays, model)
        expected = 0.5*normalize_scores(model.scores[:3]) + 0.5*normalize_scores(
            blended_scores(arrays["B3000"][:3], arrays["B15000"][:3], .75))
        self.assertTrue(np.allclose(scored["half_blend"][:3], expected))
        self.assertTrue(np.allclose(scored["half_blend"][:3],
                                    scored["half_blend"][3:]))


if __name__ == "__main__":
    unittest.main()
