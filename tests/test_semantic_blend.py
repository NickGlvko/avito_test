"""Contracts for full-pool, query-wise blending of frozen ranker scores."""
# Both score arrays refer to one unchanged candidate pool per query.
import unittest

import numpy as np

from avito_test.semantic_blend import blended_scores, normalize_scores, rank_all


class ScoreBlendContracts(unittest.TestCase):
    def test_normalization_is_invariant_to_positive_affine_score_scales(self):
        scores = np.array([-2.0, 0.5, 1.5, 9.0])
        np.testing.assert_allclose(normalize_scores(scores),
                                   normalize_scores(7.0 * scores + 21.0), atol=1e-14)

    def test_constant_model_scores_contribute_zero(self):
        old = np.full(4, 42.0)
        new = np.array([0.0, 2.0, 4.0, 8.0])
        np.testing.assert_array_equal(normalize_scores(old), np.zeros(4))
        np.testing.assert_allclose(blended_scores(old, new, .25),
                                   .25 * normalize_scores(new))

    def test_scoring_full_pool_can_promote_item_below_a_control_top50(self):
        ids = np.arange(51, dtype=np.int32)
        old = np.arange(51, 0, -1, dtype=np.float64)
        new = np.zeros(51, dtype=np.float64)
        new[-1] = 100.0
        self.assertNotIn(50, rank_all(ids, old)[:50])
        self.assertIn(50, rank_all(ids, blended_scores(old, new, .5))[:50])


if __name__ == "__main__":
    unittest.main()
