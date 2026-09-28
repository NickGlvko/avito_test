"""Checks for complete-list fusion and deterministic extremes."""
# Extreme fusion weights must reproduce the original complete lists.
import unittest

import numpy as np

from avito_test.semantic_final import blend_d, rrf_bd, select
from avito_test.semantic_blend import rank_all


class FinalFusionTests(unittest.TestCase):
    def test_rrf_extremes_reproduce_complete_model_lists(self):
        b_ids = np.arange(60, dtype=np.int32)
        d_ids = np.arange(80, dtype=np.int32)
        b_scores = np.arange(60, dtype=np.float64)
        d_scores = np.arange(80, dtype=np.float64)
        np.testing.assert_array_equal(select(d_ids, rrf_bd(
            b_ids, b_scores, d_ids, d_scores, 0.0)),
            rank_all(b_ids, b_scores)[:50])
        np.testing.assert_array_equal(select(d_ids, rrf_bd(
            b_ids, b_scores, d_ids, d_scores, 1.0)),
            rank_all(d_ids, d_scores)[:50])

    def test_rrf_uses_both_complete_lists_beyond_top_fifty(self):
        ids = np.arange(100, dtype=np.int32)
        b_scores = np.arange(100, dtype=np.float64)
        d_scores = np.arange(100, dtype=np.float64)
        d_scores[49] = 1000.0
        fused = rrf_bd(ids, b_scores, ids, d_scores, 0.5)
        self.assertEqual(len(select(ids, fused)), 50)
        self.assertTrue(np.all(np.isfinite(fused)))
        self.assertEqual(int(np.argmax(fused)), 99)
        # Item 9 is below both top-50 lists, but both complete ranks count.
        self.assertAlmostEqual(fused[9], 0.5 / (60 + 91) + 0.5 / (60 + 91))

    def test_zero_variance_d_blend_and_item_id_ties(self):
        ids = np.array([1, 2, 3], dtype=np.int32)
        scores = blend_d(np.ones(3), np.ones(3), 0.5)
        np.testing.assert_array_equal(scores, np.zeros(3))
        np.testing.assert_array_equal(select(ids, scores), ids)

    def test_missing_b_contribution_is_zero(self):
        b_ids = np.array([1, 3, 5], dtype=np.int32)
        d_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        scores = rrf_bd(b_ids, np.array([3., 2., 1.]),
                        d_ids, np.array([5., 4., 3., 2., 1.]), 0.5)
        self.assertAlmostEqual(scores[1], 0.5 / (60 + 2))
        self.assertAlmostEqual(scores[3], 0.5 / (60 + 4))


if __name__ == "__main__":
    unittest.main()
