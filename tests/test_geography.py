"""Contracts for off-location quota selection."""
# Quota rules must keep 50 distinct IDs even with missing location values.
import unittest
import tempfile
import json
from pathlib import Path

import numpy as np

from avito_test.geography import location_kind, quota_ranking
from avito_test.geography_expanded import sample_extended
from avito_test.geography_report import freeze_dev_decision


class GeographyContracts(unittest.TestCase):
    def test_quota_promotes_best_other_and_replaces_weakest_same(self):
        order = np.arange(60, dtype=np.int32)
        locations = np.ones(60, dtype=np.float64)
        locations[[0, 5, 50, 51, 52, 53]] = 2
        locations[49] = np.nan
        selected = quota_ranking(order, locations, 1, 5)
        self.assertEqual(len(selected), 50)
        self.assertEqual(len(set(selected.tolist())), 50)
        self.assertTrue({0, 5, 50, 51, 52}.issubset(set(selected.tolist())))
        self.assertIn(49, selected)  # Unknown is neither off-location nor replaceable local.
        self.assertTrue({47, 48}.isdisjoint(set(selected.tolist())))
        np.testing.assert_array_equal(quota_ranking(order, locations, 1, 2), order[:50])

    def test_fewer_off_location_candidates_uses_ordinary_fill(self):
        order = np.arange(52, dtype=np.int32)
        locations = np.ones(52, dtype=np.float64)
        locations[50] = 2
        selected = quota_ranking(order, locations, 1, 10)
        self.assertEqual(len(selected), 50)
        self.assertIn(50, selected)
        self.assertNotIn(49, selected)
        self.assertEqual(sum(locations[selected] == 2), 1)

    def test_missing_location_is_unknown(self):
        self.assertEqual(location_kind(np.nan, 1), "unknown")
        self.assertEqual(location_kind(2, None), "unknown")
        self.assertEqual(location_kind(1, 1), "same")
        self.assertEqual(location_kind(2, 1), "other")

    def test_extended_sampling_keeps_new_positive_and_new_competitors(self):
        names = ("bm25_geo16", "char_geo16", "bm25_stats", "char_stats", "history")
        results = {name: np.array([0, 1], dtype=np.int32) for name in names}
        pool = np.arange(6, dtype=np.int32)
        selected, found = sample_extended(results, pool, np.array([2, 3]),
                                          np.array([4, 5]), {3, 5},
                                          np.random.default_rng(1), limit=3)
        self.assertEqual(found, 2)
        self.assertTrue({3, 5}.issubset(set(selected.tolist())))
        self.assertTrue(set(selected.tolist()).issubset(set(pool.tolist())))

    def test_dev_rule_rejects_geographic_gain_with_overall_loss(self):
        summary = {"stage": "dev", "queries": 1, "expected_queries": 1, "schemes": {
            "union": {"variants": {
                "catboost_submitted": {"recall50": .72},
                "reserve_2": {"recall50": .73, "vs_submitted": {"ci95": [.005, .015]}},
            }},
            "benchmark_plus_positives": {"variants": {
                "catboost_submitted": {"recall50": .79},
                "reserve_2": {"recall50": .785},
            }},
        }}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dev_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            decision = freeze_dev_decision(summary, root / "dev_decision.json")
            self.assertEqual(decision["selected"], "catboost_submitted")
            summary["expected_queries"] = 2
            with self.assertRaisesRegex(ValueError, "partial dev"):
                freeze_dev_decision(summary, root / "dev_decision.json")


if __name__ == "__main__":
    unittest.main()
