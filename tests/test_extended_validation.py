"""Check corpus membership before top-k, history isolation, and complete denominators."""
# Unavailable corpus rows must be removed before source top-k selection.
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
from scipy import sparse

from avito_test.extended_validation import (
    ScopedRetriever, augmented_scope, measure, rank_in_scope, union_items,
)
from avito_test.retrieval import TextIndex, TrainSignals, channels, topk


class ScopeContracts(unittest.TestCase):
    def test_filter_before_topk_not_after(self):
        scores = np.array([1., 2., 100., 200.])
        allowed = np.array([0, 1], dtype=np.int32)
        np.testing.assert_array_equal(rank_in_scope(scores, allowed, 2), [1, 0])
        np.testing.assert_array_equal(rank_in_scope(np.zeros(4), allowed, 1), [0])

    def test_each_query_gets_only_its_missing_positives(self):
        benchmark = np.array([1, 3, 5], dtype=np.int32)
        np.testing.assert_array_equal(augmented_scope(benchmark, np.array([0, 3])), [0, 1, 3, 5])
        np.testing.assert_array_equal(augmented_scope(benchmark, np.array([2, 5])), [1, 2, 3, 5])
        np.testing.assert_array_equal(benchmark, [1, 3, 5])

    def test_sparse_optimization_preserves_ties_and_zero_filling(self):
        rng = np.random.default_rng(1)
        scores = rng.choice([0., 0., 0., 1., 2., 3.], size=2000)
        for allowed in [None, np.arange(0, 2000, 3), np.array([1, 5, 8])]:
            values = scores if allowed is None else scores[allowed]
            for positive_only in [False, True]:
                expected = topk(values, 50, positive_only)
                if allowed is not None:
                    expected = allowed[expected]
                np.testing.assert_array_equal(rank_in_scope(scores, allowed, 50, positive_only), expected)
        np.testing.assert_array_equal(augmented_scope(np.array([1, 8]), np.array([0, 2, 3, 3, 9])), [0, 1, 2, 3, 8, 9])

    def test_complete_denominator_and_loss_accounting(self):
        relevant = np.array([0, 1, 2])
        scores = measure(np.array([0, 4]), {"a": np.array([0, 1, 4])}, {"a": 1.},
                         relevant, np.array([True, False, False, True, True]))
        self.assertAlmostEqual(scores["recall@50"], 1 / 3)
        self.assertAlmostEqual(scores["candidate_recall"], 2 / 3)
        self.assertEqual(scores["lost_at_selection"], 1)
        self.assertEqual(scores["missed_by_sources"], 1)
        self.assertEqual(scores["outside_positives"], 2)
        self.assertEqual(scores["outside_recall@50"], 0)
        self.assertEqual(scores["outside_candidate_recall"], .5)
        self.assertEqual(scores["retrieved"] + scores["lost_at_selection"] + scores["missed_by_sources"], 3)

    def test_ceiling_for_more_than_fifty_positives(self):
        relevant = np.arange(80)
        result = measure(np.arange(80), {"a": np.arange(80)}, {"a": 1.}, relevant, np.ones(80, dtype=bool))
        self.assertEqual(result["lost_at_selection"], 30)
        self.assertEqual(result["unavoidable_candidate_cap_loss"], 30)
        self.assertEqual(result["avoidable_selection_loss"], 0)
        self.assertEqual(result["candidate_oracle@50"], 50 / 80)

    def test_history_scope_before_nearest_text_selection(self):
        # Twenty stronger unavailable history texts must not suppress available text 0.
        links = sparse.eye(22, format="csr", dtype=np.float32)
        signals = SimpleNamespace(categories=np.ones(22), category_prob={}, popularity=np.zeros(22),
                                  links=links, history_vectorizer=object())
        benchmark_mask = np.zeros(22, dtype=bool)
        benchmark_mask[0] = True
        retriever = ScopedRetriever(None, signals, benchmark_mask)
        similarities = np.array([.31, .32, *([.9] * 20)])
        score = retriever.history_scores(similarities, np.array([1]), augmented=True)
        self.assertGreater(score[0], 0)
        self.assertGreater(score[1], 0)
        self.assertEqual(np.count_nonzero(score[2:]), 0)
        # Another query does not inherit the previous insertion.
        other = retriever.history_scores(similarities, np.array([2]), augmented=True)
        self.assertEqual(other[1], 0)
        self.assertGreater(other[2], 0)

    def test_union_keeps_competitors_and_benchmark_snapshot(self):
        first, second, third = [f"{i:016x}" for i in [1, 2, 3]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pl.DataFrame({"item_id": [first, second], "item_title_raw": ["same", "same"]}).write_parquet(root / "benchmark_items.parquet")
            pl.DataFrame({"item_id": [first, third, third], "item_title_raw": ["old", "same", "same"]}).write_parquet(root / "train.parquet")
            items, report = union_items(root)
            self.assertEqual(items["item_id"].to_list(), [first, second, third])
            self.assertEqual(items["item_title_raw"].to_list(), ["same", "same", "same"])
            self.assertEqual(report["conflicting_shared_snapshots"], 1)

    def test_unrestricted_scores_match_original_channels(self):
        titles = ["ремонт машин", "ремонт машин машин", "ремонт шин", "ремонт шин шин"]
        items = pl.DataFrame({"item_id": [f"{i:016x}" for i in range(4)],
                              "item_title_raw": titles, "item_infm_params_text": titles,
                              "item_description_raw": titles, "item_location_id": [1, 2, 1, 2],
                              "item_category_id": [114] * 4})
        training = pl.DataFrame({"text": ["ремонт машин", "ремонт шин"],
                                 "item_id": ["0000000000000001", "0000000000000002"],
                                 "search_category": [114, 114], "item_category_id": [114, 114]})
        index, signals = TextIndex(items), TrainSignals(training, items)
        query = {"search_query": "ремонт машин", "search_location_id": 1, "search_category": 114}
        actual = ScopedRetriever(index, signals, np.array([True, False, True, False])).retrieve(query, np.array([1]))
        original = channels(index, signals, query)
        for name, ranking in actual["union"].items():
            np.testing.assert_array_equal(ranking, original[name])
        for ranking in actual["benchmark_plus_positives"].values():
            self.assertTrue(set(ranking) <= {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
