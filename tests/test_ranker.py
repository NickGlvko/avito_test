"""Contracts for OOF ranking, candidate pools and loss accounting."""
# OOF statistics must exclude the held-out query group's interactions.
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from avito_test.ranker import (
    FEATURES, POOL_SOURCES, FeatureRetriever, candidate_metrics, canonical_fold,
    model_ranking, pool_ids, sample_training_rows, training_folds,
)
from avito_test.retrieval import TextIndex, TrainSignals
from avito_test.ranker_selection import select_robust_dev


class RankerContracts(unittest.TestCase):
    def test_canonical_query_variants_share_fold(self):
        base = {"search_query": "Ёлка  ремонт", "search_location_id": 1,
                "search_is_delivery_search": 0, "search_infm_params_text": "Без  очереди",
                "search_category": 3}
        variant = {**base, "search_query": "елка ремонт",
                   "search_infm_params_text": "без очереди"}
        self.assertEqual(canonical_fold(base), canonical_fold(variant))
        data = pl.DataFrame([{**base, "group_id": "a"}, {**variant, "group_id": "b"}])
        folds = training_folds(data)
        self.assertEqual(folds["fold"].to_list()[0], folds["fold"].to_list()[1])

    def test_pool_and_negative_sample_preserve_all_found_positives(self):
        sources = {name: np.array([i, i + 10, i + 20], dtype=np.int32)
                   for i, name in enumerate(POOL_SOURCES)}
        text = set(map(int, pool_ids(sources, include_history=False)))
        full = set(map(int, pool_ids(sources)))
        self.assertEqual(full - text, set(map(int, sources["history"])) - text)
        self.assertTrue(set(map(int, sources["history"])) & full)
        truth = {0, 14, 999}
        sample, positives = sample_training_rows(sources, truth, 4, np.random.default_rng(1))
        self.assertEqual(positives, 2)
        self.assertEqual(set(map(int, sample)) & truth, {0, 14})
        self.assertLessEqual(len(sample), positives + 4)

    def test_missing_source_is_separate_from_nonzero_raw_score(self):
        self.assertIn("geo_bm25_missing", FEATURES)
        self.assertIn("bm25_raw", FEATURES)
        self.assertNotIn("in_benchmark", FEATURES)
        self.assertNotIn("added_positive", FEATURES)

    def test_raw_text_scores_precede_geographic_multiplier(self):
        titles = ["ремонт авто", "ремонт авто срочно", "ремонт швейных машин", "ремонт техники"]
        items = pl.DataFrame({"item_id": [f"{i:016x}" for i in range(4)],
                              "item_title_raw": titles, "item_infm_params_text": titles,
                              "item_description_raw": titles, "item_location_id": [1, 2, 1, 2],
                              "item_category_id": [3] * 4, "item_rating": [5.] * 4,
                              "item_rating_reviews_count": [1] * 4, "item_price": [100.] * 4})
        interactions = pl.DataFrame({"text": ["ремонт авто", "ремонт техники"],
                                     "item_id": [items["item_id"][0], items["item_id"][3]],
                                     "search_category": [3, 3], "item_category_id": [3, 3]})
        index, signals = TextIndex(items), TrainSignals(interactions, items)
        retriever = FeatureRetriever(index, signals, np.array([True] * 4), items)
        query = {"search_query": "ремонт авто", "search_location_id": 1,
                 "search_category": 3, "search_is_delivery_search": 0,
                 "search_infm_params_text": ""}
        results, raw, allowed = retriever.retrieve_with_scores(query, np.empty(0, dtype=np.int32), False)
        self.assertIsNone(allowed)
        self.assertGreater(float(raw["bm25"][1]), 0)
        self.assertAlmostEqual(float(raw["bm25_geo16"][0]), 16 * float(raw["bm25"][0]), places=5)
        self.assertAlmostEqual(float(raw["bm25_geo16"][1]), float(raw["bm25"][1]), places=5)
        features = retriever.features(query, np.array([1], dtype=np.int32), results, raw, allowed)
        self.assertAlmostEqual(float(features[0, FEATURES.index("bm25_raw")]), float(raw["bm25"][1]))

    def test_macro_denominator_and_selection_loss(self):
        signals = SimpleNamespace(popularity=np.zeros(6), locations=np.array([1, 2, 1, 1, 2, 1]))
        metrics = candidate_metrics(np.array([0]), np.array([0, 1, 2]), {0, 1, 4},
                                    signals, {"search_location_id": 1}, np.array([1, 2]), {1})
        self.assertAlmostEqual(metrics["recall@50"], 1 / 3)
        self.assertAlmostEqual(metrics["candidate_recall"], 2 / 3)
        self.assertEqual(metrics["lost_at_selection"], 1)
        self.assertEqual(metrics["missed_by_sources"], 1)
        self.assertEqual(metrics["history_only_positives"], 1)
        self.assertEqual(metrics["history_only_found"], 0)

    def test_model_selection_deterministic_and_limited_to_eligible_pool(self):
        ids = np.array([5, 2, 4, 1], dtype=np.int32)
        scores = np.array([1., 3., 3., 4.])
        np.testing.assert_array_equal(model_ranking(ids, scores), [1, 2, 4, 5])
        np.testing.assert_array_equal(model_ranking(ids, scores, np.array([1, 1, 0, 0], dtype=bool)), [2, 5])

    def test_transfer_rule_reads_dev_only(self):
        values = {"catboost_text_pool": (.70, .79),
                  "catboost_full_no_history_features": (.72, .80),
                  "catboost_full_with_history_features": (.73, .78)}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "dev").mkdir()
            (run / "test").mkdir()
            rows = [{"group_id": "a", "variant": name, "scheme": scheme, "recall@50": value}
                    for name, pair in values.items() for scheme, value in
                    zip(["union", "benchmark_plus_positives"], pair)]
            pl.DataFrame(rows).write_parquet(run / "dev" / "per_query.parquet")
            # An opposite test result must have no effect on the dev decision.
            pl.DataFrame([{**row, "recall@50": 1 - row["recall@50"]} for row in rows]).write_parquet(
                run / "test" / "per_query.parquet")
            decision = select_robust_dev(run)
            self.assertEqual(decision["union_only_variant"], "catboost_full_with_history_features")
            self.assertEqual(decision["selected_variant"], "catboost_full_no_history_features")
            self.assertEqual(decision["dev_groups"], 1)


if __name__ == "__main__":
    unittest.main()
