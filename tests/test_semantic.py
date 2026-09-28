"""Contracts for frozen E5 text preparation and exact scoped dense retrieval."""
# Dense retrieval tests keep the same corpus-scope contract as lexical retrieval.
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from avito_test import semantic_search
from avito_test.semantic_benchmark import item_text, query_text
from avito_test.semantic_extended_oof import sample_expanded
from avito_test.semantic_finalize_embeddings import finalize


class TextPreparation(unittest.TestCase):
    def test_natural_text_and_prefixes(self):
        self.assertEqual(query_text("  <b>Автоподбор</b>  "), "query: Автоподбор")
        self.assertEqual(item_text("Монтаж  <br>домофона", "Услуга: установка", "Описания &amp; фото"),
                         "passage: Монтаж домофона\nУслуга: установка\nОписания & фото")


class ScopedDenseSearch(unittest.TestCase):
    def setUp(self):
        self.old_depth = semantic_search.DEPTH
        semantic_search.DEPTH = 1
        index = semantic_search.SemanticIndex.__new__(semantic_search.SemanticIndex)
        index.item_vectors = np.zeros((4, 384), dtype=np.float32)
        index.item_vectors[:, 0] = np.array([.6, .5, 1., .9], dtype=np.float32)
        index.benchmark_mask = np.array([True, True, False, False])
        index.benchmark_rows = np.array([0, 1], dtype=np.int32)
        index.locations = np.array([1, 2, 1, 2])
        query_vector = np.zeros(384, dtype=np.float32)
        query_vector[0] = 1.
        index.queries = {"услуга": query_vector}
        index.indices = {}
        index.local_indices = {}
        index.search_seconds = 0.
        self.index = index

    def tearDown(self):
        semantic_search.DEPTH = self.old_depth

    def test_augmented_scope_inserts_only_current_positives_before_topk(self):
        result = self.index.retrieve("услуга", 1, np.array([2], dtype=np.int32), augmented=True)
        np.testing.assert_array_equal(result["semantic_global"], [2])
        np.testing.assert_array_equal(result["semantic_local"], [2])
        self.assertNotIn(3, result["semantic_global"])
        result = self.index.retrieve("услуга", 1, np.array([3], dtype=np.int32), augmented=True)
        np.testing.assert_array_equal(result["semantic_global"], [3])
        np.testing.assert_array_equal(result["semantic_local"], [0])
        self.assertNotIn(2, result["semantic_global"])
        self.assertNotIn(2, result["semantic_local"])

    def test_union_search_and_missing_location(self):
        result = self.index.retrieve("услуга", 2)
        np.testing.assert_array_equal(result["semantic_global"], [2])
        np.testing.assert_array_equal(result["semantic_local"], [3])
        result = self.index.retrieve("услуга", 3)
        self.assertEqual(len(result["semantic_local"]), 0)
        result = self.index.retrieve("услуга", None, np.array([2], dtype=np.int32), augmented=True)
        self.assertEqual(len(result["semantic_local"]), 0)

    def test_similarity_rank_uses_full_reference_pool(self):
        selected = np.array([0, 2], dtype=np.int32)
        reference = np.arange(4, dtype=np.int32)
        results = {"semantic_global": np.array([2, 1], dtype=np.int32),
                   "semantic_local": np.array([0], dtype=np.int32)}
        query = {"search_query": "услуга"}
        features = semantic_search.semantic_features(self.index, query, selected,
                                                     results, True, reference)
        np.testing.assert_allclose(features[:, :3], [[.6, 3., .4], [1., 1., 0.]], atol=1e-6)
        np.testing.assert_array_equal(features[:, 3:], [[0., 1., 1., 0.],
                                                       [1., 0., 0., 1.]])


class ExpandedOOF(unittest.TestCase):
    def test_semantic_only_positive_is_kept_without_inventing_other_positives(self):
        results = {name: np.array([1, 2, 3], dtype=np.int32) for name in
                   ("bm25_geo16", "char_geo16", "bm25_stats", "char_stats", "history")}
        results["semantic_global"] = np.array([4, 5], dtype=np.int32)
        results["semantic_local"] = np.array([5], dtype=np.int32)
        chosen, found = sample_expanded(results, {5, 99}, np.random.default_rng(7))
        self.assertEqual(found, 1)
        self.assertIn(5, chosen)
        self.assertNotIn(99, chosen)
        self.assertEqual(len(chosen), len(np.unique(chosen)))


class VectorCoverage(unittest.TestCase):
    def test_finalization_checks_every_row_and_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            vectors = np.zeros((6, 384), dtype=np.float32)
            vectors[:, 0] = 1.
            np.save(out / "unique_vectors.npy", vectors)
            np.save(out / "item_to_text.npy", np.arange(6, dtype=np.int32))
            (out / "items_manifest.json").write_text(json.dumps({"unique_texts": 6,
                "item_count": 6}))
            (out / "encode_state.json").write_text(json.dumps({"next_row": 2,
                "complete": False, "signature": {"weights_sha256": "model",
                    "item_text_sha256": "texts"}, "elapsed_seconds": 1.}))
            for start, end in ((2, 4), (4, 6)):
                (out / f"segment_{start}_{end}.json").write_text(json.dumps({
                    "next_row": end, "complete": True, "elapsed_seconds": 1.,
                    "signature": {"start": start, "end": end,
                                  "model_sha256": "model", "texts_sha256": "texts"}}))
            with redirect_stdout(StringIO()):
                report = finalize(out, [(2, 4), (4, 6)])
            self.assertTrue(report["complete"])
            self.assertEqual(report["all_vectors_norm_max_error"], 0.)
            with redirect_stdout(StringIO()):
                self.assertTrue(finalize(out, [(2, 4), (4, 6)])["complete"])
            vectors[5] = 0.
            np.save(out / "unique_vectors.npy", vectors)
            with self.assertRaisesRegex(ValueError, "unnormalized"):
                finalize(out, [(2, 4), (4, 6)])


if __name__ == "__main__":
    unittest.main()
