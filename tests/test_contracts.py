"""Tests target leakage, macro averaging, deterministic retrieval and submission failures."""
# Small fixtures check IDs, deduplication, deterministic splits and macro recall.
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from avito_test.data import enrich_train, evaluation_groups, normalize, split_train
from avito_test.evaluation import recall
from avito_test.retrieval import BM25Field, TrainSignals, fuse, topk
from avito_test.submission import validate_answer, write_answer


class RetrievalContracts(unittest.TestCase):
    def test_normalization_keeps_negations_and_numbers(self):
        self.assertEqual(normalize("<p>БЕЗ водителя — 24 часа, ёлка</p>"), "без водителя 24 часа елка")

    def test_topk_ties(self):
        np.testing.assert_array_equal(topk(np.array([1, 3, 2, 2, 0]), 3), [1, 2, 3])
        np.testing.assert_array_equal(topk(np.zeros(4), 2), [0, 1])
        self.assertEqual(len(topk(np.zeros(4), 2, positive_only=True)), 0)

    def test_macro_not_micro_recall(self):
        scores = [recall([1], [1], 50), recall([2], [2, 3, 4], 50)]
        self.assertAlmostEqual(sum(scores) / 2, 2 / 3)
        self.assertEqual(recall([1, 1, 1], [1, 2], 50), .5)

    def test_fusion_unique(self):
        actual = fuse({"a": np.array([2, 1]), "b": np.array([1, 3])}, {"a": 1, "b": 1})
        self.assertEqual(actual.tolist(), [1, 2, 3])

    def test_bm25_against_formula(self):
        model = BM25Field(["баня баня", "баня", "сауна сауна", "сауна"])
        score = model.score("баня")
        idf = np.log(1 + (4 - 2 + .5) / (2 + .5))
        expected = idf * 2 * 2.2 / (2 + 1.2 * (.25 + .75 * 2 / 1.5))
        self.assertAlmostEqual(float(score[0]), float(expected), places=5)
        self.assertEqual(float(score[2]), 0.)
        np.testing.assert_array_equal(score, model.score("баня баня"))

    def test_group_and_text_split_no_leakage(self):
        rows = [{"search_query": f"Запрос {i}", "search_location_id": loc,
                 "search_is_delivery_search": 0, "search_infm_params_text": "",
                 "search_category": 1, "item_id": str(i)} for i in range(200) for loc in [1, 2]]
        data = enrich_train(pl.DataFrame(rows + rows[:20]))
        training = split_train(data, "group")
        self.assertFalse(set(training["group_id"]) & set(data.filter(pl.col("split") >= 8)["group_id"]))
        text_training = split_train(data, "unseen")
        self.assertFalse(set(text_training["text"]) & set(data.filter(pl.col("text_split") >= 8)["text"]))
        shuffled = enrich_train(pl.DataFrame(rows[::-1]))
        self.assertEqual(data.select("group_id", "split").unique().sort("group_id").to_dicts(),
                         shuffled.select("group_id", "split").unique().sort("group_id").to_dicts())

    def test_missing_positive_excludes_whole_group(self):
        data = pl.DataFrame({"group_id": ["a", "a", "b"], "item_id": ["1", "2", "1"],
                             "split": [8, 8, 8], "search_query": ["a", "a", "b"],
                             "search_location_id": [1, 1, 1], "search_is_delivery_search": [0, 0, 0],
                             "search_infm_params_text": ["", "", ""], "search_category": [1, 1, 1],
                             "text": ["a", "a", "b"]})
        groups, report = evaluation_groups(data, ["1"], "group", "dev")
        self.assertEqual(groups["group_id"].to_list(), ["b"])
        self.assertEqual(report["excluded_missing_positives"], 1)

    def test_history_uses_only_passed_training_rows(self):
        items = pl.DataFrame({"item_id": ["a", "b"], "item_category_id": [114, 114],
                              "item_location_id": [1, 1]})
        training = pl.DataFrame({"text": ["ремонт машин"], "item_id": ["a"],
                                 "search_category": [114], "item_category_id": [114]})
        signals = TrainSignals(training, items)
        scores = signals.history({"search_query": "ремонт машин"})
        self.assertGreater(scores[0], 0)
        self.assertEqual(scores[1], 0)
        self.assertEqual(signals.popularity[1], 0)
        self.assertEqual(signals.seen_items, {"a"})


class SubmissionContracts(unittest.TestCase):
    def test_roundtrip_and_invalid_files(self):
        qids, ids = ["0Abcdefghijklmno"], ["0000000000000001", "0000000000000002"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answer.csv"
            self.assertEqual(write_answer(path, qids, [ids], ids)["queries"], 1)
            good = path.read_bytes()
            write_answer(path, qids, [ids], ids)
            self.assertEqual(good, path.read_bytes())
            for value in [ids[0] + " " + ids[0], "000000000000000A", " " + ids[0], ids[0] + "  " + ids[1],
                          "0000000000000003", str(ids)]:
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["query_id", "answer"])
                    writer.writerow([qids[0], value])
                with self.assertRaises(ValueError):
                    validate_answer(path, qids, ids)

    def test_missing_extra_duplicate_queries_and_limit(self):
        qids = ["0Abcdefghijklmno"]
        ids = [f"{i:016x}" for i in range(51)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answer.csv"
            invalid = [[], [(qids[0], ""), (qids[0], "")], [("1Abcdefghijklmno", "")],
                       [(qids[0], " ".join(ids))]]
            for rows in invalid:
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["query_id", "answer"])
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    validate_answer(path, qids, ids)


if __name__ == "__main__":
    unittest.main()
