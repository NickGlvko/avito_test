"""Validate the offline tuned answer against frozen weights, data and IDs."""
# Require the same CSV structure and hashes as the selected capacity run.
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from .data import sha256_file, source_manifest
from .semantic_embeddings import MODEL, OUT as VECTORS
from .semantic_scale_evaluate import INDEX
from .semantic_tune_cache import OUT
from .semantic_tune_control import ANSWER as CONTROL_ANSWER, EXPECTED_SHA256
from .semantic_tune_predict import ANSWER, DATA, PACKAGE, SIGNALS
from .submission import validate_answer


def verify() -> dict:
    config = json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    if (config["dev_decision"] != decision or
            config["selected"] != decision["selected"] or
            config["checkpoint"] != decision["selected_checkpoint"] or
            sha256_file(OUT / "final/dev/per_query.parquet") != decision["dev_sha256"] or
            sha256_file(CONTROL_ANSWER) != EXPECTED_SHA256 or
            config["submitted_answer_sha256"] != EXPECTED_SHA256):
        raise ValueError("Tuned answer differs from dev selection or submitted control")
    expected = [(ANSWER, config["answer_sha256"]),
                (PACKAGE / "item_id_to_vector_row.parquet",
                 config["item_id_mapping_sha256"]),
                (MODEL / "onnx/model.onnx", config["e5_weights_sha256"]),
                (MODEL / "tokenizer.json", config["e5_tokenizer_sha256"]),
                (VECTORS / "item_vectors.npy", config["item_vectors_sha256"]),
                (VECTORS / "unique_vectors.npy", config["unique_vectors_sha256"]),
                (VECTORS / "item_to_text.npy", config["item_to_text_sha256"]),
                (SIGNALS, config["all_train_signals_sha256"]),
                (Path("uv.lock"), config["uv_lock_sha256"])]
    expected.extend((PACKAGE / f"{name}.cbm", digest)
                    for name, digest in config["models"].items())
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen dependency changed: {path}")
    if (config["models"]["new_ranker"] != decision["checkpoint_model_sha256"] or
            sha256_file(Path(__file__).with_name("semantic_tune_predict.py")) !=
            config["code_sha256"] or source_manifest(DATA) != config["sources"]):
        raise ValueError("Tuned model, code or input data changed")
    queries = pl.read_parquet(DATA / "benchmark_queries.parquet",
                              columns=["query_id"])["query_id"].to_list()
    items = pl.read_parquet(DATA / "benchmark_items.parquet",
                            columns=["item_id"])["item_id"].to_list()
    validation = validate_answer(ANSWER, queries, items)
    if (validation != config["validation"] or
            validation != {"queries": 2452, "min_candidates": 50,
                           "max_candidates": 50}):
        raise ValueError("Tuned CSV has invalid IDs or candidate count")
    mapping = pl.read_parquet(PACKAGE / "item_id_to_vector_row.parquet")
    index = joblib.load(INDEX, mmap_mode="r")
    if (mapping.columns != ["item_row", "item_id"] or
            not np.array_equal(mapping["item_row"].to_numpy(),
                               np.arange(mapping.height, dtype=np.int32)) or
            mapping["item_id"].to_list() != index.item_ids or
            np.load(VECTORS / "item_vectors.npy", mmap_mode="r").shape !=
            (mapping.height, 384)):
        raise ValueError("Item ID to E5 vector row mapping changed")
    result = {"answer_sha256": config["answer_sha256"], **validation,
              "selected": decision["selected"]}
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    verify()
