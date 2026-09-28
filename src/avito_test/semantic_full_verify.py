"""Check an improved full-pool answer and its frozen local dependencies."""
# Check the answer and its model, vector and data fingerprints together.
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from .data import sha256_file, source_manifest
from .semantic_embeddings import MODEL, OUT as VECTORS
from .semantic_full_evaluate import DECISION
from .semantic_full_oof import OUT
from .semantic_full_predict import (ANSWER, DATA, PACKAGE, SIGNALS, SUBMITTED,
                                    SUBMITTED_SHA256)
from .semantic_scale_evaluate import INDEX
from .submission import validate_answer


def verify() -> dict:
    config = json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    selected = decision["selected"]
    if (config["dev_decision"] != decision or
            config["selected"] != selected or
            sha256_file(OUT / "evaluation/dev/per_query.parquet") != decision["dev_sha256"] or
            sha256_file(SUBMITTED) != SUBMITTED_SHA256 or
            config["submitted_answer_sha256"] != SUBMITTED_SHA256):
        raise ValueError("Answer differs from frozen dev selection or submitted control")
    expected = [
        (ANSWER, config["answer_sha256"]),
        (PACKAGE / "ranker.cbm", config["model_sha256"]),
        (PACKAGE / "item_id_to_vector_row.parquet", config["item_id_mapping_sha256"]),
        (MODEL / "onnx/model.onnx", config["e5_weights_sha256"]),
        (MODEL / "tokenizer.json", config["e5_tokenizer_sha256"]),
        (VECTORS / "item_vectors.npy", config["item_vectors_sha256"]),
        (VECTORS / "unique_vectors.npy", config["unique_vectors_sha256"]),
        (VECTORS / "item_to_text.npy", config["item_to_text_sha256"]),
        (SIGNALS, config["all_train_signals_sha256"]),
        (Path("uv.lock"), config["uv_lock_sha256"]),
    ]
    for path, value in expected:
        if sha256_file(path) != value:
            raise ValueError(f"Frozen dependency changed: {path}")
    if (config["model_sha256"] != decision["models"][selected] or
            sha256_file(Path(__file__).with_name("semantic_full_predict.py")) !=
            config["code_sha256"] or source_manifest(DATA) != config["sources"]):
        raise ValueError("Model, prediction code or input data changed")
    queries = pl.read_parquet(DATA / "benchmark_queries.parquet",
                              columns=["query_id"])["query_id"].to_list()
    items = pl.read_parquet(DATA / "benchmark_items.parquet",
                            columns=["item_id"])["item_id"].to_list()
    validation = validate_answer(ANSWER, queries, items)
    if (validation != config["validation"] or
            validation != {"queries": 2452, "min_candidates": 50,
                           "max_candidates": 50}):
        raise ValueError("Full-pool CSV has invalid IDs or candidate count")
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
              "selected": selected}
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    verify()
