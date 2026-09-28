"""Validate the blended CSV and every frozen local dependency."""
# Verify the exact CSV bytes and frozen local model dependencies.
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from .data import sha256_file, source_manifest
from .semantic_blend import ALPHAS, variant
from .semantic_blend_control import ANSWER as CONTROL_ANSWER, CONTROL, OUT
from .semantic_blend_predict import ANSWER, DATA, PACKAGE, SIGNALS
from .semantic_embeddings import MODEL, OUT as VECTORS
from .semantic_scale_evaluate import INDEX
from .submission import validate_answer


def verify() -> dict:
    config = json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    if (config["dev_decision"] != decision or
            decision["selected"] != variant(config["alpha"]) or
            config["alpha"] not in ALPHAS or
            config["control_answer_sha256"] != decision["control_answer_sha256"] or
            any(config[f"{name}_model_sha256"] != decision[f"{name}_model_sha256"]
                for name in ("B3000", "B15000")) or
            sha256_file(Path(decision["dev_per_query"])) != decision["dev_sha256"]):
        raise ValueError("Blend differs from frozen full-dev decision")
    expected = (
        (ANSWER, config["answer_sha256"]),
        (PACKAGE / "B3000.cbm", config["B3000_model_sha256"]),
        (PACKAGE / "B15000.cbm", config["B15000_model_sha256"]),
        (PACKAGE / "item_id_to_vector_row.parquet", config["item_id_mapping_sha256"]),
        (CONTROL / CONTROL_ANSWER.name, config["control_answer_sha256"]),
        (MODEL / "onnx/model.onnx", config["e5_weights_sha256"]),
        (MODEL / "tokenizer.json", config["e5_tokenizer_sha256"]),
        (VECTORS / "item_vectors.npy", config["item_vectors_sha256"]),
        (VECTORS / "unique_vectors.npy", config["unique_vectors_sha256"]),
        (VECTORS / "item_to_text.npy", config["item_to_text_sha256"]),
        (SIGNALS, config["all_train_signals_sha256"]),
        (Path("uv.lock"), config["uv_lock_sha256"]),
    )
    for path, hash_value in expected:
        if sha256_file(path) != hash_value:
            raise ValueError(f"Frozen SHA256 differs: {path}")
    if sha256_file(Path(__file__).with_name("semantic_blend_predict.py")) != config["code_sha256"]:
        raise ValueError("Frozen benchmark prediction code changed")
    if source_manifest(DATA) != config["sources"]:
        raise ValueError("Input data changed")
    queries = pl.read_parquet(DATA / "benchmark_queries.parquet",
                              columns=["query_id"])["query_id"].to_list()
    items = pl.read_parquet(DATA / "benchmark_items.parquet",
                            columns=["item_id"])["item_id"].to_list()
    validation = validate_answer(ANSWER, queries, items)
    if validation != config["validation"] or validation != {
            "queries": 2452, "min_candidates": 50, "max_candidates": 50}:
        raise ValueError("Blended CSV has wrong IDs or candidate count")
    mapping = pl.read_parquet(PACKAGE / "item_id_to_vector_row.parquet")
    index = joblib.load(INDEX, mmap_mode="r")
    if (mapping.columns != ["item_row", "item_id"] or
            not np.array_equal(mapping["item_row"].to_numpy(),
                               np.arange(mapping.height, dtype=np.int32)) or
            mapping["item_id"].to_list() != index.item_ids or
            np.load(VECTORS / "item_vectors.npy", mmap_mode="r").shape !=
            (mapping.height, 384)):
        raise ValueError("Item/vector row mapping changed")
    result = {"answer_sha256": config["answer_sha256"],
              "alpha": config["alpha"], **validation}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
