"""Verify the selected 15k E5 answer and its frozen local dependencies."""
# Validate the frozen 15k answer against local model and data hashes.
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from .data import sha256_file, source_manifest
from .semantic_embeddings import MODEL, OUT as VECTORS
from .semantic_scale_evaluate import INDEX
from .semantic_scale_select import CONTROL, OUT, SELECTED
from .submission import validate_answer


def verify(answer: Path = Path("answer_semantic_15000.csv"),
           package: Path = Path("models/semantic_15000_v1")) -> dict:
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    def check(path: Path, expected: str) -> None:
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen SHA256 differs: {path}")

    for path, expected in (
        (answer, config["answer_sha256"]),
        (package / "ranker.cbm", config["model_sha256"]),
        (package / "training.json", config["training_sha256"]),
        (package / "item_id_to_vector_row.parquet", config["item_id_mapping_sha256"]),
        (MODEL / "onnx/model.onnx", config["e5_weights_sha256"]),
        (MODEL / "tokenizer.json", config["e5_tokenizer_sha256"]),
        (VECTORS / "item_vectors.npy", config["item_vectors_sha256"]),
        (VECTORS / "unique_vectors.npy", config["unique_vectors_sha256"]),
        (VECTORS / "item_to_text.npy", config["item_to_text_sha256"]),
        (Path("models/ranker_v1/all_train_signals.joblib"),
         config["all_train_signals_sha256"]),
        (Path("uv.lock"), config["uv_lock_sha256"]),
        (CONTROL / "answer_semantic.csv", config["control_answer_sha256"]),
    ):
        check(path, expected)
    data = Path("src/avito_test")
    if source_manifest(data) != config["sources"]:
        raise ValueError("Input data differs from the frozen prediction")
    queries = pl.read_parquet(data / "benchmark_queries.parquet",
                              columns=["query_id"])["query_id"].to_list()
    items = pl.read_parquet(data / "benchmark_items.parquet",
                            columns=["item_id"])["item_id"].to_list()
    validation = validate_answer(answer, queries, items)
    if validation != config["validation"] or validation != {
        "queries": 2452, "min_candidates": 50, "max_candidates": 50}:
        raise ValueError("Answer CSV has wrong IDs or candidate count")
    mapping = pl.read_parquet(package / "item_id_to_vector_row.parquet")
    index = joblib.load(INDEX, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet", columns=["item_id"])
    if (mapping.columns != ["item_row", "item_id"] or
            not np.array_equal(mapping["item_row"].to_numpy(),
                               np.arange(mapping.height, dtype=np.int32)) or
            mapping["item_id"].to_list() != index.item_ids or
            metadata["item_id"].to_list() != index.item_ids):
        raise ValueError("item_id to vector-row mapping changed")
    if np.load(VECTORS / "item_vectors.npy", mmap_mode="r").shape != (
            config["item_vector_rows"], 384):
        raise ValueError("Item vector matrix shape changed")
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    if decision != config["dev_decision"] or decision["selected"] not in (
            "B15000", "D15000"):
        raise ValueError("Model differs from the frozen dev selection")
    if sha256_file(SELECTED) != json.loads((OUT / "selection.json").read_text(
            encoding="utf-8"))["selected_sha256"]:
        raise ValueError("OOF group selection changed")
    report = {"variant": config["variant"], "answer_sha256": config["answer_sha256"],
              **validation, "item_vector_rows": mapping.height,
              "e5_revision": config["e5_revision"]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    verify()
