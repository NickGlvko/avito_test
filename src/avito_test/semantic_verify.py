"""Verify the frozen E5 answer, model files, and item-to-vector alignment."""
# Confirm that item vector rows still align with corpus item IDs.
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from .data import sha256_file, source_manifest
from .semantic_embeddings import MODEL, OUT as VECTORS
from .semantic_evaluate import INDEX
from .submission import validate_answer


def verify(answer: Path = Path("answer_semantic.csv"),
           package: Path = Path("models/semantic_v1")) -> dict:
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    def check(path: Path, expected: str) -> None:
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")

    check(answer, config["answer_sha256"])
    check(package / "ranker.cbm", config["model_sha256"])
    check(package / "training.json", config["training_config_sha256"])
    check(package / "item_id_to_vector_row.parquet", config["item_id_mapping_sha256"])
    check(MODEL / "onnx/model.onnx", config["e5_sha256"])
    check(MODEL / "tokenizer.json", config["e5_tokenizer_sha256"])
    check(VECTORS / "item_vectors.npy", config["item_vectors_sha256"])
    check(VECTORS / "unique_vectors.npy", config["unique_vectors_sha256"])
    check(VECTORS / "item_to_text.npy", config["item_to_text_sha256"])
    check(Path("models/ranker_v1/all_train_signals.joblib"),
          config["all_train_signals_sha256"])
    check(Path("uv.lock"), config["uv_lock_sha256"])

    data = Path("src/avito_test")
    if source_manifest(data) != config["sources"]:
        raise ValueError("Input Parquet files differ from the submitted prediction")
    query_ids = pl.read_parquet(data / "benchmark_queries.parquet",
                                columns=["query_id"])["query_id"].to_list()
    benchmark_ids = pl.read_parquet(data / "benchmark_items.parquet",
                                    columns=["item_id"])["item_id"].to_list()
    csv_result = validate_answer(answer, query_ids, benchmark_ids)
    if csv_result != config["validation"] or csv_result != {
        "queries": 2452, "min_candidates": 50, "max_candidates": 50}:
        raise ValueError("Answer CSV does not meet the frozen 2452-by-50 contract")

    mapping = pl.read_parquet(package / "item_id_to_vector_row.parquet")
    if mapping.columns != ["item_row", "item_id"]:
        raise ValueError("Unexpected item-to-vector mapping schema")
    rows = mapping["item_row"].to_numpy()
    if not np.array_equal(rows, np.arange(mapping.height, dtype=rows.dtype)):
        raise ValueError("Vector row indices are not contiguous or in order")
    index = joblib.load(INDEX, mmap_mode="r")
    metadata_ids = pl.read_parquet(INDEX.parent / "items.parquet",
                                   columns=["item_id"])["item_id"].to_list()
    if mapping["item_id"].to_list() != index.item_ids or metadata_ids != index.item_ids:
        raise ValueError("item_id order differs from the text index or vector metadata")
    vector_shape = np.load(VECTORS / "item_vectors.npy", mmap_mode="r").shape
    if vector_shape != (mapping.height, 384) or mapping.height != config["item_vector_rows"]:
        raise ValueError("Vector matrix and item_id mapping have different lengths")
    if config["variant"] != "B" or config["decision"]["selected"] != "B":
        raise ValueError("The frozen semantic model is not dev-selected variant B")
    result = {"answer_sha256": config["answer_sha256"], **csv_result,
              "item_vector_rows": mapping.height, "vector_dimension": vector_shape[1],
              "e5_revision": config["e5_revision"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    verify()
