"""Validate final answer, selected fusion, weights and local dependencies."""
# Validate final IDs, model hashes and the selected fusion rule.
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from .data import sha256_file, source_manifest
from .semantic_embeddings import MODEL, OUT as VECTORS
from .semantic_final_evaluate import DECISION, OUT
from .semantic_final_predict import (ANSWER, B_ANSWER, DATA, PACKAGE,
                                     SIGNALS, SUBMITTED)
from .semantic_scale_evaluate import INDEX
from .submission import validate_answer


def verify() -> dict:
    config = json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    if (decision["selected"] == "D_full_unweighted" or
            config["dev_decision"] != decision or
            config["selected"] != decision["selected"] or
            sha256_file(OUT / "evaluation/dev/per_query.parquet") !=
            decision["dev_sha256"] or
            config["submitted_d_sha256"] != sha256_file(SUBMITTED) or
            config["submitted_b_sha256"] != sha256_file(B_ANSWER)):
        raise ValueError("Final answer differs from frozen dev choice or controls")
    expected = [
        (ANSWER, config["answer_sha256"]),
        (PACKAGE / "item_id_to_vector_row.parquet",
         config["item_id_mapping_sha256"]),
        (MODEL / "onnx/model.onnx", config["e5_weights_sha256"]),
        (MODEL / "tokenizer.json", config["e5_tokenizer_sha256"]),
        (VECTORS / "item_vectors.npy", config["item_vectors_sha256"]),
        (VECTORS / "unique_vectors.npy", config["unique_vectors_sha256"]),
        (VECTORS / "item_to_text.npy", config["item_to_text_sha256"]),
        (SIGNALS, config["all_train_signals_sha256"]),
        (Path("uv.lock"), config["uv_lock_sha256"]),
    ]
    expected += [(PACKAGE / f"ranker_{name}.cbm", value) for name, value in
                 config["models_sha256"].items()]
    for path, checksum in expected:
        if sha256_file(path) != checksum:
            raise ValueError(f"Frozen dependency changed: {path}")
    if (config["models_sha256"] != decision["models"] or
            config["sources"] != source_manifest(DATA) or
            any(sha256_file(Path(__file__).with_name(name)) != checksum
                for name, checksum in config["code_sha256"].items())):
        raise ValueError("Final models, code or input data changed")
    queries = pl.read_parquet(DATA / "benchmark_queries.parquet",
                              columns=["query_id"])["query_id"].to_list()
    items = pl.read_parquet(DATA / "benchmark_items.parquet",
                            columns=["item_id"])["item_id"].to_list()
    validation = validate_answer(ANSWER, queries, items)
    if (validation != config["validation"] or
            validation != {"queries": 2452, "min_candidates": 50,
                           "max_candidates": 50}):
        raise ValueError("Final CSV has invalid IDs or candidate count")
    mapping = pl.read_parquet(PACKAGE / "item_id_to_vector_row.parquet")
    index = joblib.load(INDEX, mmap_mode="r")
    if (mapping.columns != ["item_row", "item_id"] or
            not np.array_equal(mapping["item_row"].to_numpy(),
                               np.arange(mapping.height, dtype=np.int32)) or
            mapping["item_id"].to_list() != index.item_ids or
            np.load(VECTORS / "item_vectors.npy", mmap_mode="r").shape !=
            (mapping.height, 384)):
        raise ValueError("Item IDs and E5 vector rows are not aligned")
    result = {"selected": decision["selected"],
              "answer_sha256": config["answer_sha256"], **validation}
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    verify()
