"""Reproduce the dev-selected final B/D combination without network access."""
# Generate a new answer only for a variant selected on dev.
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import psutil

from .data import sha256_file, source_manifest
from .ranker import FeatureRetriever, TEXT_FEATURES, load_feature_items, pool_ids
from .semantic_benchmark import REVISION
from .semantic_embeddings import MODEL, OUT as VECTORS, encode_queries
from .semantic_final import CONTROL, blend_d, rrf_bd, select
from .semantic_final_evaluate import (B_MODEL, DB_MODEL, D_MODEL, DECISION,
                                      OUT, frozen_models)
from .semantic_full_d_predict import SIGNALS
from .semantic_scale_evaluate import INDEX
from .semantic_search import SemanticIndex, semantic_features
from .submission import write_answer


DATA = Path("src/avito_test")
PACKAGE = Path("models/semantic_final_v1")
ANSWER = Path("answer_final.csv")
SUBMITTED = Path("answer_semantic_full_D.csv")
B_ANSWER = Path("answer_semantic_full.csv")
MAPPING_SOURCE = Path("models/semantic_full_D_v1/item_id_to_vector_row.parquet")


def archived_answers(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["query_id", "answer"]:
            raise ValueError(f"Invalid frozen answer columns: {path}")
        result = {row["query_id"]: row["answer"].split(" ") for row in reader}
    if len(result) != 2452:
        raise ValueError(f"Frozen answer does not cover all queries: {path}")
    return result


def predict() -> dict:
    started = time.perf_counter()
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    selected = decision["selected"]
    if (selected == CONTROL or
            sha256_file(OUT / "evaluation/dev/per_query.parquet") !=
            decision["dev_sha256"] or
            sha256_file(SUBMITTED) != decision["submitted_answer_sha256"] or
            {name: sha256_file(path) for name, path in
             (("B", B_MODEL), ("D", D_MODEL), ("DB", DB_MODEL))} !=
            decision["models"]):
        raise ValueError("No frozen improving final choice or weights changed")
    models = frozen_models()
    benchmark = pl.read_parquet(DATA / "benchmark_queries.parquet")
    encode_queries(benchmark["search_query"].to_list())
    semantic = SemanticIndex(benchmark["search_query"].to_list(), threads=4)
    index = joblib.load(INDEX, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    if metadata["item_id"].to_list() != index.item_ids:
        raise ValueError("Text index and E5 vectors are not row-aligned")
    items = load_feature_items(DATA, INDEX.parent, index.item_ids,
                               source_manifest(DATA))
    retriever = FeatureRetriever(index, joblib.load(SIGNALS, mmap_mode="r"),
                                 metadata["in_benchmark"].to_numpy(), items)
    b_archived = archived_answers(B_ANSWER)
    d_archived = archived_answers(SUBMITTED)
    output = []
    empty = np.empty(0, dtype=np.int32)
    model_seconds = 0.0
    for number, query in enumerate(benchmark.iter_rows(named=True), 1):
        base, text_scores, allowed = retriever.retrieve_with_scores(
            query, empty, True)
        b_ids = pool_ids(base)
        b_text = retriever.features(query, b_ids, base, text_scores, allowed)
        b_semantic = semantic_features(semantic, query, b_ids, base, False)
        b_x = np.column_stack((b_text[:, :len(TEXT_FEATURES)], b_semantic))
        dense = semantic.retrieve(query["search_query"],
                                  query["search_location_id"], empty, True)
        sources = {**base, **dense}
        d_ids = np.unique(np.concatenate((b_ids, *dense.values()))).astype(np.int32)
        d_text = retriever.features(query, d_ids, base, text_scores, allowed)
        d_semantic = semantic_features(semantic, query, d_ids, sources, True, d_ids)
        d_x = np.column_stack((d_text[:, :len(TEXT_FEATURES)], d_semantic))
        tick = time.perf_counter()
        b_score = np.asarray(models["B"].predict(b_x, thread_count=1), dtype=np.float64)
        u_score = np.asarray(models["D"].predict(d_x, thread_count=1), dtype=np.float64)
        v_score = np.asarray(models["DB"].predict(d_x, thread_count=1), dtype=np.float64)
        model_seconds += time.perf_counter() - tick
        b_top = [index.item_ids[int(row)] for row in select(b_ids, b_score)]
        d_top = [index.item_ids[int(row)] for row in select(d_ids, u_score)]
        if (b_top != b_archived[query["query_id"]] or
                d_top != d_archived[query["query_id"]] or
                not np.array_equal(select(d_ids, rrf_bd(
                    b_ids, b_score, d_ids, u_score, 0.0)),
                                   select(b_ids, b_score)) or
                not np.array_equal(select(d_ids, rrf_bd(
                    b_ids, b_score, d_ids, u_score, 1.0)),
                                   select(d_ids, u_score))):
            raise ValueError("Frozen B/D benchmark answer or RRF endpoint changed")
        if selected.startswith("d_blend_"):
            score = blend_d(u_score, v_score,
                            float(selected.removeprefix("d_blend_")))
        elif selected.startswith("rrf_bd_"):
            score = rrf_bd(b_ids, b_score, d_ids, u_score,
                           float(selected.removeprefix("rrf_bd_")))
        else:
            raise ValueError("Unknown selected final combination")
        output.append([index.item_ids[int(row)] for row in select(d_ids, score)])
        if number % 250 == 0:
            print(f"Final prediction: {number}/{benchmark.height}; "
                  f"{time.perf_counter()-started:.1f}s", flush=True)
    corpus_ids = pl.read_parquet(DATA / "benchmark_items.parquet",
                                 columns=["item_id"])["item_id"].to_list()
    validation = write_answer(ANSWER, benchmark["query_id"].to_list(),
                              output, corpus_ids)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    for name, path in (("B", B_MODEL), ("D", D_MODEL), ("DB", DB_MODEL)):
        shutil.copy2(path, PACKAGE / f"ranker_{name}.cbm")
    mapping = PACKAGE / "item_id_to_vector_row.parquet"
    shutil.copy2(MAPPING_SOURCE, mapping)
    if pl.read_parquet(mapping)["item_id"].to_list() != index.item_ids:
        raise ValueError("Frozen item_id to vector-row mapping changed")
    state = json.loads((VECTORS / "encode_state.json").read_text(encoding="utf-8"))
    memory = psutil.Process().memory_info()
    config = {
        "selected": selected, "answer": str(ANSWER),
        "answer_sha256": sha256_file(ANSWER), "validation": validation,
        "dev_decision": decision,
        "models_sha256": {name: sha256_file(PACKAGE / f"ranker_{name}.cbm")
                          for name in models},
        "submitted_d_sha256": sha256_file(SUBMITTED),
        "submitted_b_sha256": sha256_file(B_ANSWER),
        "item_id_mapping_sha256": sha256_file(mapping),
        "item_vector_rows": len(index.item_ids),
        "item_vectors_sha256": sha256_file(VECTORS / "item_vectors.npy"),
        "unique_vectors_sha256": state["vector_sha256"],
        "item_to_text_sha256": state["inverse_sha256"],
        "e5_revision": REVISION, "e5_model": str(MODEL),
        "e5_weights_sha256": sha256_file(MODEL / "onnx/model.onnx"),
        "e5_tokenizer_sha256": sha256_file(MODEL / "tokenizer.json"),
        "all_train_signals_sha256": sha256_file(SIGNALS),
        "uv_lock_sha256": sha256_file(Path("uv.lock")),
        "code_sha256": {name: sha256_file(Path(__file__).with_name(name))
                        for name in ("semantic_final.py", "semantic_final_predict.py")},
        "sources": source_manifest(DATA),
        "seconds": time.perf_counter()-started,
        "model_predict_seconds": model_seconds,
        "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
    }
    (PACKAGE / "config.json").write_text(json.dumps(config, indent=2,
                                                      ensure_ascii=False),
                                         encoding="utf-8")
    print(json.dumps({"answer": str(ANSWER), "sha256": config["answer_sha256"],
                      **validation}, ensure_ascii=False), flush=True)
    return config


if __name__ == "__main__":
    predict()
