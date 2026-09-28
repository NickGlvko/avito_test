"""Offline benchmark prediction from the dev-selected 15k B or D ranker."""
# Load the dev-selected model and preserve the benchmark item IDs.
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import psutil
from catboost import CatBoostRanker

from .data import sha256_file, source_manifest
from .ranker import FeatureRetriever, TEXT_FEATURES, load_feature_items, model_ranking, pool_ids
from .semantic_benchmark import REVISION
from .semantic_embeddings import MODEL, OUT as VECTORS, encode_queries
from .semantic_scale_evaluate import INDEX, MODELS
from .semantic_scale_select import CONTROL, OUT
from .semantic_search import SemanticIndex, semantic_features
from .submission import write_answer


DATA = Path("src/avito_test")
PACKAGE = Path("models/semantic_15000_v1")
ANSWER = Path("answer_semantic_15000.csv")


def predict(answer: Path = ANSWER) -> dict:
    start = time.perf_counter()
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    selected = decision["selected"]
    if selected not in ("B15000", "D15000"):
        raise ValueError("No larger model selected on dev; keep submitted answer_semantic.csv")
    if not any(sha256_file(path) == decision["dev_per_query_sha256"] for path in
               (OUT / "evaluation/dev/full").glob("*/per_query.parquet")):
        raise ValueError("Frozen dev decision does not match full-dev predictions")
    benchmark = pl.read_parquet(DATA / "benchmark_queries.parquet")
    encode_queries(benchmark["search_query"].to_list())
    semantic = SemanticIndex(benchmark["search_query"].to_list(), threads=4)
    index = joblib.load(INDEX, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    if metadata["item_id"].to_list() != index.item_ids:
        raise ValueError("Text index and item vector rows are misaligned")
    feature_items = load_feature_items(DATA, INDEX.parent, index.item_ids,
                                       source_manifest(DATA))
    signals = joblib.load(Path("models/ranker_v1/all_train_signals.joblib"), mmap_mode="r")
    retriever = FeatureRetriever(index, signals, metadata["in_benchmark"].to_numpy(),
                                 feature_items)
    model = CatBoostRanker()
    model.load_model(str(MODELS[selected]))
    predictions = []
    for number, query in enumerate(benchmark.iter_rows(named=True), 1):
        empty = np.empty(0, dtype=np.int32)
        base, scores, allowed = retriever.retrieve_with_scores(query, empty, True)
        old_pool = pool_ids(base)
        if selected == "B15000":
            candidates = old_pool
            text = retriever.features(query, candidates, base, scores, allowed)[:, :len(TEXT_FEATURES)]
            features = np.column_stack((text, semantic_features(
                semantic, query, candidates, base, False)))
        else:
            dense = semantic.retrieve(query["search_query"], query["search_location_id"],
                                      empty, True)
            results = {**base, **dense}
            candidates = np.unique(np.concatenate((old_pool, *dense.values()))).astype(np.int32)
            text = retriever.features(query, candidates, base, scores, allowed)[:, :len(TEXT_FEATURES)]
            features = np.column_stack((text, semantic_features(
                semantic, query, candidates, results, True)))
        values = model.predict(features, thread_count=1)
        prediction = model_ranking(candidates, values)
        predictions.append([index.item_ids[int(row)] for row in prediction])
        if number % 250 == 0:
            print(f"Scaled semantic prediction: {number}/{benchmark.height}; "
                  f"{time.perf_counter()-start:.1f}s", flush=True)
    benchmark_items = pl.read_parquet(DATA / "benchmark_items.parquet",
                                      columns=["item_id"])["item_id"].to_list()
    validation = write_answer(answer, benchmark["query_id"].to_list(),
                              predictions, benchmark_items)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    weights = PACKAGE / "ranker.cbm"
    shutil.copy2(MODELS[selected], weights)
    training = Path(str(MODELS[selected]).replace(".cbm", ".json"))
    shutil.copy2(training, PACKAGE / "training.json")
    mapping_source = Path("models/semantic_v1/item_id_to_vector_row.parquet")
    mapping = PACKAGE / "item_id_to_vector_row.parquet"
    shutil.copy2(mapping_source, mapping)
    saved_ids = pl.read_parquet(mapping)["item_id"].to_list()
    if saved_ids != index.item_ids:
        raise ValueError("Frozen item_id to vector-row mapping changed")
    state = json.loads((VECTORS / "encode_state.json").read_text(encoding="utf-8"))
    vector_shape = np.load(VECTORS / "item_vectors.npy", mmap_mode="r").shape
    if vector_shape != (len(index.item_ids), 384):
        raise ValueError("Unexpected item vector shape")
    config = {"variant": selected, "answer": str(answer),
              "answer_sha256": sha256_file(answer), "validation": validation,
              "model_sha256": sha256_file(weights),
              "training_sha256": sha256_file(PACKAGE / "training.json"),
              "control_answer_sha256": sha256_file(CONTROL / "answer_semantic.csv"),
              "dev_decision": decision,
              "e5_revision": REVISION, "e5_model": str(MODEL),
              "e5_weights_sha256": sha256_file(MODEL / "onnx/model.onnx"),
              "e5_tokenizer_sha256": sha256_file(MODEL / "tokenizer.json"),
              "item_id_mapping_sha256": sha256_file(mapping),
              "item_vector_rows": len(index.item_ids),
              "item_vectors_sha256": sha256_file(VECTORS / "item_vectors.npy"),
              "unique_vectors_sha256": state["vector_sha256"],
              "item_to_text_sha256": state["inverse_sha256"],
              "all_train_signals_sha256": sha256_file(
                  Path("models/ranker_v1/all_train_signals.joblib")),
              "uv_lock_sha256": sha256_file(Path("uv.lock")),
              "code_sha256": sha256_file(Path(__file__)),
              "sources": source_manifest(DATA),
              "seconds": time.perf_counter()-start,
              "peak_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset",
                                        psutil.Process().memory_info().rss)}
    (PACKAGE / "config.json").write_text(json.dumps(config, indent=2,
                                                       ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"answer": str(answer), "sha256": config["answer_sha256"],
                      **validation}, ensure_ascii=False), flush=True)
    return config


if __name__ == "__main__":
    predict()
