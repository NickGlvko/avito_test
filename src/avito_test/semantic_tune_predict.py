"""Offline benchmark prediction for a dev-selected tuned B ranker or 50/50 blend."""
# Use only the dev-frozen tree count and blend rule at inference.
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
from .ranker import FeatureRetriever, TEXT_FEATURES, load_feature_items, pool_ids
from .semantic_benchmark import REVISION
from .semantic_blend import blended_scores, normalize_scores, rank_all
from .semantic_embeddings import MODEL, OUT as VECTORS, encode_queries
from .semantic_scale_evaluate import INDEX
from .semantic_search import SemanticIndex, semantic_features
from .semantic_tune_cache import OUT
from .semantic_tune_control import ANSWER as CONTROL_ANSWER, EXPECTED_SHA256
from .semantic_tune_select import SELECTED_MODEL
from .submission import write_answer


DATA = Path("src/avito_test")
PACKAGE = Path("models/semantic_tune_v1")
ANSWER = Path("answer_semantic_tuned.csv")
SIGNALS = Path("models/ranker_v1/all_train_signals.joblib")
BLEND_PACKAGE = Path("models/semantic_blend_v1")


def predict() -> dict:
    started = time.perf_counter()
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    if decision["selected"] not in ("new_ranker", "half_blend"):
        raise ValueError("No tuned model improved the submitted dev control")
    if (sha256_file(OUT / "final/dev/per_query.parquet") != decision["dev_sha256"] or
            sha256_file(SELECTED_MODEL) != decision["checkpoint_model_sha256"] or
            sha256_file(CONTROL_ANSWER) != EXPECTED_SHA256):
        raise ValueError("Frozen dev choice or submitted answer changed")
    benchmark = pl.read_parquet(DATA / "benchmark_queries.parquet")
    encode_queries(benchmark["search_query"].to_list())
    semantic = SemanticIndex(benchmark["search_query"].to_list(), threads=4)
    index = joblib.load(INDEX, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    if metadata["item_id"].to_list() != index.item_ids:
        raise ValueError("Text index and semantic item vectors differ in row order")
    features = load_feature_items(DATA, INDEX.parent, index.item_ids, source_manifest(DATA))
    retriever = FeatureRetriever(index, joblib.load(SIGNALS, mmap_mode="r"),
                                 metadata["in_benchmark"].to_numpy(), features)
    models = {}
    for name, path in (("B3000", BLEND_PACKAGE / "B3000.cbm"),
                       ("B15000", BLEND_PACKAGE / "B15000.cbm"),
                       ("new", SELECTED_MODEL)):
        model = CatBoostRanker()
        model.load_model(str(path))
        models[name] = model
    predictions = []
    empty = np.empty(0, dtype=np.int32)
    predict_seconds = 0.
    for number, query in enumerate(benchmark.iter_rows(named=True), 1):
        sources, text_scores, allowed = retriever.retrieve_with_scores(
            query, empty, True)
        ids = pool_ids(sources)
        lexical = retriever.features(query, ids, sources, text_scores, allowed)
        embedding = semantic_features(semantic, query, ids, sources, False)
        x = np.column_stack((lexical[:, :len(TEXT_FEATURES)], embedding)).astype(np.float32)
        tick = time.perf_counter()
        old = np.asarray(models["B3000"].predict(x, thread_count=1), dtype=np.float64)
        base = np.asarray(models["B15000"].predict(x, thread_count=1), dtype=np.float64)
        new = np.asarray(models["new"].predict(x, thread_count=1), dtype=np.float64)
        control = blended_scores(old, base, 0.75)
        scores = new if decision["selected"] == "new_ranker" else (
            0.5*normalize_scores(new) + 0.5*normalize_scores(control))
        chosen = rank_all(ids, scores)[:50]
        predict_seconds += time.perf_counter()-tick
        predictions.append([index.item_ids[int(row)] for row in chosen])
        if number % 250 == 0:
            print(f"Tuned prediction: {number}/{benchmark.height}; "
                  f"{time.perf_counter()-started:.1f}s", flush=True)
    corpus_ids = pl.read_parquet(DATA / "benchmark_items.parquet",
                                 columns=["item_id"])["item_id"].to_list()
    validation = write_answer(ANSWER, benchmark["query_id"].to_list(),
                              predictions, corpus_ids)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    for name in ("B3000", "B15000"):
        shutil.copy2(BLEND_PACKAGE / f"{name}.cbm", PACKAGE / f"{name}.cbm")
    shutil.copy2(SELECTED_MODEL, PACKAGE / "new_ranker.cbm")
    shutil.copy2(BLEND_PACKAGE / "item_id_to_vector_row.parquet",
                 PACKAGE / "item_id_to_vector_row.parquet")
    state = json.loads((VECTORS / "encode_state.json").read_text(encoding="utf-8"))
    config = {"selected": decision["selected"], "checkpoint": decision["selected_checkpoint"],
              "answer": str(ANSWER), "answer_sha256": sha256_file(ANSWER),
              "validation": validation, "dev_decision": decision,
              "submitted_answer_sha256": EXPECTED_SHA256,
              "models": {name: sha256_file(PACKAGE / f"{name}.cbm") for name in
                         ("B3000", "B15000", "new_ranker")},
              "item_id_mapping_sha256": sha256_file(
                  PACKAGE / "item_id_to_vector_row.parquet"),
              "item_vector_rows": len(index.item_ids),
              "item_vectors_sha256": sha256_file(VECTORS / "item_vectors.npy"),
              "unique_vectors_sha256": state["vector_sha256"],
              "item_to_text_sha256": state["inverse_sha256"],
              "e5_revision": REVISION, "e5_model": str(MODEL),
              "e5_weights_sha256": sha256_file(MODEL / "onnx/model.onnx"),
              "e5_tokenizer_sha256": sha256_file(MODEL / "tokenizer.json"),
              "all_train_signals_sha256": sha256_file(SIGNALS),
              "uv_lock_sha256": sha256_file(Path("uv.lock")),
              "code_sha256": sha256_file(Path(__file__)),
              "sources": source_manifest(DATA),
              "seconds": time.perf_counter()-started,
              "model_predict_seconds": predict_seconds,
              "peak_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset",
                                        psutil.Process().memory_info().rss)}
    (PACKAGE / "config.json").write_text(json.dumps(config, indent=2,
                                                       ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"answer": str(ANSWER), "sha256": config["answer_sha256"],
                      **validation}, ensure_ascii=False), flush=True)
    return config


if __name__ == "__main__":
    predict()
