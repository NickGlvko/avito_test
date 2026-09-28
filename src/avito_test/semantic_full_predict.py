"""Reproduce an improved full-pool B answer entirely from local artifacts."""
# Restore item_id from the frozen corpus mapping after ranking.
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
from .semantic_blend import rank_all
from .semantic_embeddings import MODEL, OUT as VECTORS, encode_queries
from .semantic_full_evaluate import CONTROL, DECISION
from .semantic_full_oof import OUT
from .semantic_scale_evaluate import INDEX
from .semantic_search import SemanticIndex, semantic_features
from .submission import write_answer


DATA = Path("src/avito_test")
PACKAGE = Path("models/semantic_full_v1")
ANSWER = Path("answer_semantic_full.csv")
SIGNALS = Path("models/ranker_v1/all_train_signals.joblib")
SUBMITTED = Path("answer_semantic_blend.csv")
SUBMITTED_SHA256 = "8124b99e143fcc1091705be80e9bc4c2b62a42c3cf600909ae352afdf54ee707"
BLEND_PACKAGE = Path("models/semantic_blend_v1")


def predict() -> dict:
    started = time.perf_counter()
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    selected = decision["selected"]
    if selected == CONTROL or selected not in ("full_unweighted", "full_balanced"):
        raise ValueError("No new full-pool model improved the submitted blend on dev")
    model_path = OUT / f"{selected}.cbm"
    report = json.loads((OUT / f"{selected}.json").read_text(encoding="utf-8"))
    if (sha256_file(OUT / "evaluation/dev/per_query.parquet") != decision["dev_sha256"] or
            sha256_file(model_path) != decision["models"][selected] or
            sha256_file(model_path) != report["model_sha256"] or
            sha256_file(SUBMITTED) != SUBMITTED_SHA256):
        raise ValueError("Frozen dev decision, model or submitted answer changed")
    benchmark = pl.read_parquet(DATA / "benchmark_queries.parquet")
    encode_queries(benchmark["search_query"].to_list())
    semantic = SemanticIndex(benchmark["search_query"].to_list(), threads=4)
    index = joblib.load(INDEX, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    if metadata["item_id"].to_list() != index.item_ids:
        raise ValueError("Text index and E5 item vectors differ in row order")
    items = load_feature_items(DATA, INDEX.parent, index.item_ids, source_manifest(DATA))
    retriever = FeatureRetriever(
        index, joblib.load(SIGNALS, mmap_mode="r"),
        metadata["in_benchmark"].to_numpy(), items
    )
    model = CatBoostRanker()
    model.load_model(str(model_path))
    answers = []
    empty = np.empty(0, dtype=np.int32)
    model_seconds = 0.
    for number, query in enumerate(benchmark.iter_rows(named=True), 1):
        sources, text_scores, allowed = retriever.retrieve_with_scores(query, empty, True)
        ids = pool_ids(sources)
        lexical = retriever.features(query, ids, sources, text_scores, allowed)
        semantic_part = semantic_features(semantic, query, ids, sources, False)
        x = np.column_stack((lexical[:, :len(TEXT_FEATURES)], semantic_part)).astype(np.float32)
        tick = time.perf_counter()
        score = np.asarray(model.predict(x, thread_count=1), dtype=np.float64)
        model_seconds += time.perf_counter()-tick
        chosen = rank_all(ids, score)[:50]
        answers.append([index.item_ids[int(row)] for row in chosen])
        if number % 250 == 0:
            print(f"Full-pool prediction: {number}/{benchmark.height}; "
                  f"{time.perf_counter()-started:.1f}s", flush=True)
    corpus_ids = pl.read_parquet(DATA / "benchmark_items.parquet",
                                 columns=["item_id"])["item_id"].to_list()
    validation = write_answer(ANSWER, benchmark["query_id"].to_list(),
                              answers, corpus_ids)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, PACKAGE / "ranker.cbm")
    mapping = PACKAGE / "item_id_to_vector_row.parquet"
    shutil.copy2(BLEND_PACKAGE / mapping.name, mapping)
    if pl.read_parquet(mapping)["item_id"].to_list() != index.item_ids:
        raise ValueError("Saved item_id to vector-row mapping differs")
    state = json.loads((VECTORS / "encode_state.json").read_text(encoding="utf-8"))
    if np.load(VECTORS / "item_vectors.npy", mmap_mode="r").shape != (len(index.item_ids), 384):
        raise ValueError("Unexpected E5 item vector matrix")
    memory = psutil.Process().memory_info()
    config = {
        "selected": selected, "answer": str(ANSWER),
        "answer_sha256": sha256_file(ANSWER), "validation": validation,
        "dev_decision": decision, "submitted_answer_sha256": SUBMITTED_SHA256,
        "model_sha256": sha256_file(PACKAGE / "ranker.cbm"),
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
        "code_sha256": sha256_file(Path(__file__)),
        "sources": source_manifest(DATA),
        "seconds": time.perf_counter()-started,
        "model_predict_seconds": model_seconds,
        "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
    }
    (PACKAGE / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"answer": str(ANSWER), "sha256": config["answer_sha256"],
                      **validation}, ensure_ascii=False), flush=True)
    return config


if __name__ == "__main__":
    predict()
