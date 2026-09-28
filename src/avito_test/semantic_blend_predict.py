"""Offline benchmark prediction for the dev-selected B blend or B15000 fallback."""
# Apply only the blend weight fixed on dev to benchmark queries.
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
from .ranker import FeatureRetriever, load_feature_items
from .semantic_benchmark import REVISION
from .semantic_blend import CONTROL, blended_scores, rank_all, score_candidate_pool
from .semantic_blend_control import ANSWER as CONTROL_ANSWER, CONTROL as CONTROL_PATH, OUT
from .semantic_embeddings import MODEL, OUT as VECTORS, encode_queries
from .semantic_scale_evaluate import INDEX, MODELS
from .semantic_search import SemanticIndex
from .submission import write_answer


DATA = Path("src/avito_test")
PACKAGE = Path("models/semantic_blend_v1")
ANSWER = Path("answer_semantic_blend.csv")
SIGNALS = Path("models/ranker_v1/all_train_signals.joblib")


def predict() -> dict:
    start = time.perf_counter()
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    if sha256_file(Path(decision["dev_per_query"])) != decision["dev_sha256"]:
        raise ValueError("Selected blend no longer matches full dev")
    if any(sha256_file(MODELS[name]) != decision[f"{name}_model_sha256"]
           for name in ("B3000", "B15000")):
        raise ValueError("Frozen ranker weights changed")
    selected = decision["selected"]
    fallback = selected == CONTROL
    answer = (OUT / "answer_B15000_repeat.csv") if fallback else ANSWER
    benchmark = pl.read_parquet(DATA / "benchmark_queries.parquet")
    encode_queries(benchmark["search_query"].to_list())
    semantic = SemanticIndex(benchmark["search_query"].to_list(), threads=4)
    index = joblib.load(INDEX, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    if metadata["item_id"].to_list() != index.item_ids:
        raise ValueError("Text index and semantic item vectors differ in row order")
    features = load_feature_items(DATA, INDEX.parent, index.item_ids, source_manifest(DATA))
    signals = joblib.load(SIGNALS, mmap_mode="r")
    retriever = FeatureRetriever(index, signals, metadata["in_benchmark"].to_numpy(),
                                 features)
    models = {}
    for name in ("B3000", "B15000"):
        model = CatBoostRanker()
        model.load_model(str(MODELS[name]))
        models[name] = model
    predictions = []
    empty = np.empty(0, dtype=np.int32)
    for number, query in enumerate(benchmark.iter_rows(named=True), 1):
        ids, _, old, new = score_candidate_pool(retriever, semantic, models,
                                                 query, empty, True)
        values = new if fallback else blended_scores(old, new, decision["alpha"])
        chosen = rank_all(ids, values)[:50]
        predictions.append([index.item_ids[int(row)] for row in chosen])
        if number % 250 == 0:
            print(f"Blend prediction: {number}/{benchmark.height}; "
                  f"{time.perf_counter()-start:.1f}s", flush=True)
    corpus_ids = pl.read_parquet(DATA / "benchmark_items.parquet",
                                 columns=["item_id"])["item_id"].to_list()
    validation = write_answer(answer, benchmark["query_id"].to_list(),
                              predictions, corpus_ids)
    answer_sha = sha256_file(answer)
    if fallback:
        expected = sha256_file(CONTROL_PATH / CONTROL_ANSWER.name)
        if answer_sha != expected or answer_sha != sha256_file(CONTROL_ANSWER):
            raise ValueError("Repeated offline B15000 answer changed")
        result = {"selected": selected, "answer": str(answer),
                  "repeat_sha256": answer_sha, "control_sha256": expected,
                  "validation": validation, "seconds": time.perf_counter()-start}
        (OUT / "fallback_repeat.json").write_text(json.dumps(result, indent=2),
                                                   encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return result
    PACKAGE.mkdir(parents=True, exist_ok=True)
    for name in ("B3000", "B15000"):
        shutil.copy2(MODELS[name], PACKAGE / f"{name}.cbm")
    mapping = PACKAGE / "item_id_to_vector_row.parquet"
    shutil.copy2(Path("models/semantic_15000_v1/item_id_to_vector_row.parquet"), mapping)
    if pl.read_parquet(mapping)["item_id"].to_list() != index.item_ids:
        raise ValueError("Frozen item_id to vector-row mapping differs")
    state = json.loads((VECTORS / "encode_state.json").read_text(encoding="utf-8"))
    if np.load(VECTORS / "item_vectors.npy", mmap_mode="r").shape != (len(index.item_ids), 384):
        raise ValueError("Unexpected semantic item vector matrix")
    config = {"selected": selected, "alpha": decision["alpha"],
              "answer": str(answer), "answer_sha256": answer_sha,
              "validation": validation, "dev_decision": decision,
              "B3000_model_sha256": sha256_file(PACKAGE / "B3000.cbm"),
              "B15000_model_sha256": sha256_file(PACKAGE / "B15000.cbm"),
              "control_answer_sha256": sha256_file(CONTROL_PATH / CONTROL_ANSWER.name),
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
              "seconds": time.perf_counter()-start,
              "peak_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset",
                                        psutil.Process().memory_info().rss)}
    (PACKAGE / "config.json").write_text(json.dumps(config, indent=2,
                                                       ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"answer": str(answer), "sha256": answer_sha, **validation},
                     ensure_ascii=False), flush=True)
    return config


if __name__ == "__main__":
    predict()
