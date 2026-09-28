"""Generate the benchmark CSV from the dev-selected semantic model, offline."""
# Prediction uses all train interactions only after validation is finished.
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
from .semantic_embeddings import MODEL as E5_MODEL, OUT as E5_VECTORS
from .semantic_benchmark import REVISION as E5_REVISION
from .semantic_evaluate import INDEX, MODEL_FILES
from .semantic_search import SemanticIndex, semantic_features
from .submission import write_answer

DATA = Path("src/avito_test")
DECISION = Path("artifacts/semantic/decision.json")
ANSWER = Path("answer_semantic.csv")
PACKAGE = Path("models/semantic_v1")


def predict(answer: Path = ANSWER) -> dict:
    start = time.perf_counter()
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    variant = decision["selected"]
    if variant not in ("B", "C", "D"):
        raise ValueError("No semantic variant was selected on dev; retain answer_ranker.csv")
    dev_paths = list(Path("artifacts/semantic/evaluation/dev/full").glob("*/per_query.parquet"))
    if not any(sha256_file(path) == decision["dev_per_query_sha256"] for path in dev_paths):
        raise ValueError("Frozen dev decision does not match its evaluation")
    queries = pl.read_parquet(DATA / "benchmark_queries.parquet")
    from .semantic_embeddings import encode_queries
    encode_queries(queries["search_query"].to_list())
    semantic = SemanticIndex(queries["search_query"].to_list(), threads=4)
    index = joblib.load(INDEX, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    if metadata["item_id"].to_list() != index.item_ids:
        raise ValueError("Item metadata and vector rows have different item_id order")
    items = load_feature_items(DATA, INDEX.parent, index.item_ids, source_manifest(DATA))
    signals = joblib.load(Path("models/ranker_v1/all_train_signals.joblib"), mmap_mode="r")
    retriever = FeatureRetriever(index, signals, metadata["in_benchmark"].to_numpy(), items)
    model = CatBoostRanker()
    model.load_model(str(MODEL_FILES[variant]))
    predictions = []
    for number, query in enumerate(queries.iter_rows(named=True), start=1):
        # No labels exist here. The augmented scope with an empty set is exactly benchmark_items.
        empty = np.empty(0, dtype=np.int32)
        base, scores, allowed = retriever.retrieve_with_scores(query, empty, augmented=True)
        old = pool_ids(base)
        if variant == "B":
            candidates = old
            x = retriever.features(query, candidates, base, scores, allowed)[:, :len(TEXT_FEATURES)]
            x = np.column_stack((x, semantic_features(semantic, query, candidates, base, False)))
        else:
            dense = semantic.retrieve(query["search_query"], query["search_location_id"], empty, True)
            results = {**base, **dense}
            candidates = np.unique(np.concatenate((old, *dense.values()))).astype(np.int32)
            x = retriever.features(query, candidates, base, scores, allowed)[:, :len(TEXT_FEATURES)]
            if variant == "D":
                x = np.column_stack((x, semantic_features(semantic, query, candidates, results, True)))
        scores = model.predict(x, thread_count=1)
        ranking = model_ranking(candidates, scores)
        predictions.append([index.item_ids[int(item)] for item in ranking])
        if number % 250 == 0:
            print(f"Semantic prediction: {number}/{queries.height}; {time.perf_counter()-start:.1f}s", flush=True)
    benchmark_ids = pl.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id"])["item_id"].to_list()
    validation = write_answer(answer, queries["query_id"].to_list(), predictions, benchmark_ids)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    selected_weights = PACKAGE / "ranker.cbm"
    shutil.copy2(MODEL_FILES[variant], selected_weights)
    mapping = PACKAGE / "item_id_to_vector_row.parquet"
    pl.DataFrame({"item_row": np.arange(len(index.item_ids), dtype=np.int32),
                  "item_id": index.item_ids}).write_parquet(mapping)
    item_vectors = np.load(E5_VECTORS / "item_vectors.npy", mmap_mode="r")
    if item_vectors.shape != (len(index.item_ids), 384):
        raise ValueError("Item embedding matrix and item_id mapping have different lengths")
    vector_state = json.loads((E5_VECTORS / "encode_state.json").read_text(encoding="utf-8"))
    training_config = Path(str(MODEL_FILES[variant]).replace(".cbm", ".json"))
    shutil.copy2(training_config, PACKAGE / "training.json")
    report = {"variant": variant, "answer": str(answer), "answer_sha256": sha256_file(answer),
              "validation": validation, "model_sha256": sha256_file(selected_weights),
              "model_source": str(MODEL_FILES[variant]),
              "e5_model": str(E5_MODEL),
              "e5_revision": E5_REVISION,
              "e5_sha256": sha256_file(E5_MODEL / "onnx/model.onnx"),
              "e5_tokenizer_sha256": sha256_file(E5_MODEL / "tokenizer.json"),
              "item_id_mapping": str(mapping), "item_id_mapping_sha256": sha256_file(mapping),
              "item_vector_rows": len(index.item_ids),
              "item_vectors_path": str(E5_VECTORS / "item_vectors.npy"),
              "item_vectors_sha256": sha256_file(E5_VECTORS / "item_vectors.npy"),
              "unique_vectors_sha256": vector_state["vector_sha256"],
              "item_to_text_sha256": vector_state["inverse_sha256"],
              "training_config_sha256": sha256_file(PACKAGE / "training.json"),
              "uv_lock_sha256": sha256_file(Path("uv.lock")),
              "prediction_code_sha256": sha256_file(Path(__file__)),
              "decision": decision, "sources": source_manifest(DATA),
              "all_train_signals_sha256": sha256_file(Path("models/ranker_v1/all_train_signals.joblib")),
              "seconds": time.perf_counter()-start,
              "peak_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset",
                                        psutil.Process().memory_info().rss)}
    (PACKAGE / "config.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"answer": str(answer), "sha256": report["answer_sha256"],
                      **validation}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    predict()
