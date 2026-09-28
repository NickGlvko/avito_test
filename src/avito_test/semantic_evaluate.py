"""Paired A/B/C/D evaluation on exact union and benchmark+positive scopes."""
# Evaluate each ablation on the same full query groups and corpus scope.
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import psutil
from catboost import CatBoostRanker
from threadpoolctl import threadpool_limits

from .data import digest, enrich_train, sha256_file, source_manifest
from .ranker import (FeatureRetriever, TEXT_FEATURES, candidate_metrics,
                     group_queries, load_feature_items, pool_ids)
from .semantic_fit import OUT as MODELS
from .semantic_search import SemanticIndex, build_item_vectors, semantic_features

INDEX = Path("artifacts/expanded/cache/73915bb3bc082213/union_index.joblib")
SIGNALS = Path("artifacts/ranker/full/c392362a99e2c4d1/full_train_signals.joblib")
OLD_MODEL = Path("models/ranker_v1/text_model.cbm")
OUT = Path("artifacts/semantic/evaluation")
MODEL_FILES = {
    "B": MODELS / "B_existing_pool_semantic_features.cbm",
    "C": MODELS / "C_expanded_pool_no_semantic_features.cbm",
    "D": MODELS / "D_expanded_pool_semantic_features.cbm",
}
CHUNK = 48
_WORKER = None


def initialize(query_texts: list[str], active: tuple[str, ...]) -> None:
    global _WORKER
    threadpool_limits(limits=1)
    index = joblib.load(INDEX, mmap_mode="r")
    signals = joblib.load(SIGNALS, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    feature_items = pl.read_parquet(INDEX.parent / "features.parquet")
    retriever = FeatureRetriever(index, signals, metadata["in_benchmark"].to_numpy(), feature_items)
    semantic = SemanticIndex(query_texts, threads=1)
    models = {}
    for name, path in {"A": OLD_MODEL, **MODEL_FILES}.items():
        if name not in active:
            continue
        model = CatBoostRanker()
        model.load_model(str(path))
        models[name] = model
    _WORKER = retriever, semantic, models


def score_chunk(number: int, queries: list[dict], parts: str) -> tuple[int, int, int, float]:
    retriever, semantic, models = _WORKER
    records = []
    positive_details = []
    start_search = semantic.search_seconds
    for query in queries:
        relevant = np.asarray(query["relevant_rows"], dtype=np.int32)
        truth = set(map(int, relevant))
        for scheme, augmented in [("union", False), ("benchmark_plus_positives", True)]:
            base_results, scores, allowed = retriever.retrieve_with_scores(query, relevant, augmented)
            old_pool = pool_ids(base_results)
            dense = semantic.retrieve(query["search_query"], query["search_location_id"], relevant, augmented)
            results = {**base_results, **dense}
            expanded = np.unique(np.concatenate((old_pool, *dense.values()))).astype(np.int32)
            lexical = retriever.features(query, expanded, base_results, scores, allowed)[:, :len(TEXT_FEATURES)]
            old_positions = np.searchsorted(expanded, old_pool)
            old_features = lexical[old_positions]
            def ordered(ids: np.ndarray, values: np.ndarray) -> np.ndarray:
                return ids[np.lexsort((ids, -values))]

            rankings = {"A": ordered(old_pool,
                models["A"].predict(old_features, thread_count=1))}
            if "B" in models:
                b_semantic = semantic_features(semantic, query, old_pool, results, False)
                rankings["B"] = ordered(old_pool, models["B"].predict(
                    np.column_stack((old_features, b_semantic)), thread_count=1))
            if "C" in models:
                rankings["C"] = ordered(expanded,
                    models["C"].predict(lexical, thread_count=1))
            if "D" in models:
                d_semantic = semantic_features(semantic, query, expanded, results, True)
                rankings["D"] = ordered(expanded, models["D"].predict(
                    np.column_stack((lexical, d_semantic)), thread_count=1))
            predictions = {variant: ranking[:50] for variant, ranking in rankings.items()}
            old_set = set(map(int, old_pool))
            new_positive = truth & (set(map(int, expanded)) - old_set)
            history_only = set(map(int, base_results["history"])) - set(map(int, pool_ids(base_results, include_history=False)))
            standalone = {}
            for name in ("semantic_global", "semantic_local"):
                ids = dense[name]
                standalone[name + "_recall50"] = len(truth & set(map(int, ids[:50]))) / len(truth)
                standalone[name + "_recall200"] = len(truth & set(map(int, ids))) / len(truth)
            standalone["semantic_union_recall200"] = len(truth & set(map(int, np.concatenate(list(dense.values()))))) / len(truth)
            for variant, prediction in predictions.items():
                pool = old_pool if variant in ("A", "B") else expanded
                row = candidate_metrics(prediction, pool, truth, retriever.signals,
                                        query, base_results["history"], history_only)
                row.update({"stage": query["stage"], "group_id": query["group_id"],
                            "text": query["text"], "scheme": scheme, "variant": variant,
                            "new_semantic_positives": len(new_positive),
                            "new_semantic_positives_in_top50": len(new_positive & set(map(int, prediction))),
                            **standalone})
                records.append(row)
            source_positions = {name: {int(item): rank for rank, item in enumerate(dense[name], 1)}
                                for name in ("semantic_global", "semantic_local")}
            for item in relevant:
                value = int(item)
                detail = {"stage": query["stage"], "group_id": query["group_id"],
                          "text": query["text"], "scheme": scheme, "item_row": value,
                          "same_location": bool(retriever.signals.locations[value] == query["search_location_id"]),
                          "in_old_pool": value in old_set,
                          "semantic_global_rank": source_positions["semantic_global"].get(value, 0),
                          "semantic_local_rank": source_positions["semantic_local"].get(value, 0),
                          "semantic_cosine": float(semantic.similarity(query["search_query"],
                                                                        np.array([value], dtype=np.int32))[0])}
                for variant, ranking in rankings.items():
                    positions = np.flatnonzero(ranking == value)
                    detail[variant + "_rank"] = int(positions[0]) + 1 if len(positions) else 0
                positive_details.append(detail)
    pl.DataFrame(records, infer_schema_length=None,
        schema_overrides={f"{name}_recall@50": pl.Float64 for name in
            ("seen_item", "new_item", "same_location", "other_location",
             "history_source", "outside_history_source")}).write_parquet(
                Path(parts) / f"metrics_{number:05d}.parquet")
    pl.DataFrame(positive_details, infer_schema_length=None).write_parquet(
        Path(parts) / f"positives_{number:05d}.parquet")
    (Path(parts) / f"done_{number:05d}.json").write_text(json.dumps({"queries": len(queries)}), encoding="utf-8")
    memory = psutil.Process().memory_info()
    return number, len(queries), getattr(memory, "peak_wset", memory.rss), semantic.search_seconds - start_search


def evaluate(stage: str = "dev", limit: int = 3000, workers: int = 2) -> dict:
    if stage not in ("dev", "test") or limit < 0 or workers < 1:
        raise ValueError("Invalid stage, limit, or worker count")
    if stage == "test":
        decision_path = Path("artifacts/semantic/decision.json")
        if not decision_path.exists():
            raise ValueError("Diagnostic test requires a frozen full-dev decision")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        full_dev = list(Path("artifacts/semantic/evaluation/dev/full").glob("*/per_query.parquet"))
        if not any(sha256_file(path) == decision["dev_per_query_sha256"] for path in full_dev):
            raise ValueError("Frozen decision does not match full dev predictions")
        selected = decision["selected"]
        active = tuple(dict.fromkeys(("A", selected)))
    else:
        active = ("A", "B", "C", "D")
    train = enrich_train(pl.read_parquet("src/avito_test/train.parquet"))
    queries = group_queries(train, stage, limit)
    index = joblib.load(INDEX, mmap_mode="r")
    mapping = {item: pos for pos, item in enumerate(index.item_ids)}
    for query in queries:
        query["stage"] = stage
        query["relevant_rows"] = sorted(mapping[item] for item in query["relevant"])
    from .semantic_embeddings import encode_queries
    encode_queries([query["search_query"] for query in queries])
    build_item_vectors()
    module_dir = Path(__file__).parent
    code = {name: sha256_file(module_dir / name) for name in
            ("semantic_evaluate.py", "semantic_search.py", "semantic_embeddings.py",
             "semantic_benchmark.py", "ranker.py", "retrieval.py", "data.py")}
    signature = digest({"stage": stage, "limit": limit, "active": active, "models":
                       {name: sha256_file(path) for name, path in {"A": OLD_MODEL, **MODEL_FILES}.items() if name in active},
                       "code": code, "sources": source_manifest(Path("src/avito_test")), "vector_state":
                       json.loads(Path("artifacts/semantic/embeddings/encode_state.json").read_text(encoding="utf-8"))["vector_sha256"]})
    path = OUT / stage / ("full" if not limit else f"slice_{limit}") / signature[:16]
    parts = path / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    chunks = [queries[i:i+CHUNK] for i in range(0, len(queries), CHUNK)]
    pending = [(number, chunk) for number, chunk in enumerate(chunks)
               if not ((parts / f"done_{number:05d}.json").exists() and
                       (parts / f"metrics_{number:05d}.parquet").exists() and
                       (parts / f"positives_{number:05d}.parquet").exists())]
    start = time.perf_counter()
    peak = 0
    search_seconds = 0.
    print(f"Semantic {stage}: {len(queries)} queries; {len(chunks)-len(pending)} cached chunks", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                             initargs=([query["search_query"] for query in queries], active)) as pool:
        futures = [pool.submit(score_chunk, number, chunk, str(parts)) for number, chunk in pending]
        completed = (len(chunks) - len(pending)) * CHUNK
        for future in as_completed(futures):
            _, count, memory, seconds = future.result()
            completed += count
            peak = max(peak, memory)
            search_seconds += seconds
            if completed % (CHUNK * 10) < CHUNK:
                print(f"Semantic {stage}: {min(completed,len(queries))}/{len(queries)}; {time.perf_counter()-start:.1f}s", flush=True)
    frame = pl.read_parquet(str(parts / "metrics_*.parquet"))
    expected = len(queries) * 2 * len(active)
    if frame.height != expected or frame.unique(["group_id", "scheme", "variant"]).height != expected:
        raise ValueError(f"Incomplete semantic predictions: {frame.height} != {expected}")
    frame.sort("group_id", "scheme", "variant").write_parquet(path / "per_query.parquet")
    positives = pl.read_parquet(str(parts / "positives_*.parquet"))
    expected_positives = 2 * sum(len(query["relevant_rows"]) for query in queries)
    if positives.height != expected_positives:
        raise ValueError(f"Incomplete positive diagnostics: {positives.height} != {expected_positives}")
    positives.sort("group_id", "scheme", "item_row").write_parquet(path / "positive_ranks.parquet")
    summary = frame.group_by("stage", "scheme", "variant").agg(
        pl.len().alias("queries"), pl.col("recall@50").mean(),
        pl.col("candidate_recall").mean(), pl.col("lost_at_selection").sum(),
        pl.col("same_location_recall@50").mean(),
        pl.col("other_location_recall@50").mean(),
        pl.col("new_semantic_positives").sum(),
        pl.col("new_semantic_positives_in_top50").sum(),
        pl.col("semantic_global_recall50").mean(),
        pl.col("semantic_global_recall200").mean(),
        pl.col("semantic_local_recall50").mean(),
        pl.col("semantic_local_recall200").mean(),
        pl.col("semantic_union_recall200").mean(),
    ).sort("scheme", "variant")
    summary.write_csv(path / "summary.csv")
    original = pl.read_parquet(Path("artifacts/ranker/full/c392362a99e2c4d1") / stage / "per_query.parquet")
    original = original.filter(pl.col("variant") == "catboost_full_no_history_features").select(
        "group_id", "scheme", pl.col("recall@50").alias("old_recall"),
        pl.col("candidate_recall").alias("old_pool_recall"))
    control = frame.filter(pl.col("variant") == "A").join(original, on=["group_id", "scheme"])
    if (control.height != len(queries) * 2 or
            not np.allclose(control["recall@50"].to_numpy(), control["old_recall"].to_numpy(), atol=1e-8) or
            not np.allclose(control["candidate_recall"].to_numpy(), control["old_pool_recall"].to_numpy(), atol=1e-8)):
        raise ValueError("Control A does not reproduce archived stage predictions")
    report = {"stage": stage, "query_count": len(queries), "seconds": time.perf_counter()-start,
              "dense_search_seconds_sum": search_seconds, "worker_peak_rss_bytes": peak,
              "workers": workers, "control_A_reproduced": True,
              "per_query": str(path / "per_query.parquet"), "summary": str(path / "summary.csv")}
    (path / "timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary, flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    evaluate(stage=args.stage, limit=args.limit, workers=args.workers)
