"""Frozen-pool geography diagnostics and small off-location quotas."""
# Treat a location mismatch as evidence, not a reason to discard an item.
from __future__ import annotations

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

from .data import digest, sha256_file, source_manifest, write_json
from .extended_validation import prepare_union, rank_in_scope
from .ranker import CHUNK_SIZE, FEATURES, TEXT_FEATURES, FeatureRetriever, group_queries, load_feature_items, pool_ids

QUOTAS = (0, 2, 5, 10)
GLOBAL_DEPTH = 100
_WORKER = None


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def location_kind(item_location: object, query_location: object) -> str:
    if _missing(item_location) or _missing(query_location):
        return "unknown"
    return "same" if item_location == query_location else "other"


def quota_ranking(order: np.ndarray, locations: np.ndarray, query_location: object,
                  minimum_other: int) -> np.ndarray:
    """Keep 50; promote best missing off-location items over weakest local items."""
    base = [int(item) for item in order[:50]]
    if minimum_other <= 0 or not base:
        return np.array(base, dtype=np.int32)
    kinds = {item: location_kind(locations[item], query_location) for item in base}
    deficit = max(0, minimum_other - sum(kind == "other" for kind in kinds.values()))
    if deficit == 0:
        return np.array(base, dtype=np.int32)
    promotions = [int(item) for item in order[50:] if
                  location_kind(locations[int(item)], query_location) == "other"][:deficit]
    replaceable = [item for item in base[::-1] if kinds[item] == "same"][:len(promotions)]
    if not replaceable:
        return np.array(base, dtype=np.int32)
    chosen = (set(base) - set(replaceable)) | set(promotions[:len(replaceable)])
    result = np.array([int(item) for item in order if int(item) in chosen][:len(base)], dtype=np.int32)
    if len(result) != len(base) or len(set(map(int, result))) != len(result):
        raise ValueError("Quota changed output size or introduced a duplicate")
    return result


def location_counts(truth: np.ndarray, candidates: np.ndarray,
                    prediction: np.ndarray, locations: np.ndarray,
                    query_location: object) -> dict:
    pool = set(map(int, candidates))
    selected = set(map(int, prediction))
    result = {}
    for kind in ("same", "other", "unknown"):
        relevant = {int(item) for item in truth if
                    location_kind(locations[int(item)], query_location) == kind}
        result[f"{kind}_positives"] = len(relevant)
        result[f"{kind}_candidate_found"] = len(relevant & pool)
        result[f"{kind}_found"] = len(relevant & selected)
    result["positives"] = len(truth)
    result["candidate_found"] = sum(result[f"{kind}_candidate_found"] for kind in
                                    ("same", "other", "unknown"))
    result["found"] = sum(result[f"{kind}_found"] for kind in ("same", "other", "unknown"))
    result["recall@50"] = result["found"] / len(truth)
    result["candidate_recall"] = result["candidate_found"] / len(truth)
    result["selected_other"] = sum(location_kind(locations[int(item)], query_location) == "other"
                                   for item in prediction)
    result["selected_unknown"] = sum(location_kind(locations[int(item)], query_location) == "unknown"
                                     for item in prediction)
    result["candidate_count"] = len(pool)
    return result


def init_worker(index_path: str, metadata_path: str, feature_path: str,
                signals_path: str, model_path: str) -> None:
    global _WORKER
    threadpool_limits(limits=1)
    index = joblib.load(index_path, mmap_mode="r")
    metadata = pl.read_parquet(metadata_path)
    feature_items = pl.read_parquet(feature_path)
    signals = joblib.load(signals_path, mmap_mode="r")
    retriever = FeatureRetriever(index, signals, metadata["in_benchmark"].to_numpy(), feature_items)
    model = CatBoostRanker()
    model.load_model(model_path)
    _WORKER = retriever, model


def score_chunk(number: int, queries: list[dict], parts: str,
                schemes: tuple[str, ...]) -> tuple[int, int, int]:
    retriever, model = _WORKER
    locations = retriever.signals.locations
    text_columns = [FEATURES.index(name) for name in TEXT_FEATURES]
    metrics, positives, probes = [], [], []
    for query in queries:
        truth = np.array(query["relevant_rows"], dtype=np.int32)
        for scheme in schemes:
            augmented = scheme == "benchmark_plus_positives"
            results, scores, allowed = retriever.retrieve_with_scores(query, truth, augmented)
            pool = pool_ids(results)
            x = retriever.features(query, pool, results, scores, allowed)
            values = model.predict(x[:, text_columns], thread_count=1)
            order = pool[np.lexsort((pool, -values))]
            bm25_global = rank_in_scope(scores["bm25"], allowed, GLOBAL_DEPTH)
            char_global = rank_in_scope(scores["char"], allowed, GLOBAL_DEPTH)
            extended = np.unique(np.concatenate((pool, bm25_global, char_global))).astype(np.int32)
            pool_set, extended_set = set(map(int, pool)), set(map(int, extended))
            query_location = query["search_location_id"]
            for quota in QUOTAS:
                ranking = quota_ranking(order, locations, query_location, quota)
                if len(ranking) != 50:
                    raise ValueError("A query returned fewer than 50 items")
                metrics.append({"group_id": query["group_id"], "text": query["text"],
                                "stage": query["stage"], "scheme": scheme, "quota": quota,
                                **location_counts(truth, pool, ranking, locations, query_location)})
            positive_kinds = {kind: [int(item) for item in truth if
                               location_kind(locations[int(item)], query_location) == kind]
                              for kind in ("same", "other", "unknown")}
            probe = {"group_id": query["group_id"], "text": query["text"],
                     "stage": query["stage"], "scheme": scheme,
                     "pool_count": len(pool), "extended_count": len(extended)}
            for kind, items in positive_kinds.items():
                probe[f"{kind}_positives"] = len(items)
                probe[f"{kind}_pool_found"] = sum(item in pool_set for item in items)
                probe[f"{kind}_extended_found"] = sum(item in extended_set for item in items)
            probes.append(probe)
            positions = {int(item): rank for rank, item in enumerate(order, 1)}
            for item in truth:
                item = int(item)
                positives.append({"group_id": query["group_id"], "stage": query["stage"],
                                  "scheme": scheme, "item_id": retriever.index.item_ids[item],
                                  "location_kind": location_kind(locations[item], query_location),
                                  "catboost_rank": positions.get(item, 0),
                                  "in_current_pool": item in pool_set,
                                  "in_extended_pool": item in extended_set,
                                  "new_from_global": item in extended_set and item not in pool_set})
    destination = Path(parts)
    pl.DataFrame(metrics).write_parquet(destination / f"metrics_{number:05d}.parquet")
    pl.DataFrame(positives).write_parquet(destination / f"positives_{number:05d}.parquet")
    pl.DataFrame(probes).write_parquet(destination / f"probes_{number:05d}.parquet")
    write_json(destination / f"done_{number:05d}.json", {"queries": len(queries)})
    memory = psutil.Process().memory_info()
    return number, len(queries), getattr(memory, "peak_wset", memory.rss)


def diagnose(data_dir: Path, output: Path, train: pl.DataFrame, stage: str = "dev",
             workers: int = 4, limit: int = 0,
             schemes: tuple[str, ...] = ("union", "benchmark_plus_positives"),
             ranker_run: Path | None = None) -> Path:
    if workers < 1 or stage not in ("dev", "test") or limit < 0:
        raise ValueError("Invalid geography evaluation arguments")
    sources = source_manifest(data_dir)
    index_path, metadata_path, _ = prepare_union(data_dir, output / "expanded", sources)
    index = joblib.load(index_path, mmap_mode="r")
    metadata = pl.read_parquet(metadata_path)
    load_feature_items(data_dir, index_path.parent, index.item_ids, sources)
    run = ranker_run or Path(json.loads((output / "ranker" / "latest.json").read_text(
        encoding="utf-8"))["run"])
    signals_path, model_path = run / "full_train_signals.joblib", run / "text_model.cbm"
    if sha256_file(model_path) != json.loads((run / "models.json").read_text(encoding="utf-8"))["model_sha256"]["text"]:
        raise ValueError("Frozen CatBoost model changed")
    code = {filename: sha256_file(Path(__file__).with_name(filename)) for filename in
            ("geography.py", "ranker.py", "retrieval.py", "extended_validation.py", "data.py")}
    signature = digest({"sources": sources, "code": code, "model": sha256_file(model_path),
                        "signals": sha256_file(signals_path), "stage": stage, "limit": limit,
                        "schemes": schemes, "quotas": QUOTAS, "global_depth": GLOBAL_DEPTH})
    result = output / "geography" / ("smoke" if limit else "full") / signature[:16] / stage
    result.mkdir(parents=True, exist_ok=True)
    mapping = {item: number for number, item in enumerate(metadata["item_id"])}
    queries = group_queries(train, stage, limit)
    for query in queries:
        query["relevant_rows"] = [mapping[item] for item in query["relevant"]]
        query["stage"] = stage
    chunks = [queries[i:i + CHUNK_SIZE] for i in range(0, len(queries), CHUNK_SIZE)]
    parts = result / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    pending = [(number, chunk) for number, chunk in enumerate(chunks) if not all(
        (parts / f"{prefix}_{number:05d}.{extension}").exists() for prefix, extension in
        (("done", "json"), ("metrics", "parquet"), ("positives", "parquet"), ("probes", "parquet")))]
    start, peak = time.perf_counter(), 0
    completed = len(queries) - sum(len(chunk) for _, chunk in pending)
    print(f"Geography {stage}: {len(queries)} queries, {completed} cached, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(str(index_path), str(metadata_path),
                                       str(index_path.parent / "features.parquet"),
                                       str(signals_path), str(model_path))) as pool:
        futures = [pool.submit(score_chunk, number, chunk, str(parts), schemes)
                   for number, chunk in pending]
        for future in as_completed(futures):
            _, count, memory = future.result()
            completed += count
            peak = max(peak, memory)
            if completed % (CHUNK_SIZE * 12) < CHUNK_SIZE:
                print(f"Geography {stage}: {completed}/{len(queries)}, "
                      f"{time.perf_counter()-start:.1f}s", flush=True)
    for name, expected in (("metrics", len(queries) * len(schemes) * len(QUOTAS)),
                           ("probes", len(queries) * len(schemes))):
        frame = pl.read_parquet(str(parts / f"{name}_*.parquet"))
        if frame.height != expected:
            raise ValueError(f"Incomplete geography {name}: {frame.height} != {expected}")
        frame.write_parquet(result / f"{name}.parquet")
    positive_frame = pl.read_parquet(str(parts / "positives_*.parquet"))
    positive_frame.write_parquet(result / "positives.parquet")
    write_json(result / "run.json", {"signature": signature, "stage": stage,
               "queries": len(queries), "schemes": schemes, "quotas": QUOTAS,
               "global_depth": GLOBAL_DEPTH, "elapsed_seconds": time.perf_counter()-start,
               "peak_worker_rss_bytes": peak, "workers": workers,
               "cached_queries": len(queries) - sum(len(chunk) for _, chunk in pending),
               "source_manifest": sources, "code_sha256": code})
    write_json(output / "geography" / "latest.json", {"run": str(result.parent), "signature": signature})
    return result
