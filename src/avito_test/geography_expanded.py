"""OOF training and evaluation after adding two global text retrieval channels."""
# Rebuild OOF training examples when the candidate sources change.
from __future__ import annotations

import gc
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import psutil
from catboost import CatBoostRanker, Pool
from threadpoolctl import threadpool_limits

from .data import SEED, digest, sha256_file, source_manifest, write_json
from .extended_validation import prepare_union, rank_in_scope
from .geography import GLOBAL_DEPTH, location_counts, location_kind
from .ranker import (CHUNK_SIZE, FEATURES, MODEL_PARAMS, NEGATIVES_PER_QUERY, POOL_SOURCES,
                     POOL_RRF, TEXT_FEATURES, FeatureRetriever, group_queries,
                     load_feature_items, oof_queries, pool_ids, training_folds)
from .retrieval import TrainSignals, fuse


def expanded_pool(results: dict, scores: dict, allowed: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_bm25 = rank_in_scope(scores["bm25"], allowed, GLOBAL_DEPTH)
    global_char = rank_in_scope(scores["char"], allowed, GLOBAL_DEPTH)
    candidates = np.unique(np.concatenate([pool_ids(results), global_bm25, global_char])).astype(np.int32)
    return candidates, global_bm25, global_char


def sample_extended(results: dict, pool: np.ndarray, global_bm25: np.ndarray,
                    global_char: np.ndarray, truth: set[int], rng: np.random.Generator,
                    limit: int = NEGATIVES_PER_QUERY) -> tuple[np.ndarray, int]:
    positives = truth & set(map(int, pool))
    if not positives:
        return np.empty(0, dtype=np.int32), 0
    chosen: list[int] = []
    seen = set(positives)

    def take(items, count):
        added = 0
        for item in items:
            item = int(item)
            if item not in seen:
                chosen.append(item)
                seen.add(item)
                added += 1
                if added >= count or len(chosen) >= limit:
                    break

    rrf = fuse(results, POOL_RRF, depth=len(pool))
    take(rrf, 60)
    take(global_bm25, 25)
    take(global_char, 25)
    membership = Counter(int(item) for name in POOL_SOURCES for item in results[name])
    singles = [int(item) for item in rrf if membership[int(item)] == 1]
    take(singles, 20)
    remaining = [int(item) for item in pool if int(item) not in seen]
    if remaining and len(chosen) < limit:
        take(rng.permutation(remaining), limit - len(chosen))
    return np.array(sorted(positives | set(chosen)), dtype=np.int32), len(positives)


def train(data_dir: Path, output: Path, training: pl.DataFrame,
          per_fold: int = 1000) -> Path:
    sources = source_manifest(data_dir)
    index_path, metadata_path, _ = prepare_union(data_dir, output / "expanded", sources)
    index = joblib.load(index_path, mmap_mode="r")
    items = pl.read_parquet(metadata_path)
    feature_items = load_feature_items(data_dir, index_path.parent, index.item_ids, sources)
    signature = digest({"sources": sources, "code": sha256_file(Path(__file__)),
                        "ranker_code": sha256_file(Path(__file__).with_name("ranker.py")),
                        "folds": per_fold, "seed": SEED, "global_depth": GLOBAL_DEPTH,
                        "params": MODEL_PARAMS, "negatives": NEGATIVES_PER_QUERY})
    run = output / "geography" / "expanded_model" / signature[:16]
    run.mkdir(parents=True, exist_ok=True)
    model_path = run / "text_model.cbm"
    if model_path.exists() and (run / "training.json").exists():
        saved = json.loads((run / "training.json").read_text(encoding="utf-8"))
        if saved["model_sha256"] == sha256_file(model_path):
            return run
    folded = training_folds(training.filter(pl.col("split") < 8))
    queries = oof_queries(folded, per_fold)
    mapping = {item: i for i, item in enumerate(index.item_ids)}
    mask = items["in_benchmark"].to_numpy()
    batches, labels, groups = [], [], []
    audit = {"requested_queries": len(queries), "used_queries": 0,
             "no_positive_in_pool": 0, "known_positives": 0,
             "base_found": 0, "expanded_found": 0, "global_new_other": 0,
             "history_rows_by_fold": {}, "sampling": "all found positives; 60 RRF, 25+25 global, 20 single-channel, random to 150 negatives"}
    started = time.perf_counter()
    for fold in range(3):
        rows = folded.filter(pl.col("fold") != fold).drop("fold")
        fold_queries = [query for query in queries if query["fold"] == fold]
        if {query["group_id"] for query in fold_queries} & set(rows["group_id"]):
            raise ValueError("OOF history includes a heldout query")
        signals = TrainSignals(rows, items)
        retriever = FeatureRetriever(index, signals, mask, feature_items)
        audit["history_rows_by_fold"][str(fold)] = rows.height
        for number, query in enumerate(fold_queries, 1):
            truth = {mapping[item] for item in query["relevant"]}
            positive_ids = np.array(sorted(truth), dtype=np.int32)
            results, scores, allowed = retriever.retrieve_with_scores(query, positive_ids, False)
            base = set(map(int, pool_ids(results)))
            pool, bm25_global, char_global = expanded_pool(results, scores, allowed)
            full = set(map(int, pool))
            audit["known_positives"] += len(truth)
            audit["base_found"] += len(truth & base)
            audit["expanded_found"] += len(truth & full)
            audit["global_new_other"] += sum(
                location_kind(signals.locations[item], query["search_location_id"]) == "other"
                for item in truth & (full - base))
            rng = np.random.default_rng(int(query["group_id"][:16], 16) ^ SEED)
            selected, found = sample_extended(results, pool, bm25_global, char_global, truth, rng)
            if not found:
                audit["no_positive_in_pool"] += 1
                continue
            if len(selected) == found:
                continue
            x = retriever.features(query, selected, results, scores, allowed)
            batches.append(x[:, [FEATURES.index(name) for name in TEXT_FEATURES]])
            labels.append(np.array([int(int(item) in truth) for item in selected], dtype=np.uint8))
            groups.append(np.full(len(selected), audit["used_queries"], dtype=np.int32))
            audit["used_queries"] += 1
            if number % 200 == 0:
                print(f"Geo OOF fold {fold}: {number}/{len(fold_queries)}, {time.perf_counter()-started:.1f}s", flush=True)
        del retriever, signals, rows
        gc.collect()
    x = np.concatenate(batches)
    y = np.concatenate(labels)
    group_id = np.concatenate(groups)
    np.savez_compressed(run / "oof.npz", features=x, labels=y, group_ids=group_id)
    model = CatBoostRanker(**MODEL_PARAMS)
    model.fit(Pool(x, label=y, group_id=group_id, feature_names=TEXT_FEATURES))
    model.save_model(str(model_path))
    audit.update({"signature": signature, "model_sha256": sha256_file(model_path),
                  "features": TEXT_FEATURES, "params": MODEL_PARAMS,
                  "sampled_rows": len(y), "sampled_positives": int(y.sum()),
                  "elapsed_seconds": time.perf_counter() - started,
                  "source_manifest": sources})
    write_json(run / "training.json", audit)
    write_json(output / "geography" / "expanded_model_latest.json", {"run": str(run)})
    print(f"Geo model ready: {run}, {audit['elapsed_seconds']:.1f}s", flush=True)
    return run


_WORKER = None


def init_eval(index_path: str, metadata_path: str, feature_path: str,
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


def eval_chunk(number: int, queries: list[dict], parts: str,
               schemes: tuple[str, ...]) -> tuple[int, int, int]:
    retriever, model = _WORKER
    locations = retriever.signals.locations
    text_columns = [FEATURES.index(name) for name in TEXT_FEATURES]
    rows = []
    for query in queries:
        truth = np.array(query["relevant_rows"], dtype=np.int32)
        for scheme in schemes:
            results, scores, allowed = retriever.retrieve_with_scores(
                query, truth, scheme == "benchmark_plus_positives")
            candidates, _, _ = expanded_pool(results, scores, allowed)
            x = retriever.features(query, candidates, results, scores, allowed)
            values = model.predict(x[:, text_columns], thread_count=1)
            order = candidates[np.lexsort((candidates, -values))][:50]
            rows.append({"group_id": query["group_id"], "text": query["text"],
                         "stage": query["stage"], "scheme": scheme,
                         **location_counts(truth, candidates, order, locations,
                                           query["search_location_id"])})
    destination = Path(parts)
    pl.DataFrame(rows).write_parquet(destination / f"metrics_{number:05d}.parquet")
    write_json(destination / f"done_{number:05d}.json", {"queries": len(queries)})
    memory = psutil.Process().memory_info()
    return number, len(queries), getattr(memory, "peak_wset", memory.rss)


def evaluate(data_dir: Path, output: Path, training: pl.DataFrame,
             model_run: Path, ranker_run: Path, stage: str = "dev",
             workers: int = 4, limit: int = 0,
             schemes: tuple[str, ...] = ("union", "benchmark_plus_positives")) -> Path:
    if stage not in ("dev", "test") or workers < 1 or limit < 0:
        raise ValueError("Invalid expanded geography evaluation arguments")
    sources = source_manifest(data_dir)
    index_path, metadata_path, _ = prepare_union(data_dir, output / "expanded", sources)
    index = joblib.load(index_path, mmap_mode="r")
    metadata = pl.read_parquet(metadata_path)
    load_feature_items(data_dir, index_path.parent, index.item_ids, sources)
    model_path = model_run / "text_model.cbm"
    if json.loads((model_run / "training.json").read_text(encoding="utf-8"))["model_sha256"] != sha256_file(model_path):
        raise ValueError("Expanded model changed")
    signals_path = ranker_run / "full_train_signals.joblib"
    signature = digest({"stage": stage, "schemes": schemes, "limit": limit,
                        "model": sha256_file(model_path), "signals": sha256_file(signals_path),
                        "sources": sources, "code": sha256_file(Path(__file__))})
    result = output / "geography" / "expanded_evaluation" / signature[:16] / stage
    parts = result / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    mapping = {item: i for i, item in enumerate(index.item_ids)}
    queries = group_queries(training, stage, limit)
    for query in queries:
        query["relevant_rows"] = [mapping[item] for item in query["relevant"]]
        query["stage"] = stage
    chunks = [queries[i:i + CHUNK_SIZE] for i in range(0, len(queries), CHUNK_SIZE)]
    pending = [(number, chunk) for number, chunk in enumerate(chunks) if not all(
        (parts / f"{prefix}_{number:05d}.{extension}").exists() for prefix, extension in
        (("done", "json"), ("metrics", "parquet")))]
    start, peak = time.perf_counter(), 0
    completed = len(queries) - sum(len(chunk) for _, chunk in pending)
    print(f"Geo expanded {stage}: {len(queries)} queries, {completed} cached, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_eval,
                             initargs=(str(index_path), str(metadata_path),
                                       str(index_path.parent / "features.parquet"),
                                       str(signals_path), str(model_path))) as pool:
        futures = [pool.submit(eval_chunk, number, chunk, str(parts), schemes)
                   for number, chunk in pending]
        for future in as_completed(futures):
            _, count, memory = future.result()
            completed += count
            peak = max(peak, memory)
            if completed % (CHUNK_SIZE * 12) < CHUNK_SIZE:
                print(f"Geo expanded {stage}: {completed}/{len(queries)}, "
                      f"{time.perf_counter()-start:.1f}s", flush=True)
    frame = pl.read_parquet(str(parts / "metrics_*.parquet"))
    if frame.height != len(queries) * len(schemes):
        raise ValueError("Incomplete expanded geography evaluation")
    frame.write_parquet(result / "metrics.parquet")
    write_json(result / "run.json", {"signature": signature, "model_run": str(model_run),
               "ranker_run": str(ranker_run), "stage": stage, "queries": len(queries),
               "elapsed_seconds": time.perf_counter()-start,
               "peak_worker_rss_bytes": peak, "workers": workers,
               "cached_queries": len(queries) - sum(len(chunk) for _, chunk in pending)})
    return result
