"""Leakage-free OOF training rows for the expanded lexical + E5 candidate pool."""
# Exclude a query fold from the interaction signals used for its features.
from __future__ import annotations

import gc
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from .data import SEED, enrich_train, source_manifest
from .ranker import (FeatureRetriever, NEGATIVES_PER_QUERY, POOL_RRF,
                     POOL_SOURCES, TEXT_FEATURES, load_feature_items, pool_ids,
                     training_folds)
from .retrieval import TrainSignals, fuse
from .semantic_fit import (BASE_SEMANTIC_FEATURES, OUT, SOURCE_SEMANTIC_FEATURES,
                           fit)
from .semantic_oof import INDEX, OUT as REPLAY
from .semantic_search import SemanticIndex, semantic_features

SEMANTIC_SOURCES = ("semantic_global", "semantic_local")


def sample_expanded(results: dict, truth: set[int], rng: np.random.Generator) -> tuple[np.ndarray, int]:
    names = (*POOL_SOURCES, *SEMANTIC_SOURCES)
    pool = np.unique(np.concatenate([results[name] for name in names])).astype(np.int32)
    positives = sorted(truth & set(map(int, pool)))
    if not positives:
        return np.empty(0, dtype=np.int32), 0
    positive = set(positives)
    base_rank = fuse(results, POOL_RRF, depth=len(pool))
    sem_rank = fuse(results, {name: 1. for name in SEMANTIC_SOURCES}, depth=len(pool))
    chosen: list[int] = []
    base_pool = set(map(int, pool_ids(results)))
    for item in base_rank:
        if int(item) not in positive:
            chosen.append(int(item))
        if len(chosen) == 70:
            break
    for item in sem_rank:
        value = int(item)
        if value not in base_pool and value not in positive and value not in chosen:
            chosen.append(value)
        if len(chosen) == 100:
            break
    memberships = Counter(int(item) for name in names for item in results[name])
    all_rank = fuse(results, {name: 1. for name in names}, depth=len(pool))
    for item in all_rank:
        value = int(item)
        if memberships[value] == 1 and value not in positive and value not in chosen:
            chosen.append(value)
        if len(chosen) == 120:
            break
    remaining = [int(item) for item in all_rank[::-1]
                 if int(item) not in positive and int(item) not in chosen]
    count = min(NEGATIVES_PER_QUERY - len(chosen), len(remaining))
    if count:
        chosen.extend(int(x) for x in rng.choice(remaining, size=count, replace=False))
    return np.array(sorted(positive | set(chosen)), dtype=np.int32), len(positives)


def prepare() -> dict:
    if not (REPLAY / "all_queries.parquet").exists():
        raise ValueError("Replay the original OOF queries first")
    queries = pl.read_parquet(REPLAY / "all_queries.parquet").to_dicts()
    semantic = SemanticIndex([q["search_query"] for q in queries])
    index = joblib.load(INDEX, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    mapping = {item: pos for pos, item in enumerate(index.item_ids)}
    items = load_feature_items(Path("src/avito_test"), INDEX.parent,
                               index.item_ids, source_manifest(Path("src/avito_test")))
    training = training_folds(enrich_train(pl.read_parquet("src/avito_test/train.parquet")).filter(pl.col("split") < 8))
    matrices, labels, group_ids = [], [], []
    audit = {"queries": len(queries), "groups_with_positive_in_pool": 0,
             "positive_rows": 0, "known_positives": 0, "base_found": 0,
             "expanded_found": 0, "pool_macro_recall_sum": 0.}
    start = time.perf_counter()
    for fold in range(3):
        history = training.filter(pl.col("fold") != fold).drop("fold")
        retriever = FeatureRetriever(index, TrainSignals(history, metadata),
                                     metadata["in_benchmark"].to_numpy(), items)
        fold_queries = [q for q in queries if q["fold"] == fold]
        for number, query in enumerate(fold_queries):
            truth = {mapping[item] for item in query["relevant"]}
            relevant = np.array(sorted(truth), dtype=np.int32)
            base_results, scores, allowed = retriever.retrieve_with_scores(query, relevant, False)
            dense = semantic.retrieve(query["search_query"], query["search_location_id"])
            results = {**base_results, **dense}
            base_pool = pool_ids(base_results)
            expanded_pool = np.unique(np.concatenate([base_pool, *dense.values()])).astype(np.int32)
            audit["known_positives"] += len(truth)
            audit["base_found"] += len(truth & set(map(int, base_pool)))
            audit["expanded_found"] += len(truth & set(map(int, expanded_pool)))
            audit["pool_macro_recall_sum"] += len(truth & set(map(int, expanded_pool))) / len(truth)
            rng = np.random.default_rng(int(query["group_id"][:16], 16) ^ SEED)
            selected, positives = sample_expanded(results, truth, rng)
            if not positives or len(selected) == positives:
                continue
            lexical = retriever.features(query, selected, base_results, scores, allowed)[:, :len(TEXT_FEATURES)]
            embedding = semantic_features(semantic, query, selected, results, True, expanded_pool)
            matrices.append(np.column_stack((lexical, embedding)).astype(np.float32))
            target = np.isin(selected, relevant).astype(np.uint8)
            labels.append(target)
            group_ids.append(np.full(len(selected), audit["groups_with_positive_in_pool"], dtype=np.int32))
            audit["groups_with_positive_in_pool"] += 1
            audit["positive_rows"] += int(target.sum())
            if (number + 1) % 200 == 0:
                print(f"Expanded OOF fold {fold}: {number+1}/{len(fold_queries)}; {time.perf_counter()-start:.1f}s", flush=True)
        del retriever, history
        gc.collect()
    matrix = np.concatenate(matrices)
    target = np.concatenate(labels)
    groups = np.concatenate(group_ids)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "expanded_oof.npz", features=matrix, labels=target, group_ids=groups)
    audit.update({"sampled_rows": len(target), "pool_macro_recall": audit["pool_macro_recall_sum"] / len(queries),
                  "search_seconds": semantic.search_seconds,
                  "elapsed_seconds": time.perf_counter()-start})
    (OUT / "expanded_oof.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return audit


def train() -> None:
    if not (OUT / "expanded_oof.npz").exists():
        prepare()
    with np.load(OUT / "expanded_oof.npz") as data:
        x, labels, groups = data["features"], data["labels"], data["group_ids"]
    fit("C_expanded_pool_no_semantic_features", x[:, :len(TEXT_FEATURES)], labels, groups,
        TEXT_FEATURES)
    fit("D_expanded_pool_semantic_features", x, labels, groups,
        TEXT_FEATURES + BASE_SEMANTIC_FEATURES + SOURCE_SEMANTIC_FEATURES)


if __name__ == "__main__":
    train()
