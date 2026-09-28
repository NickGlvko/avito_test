"""Replay the exact control OOF sample and retain its candidate identity and pool."""
# Replay the same candidate rows when isolating an E5 feature change.
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from .data import SEED, enrich_train, sha256_file, source_manifest
from .ranker import (NEGATIVES_PER_QUERY, FeatureRetriever, load_feature_items,
                     oof_queries, pool_ids, sample_training_rows, training_folds)
from .retrieval import TrainSignals

INDEX = Path("artifacts/expanded/cache/73915bb3bc082213/union_index.joblib")
OOF = Path("artifacts/ranker/full/c392362a99e2c4d1/oof_1000.npz")
OUT = Path("artifacts/semantic/oof_replay")


def replay() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    index = joblib.load(INDEX, mmap_mode="r")
    mapping = {item: i for i, item in enumerate(index.item_ids)}
    items = load_feature_items(Path("src/avito_test"), INDEX.parent, index.item_ids,
                               source_manifest(Path("src/avito_test")))
    train = enrich_train(pl.read_parquet("src/avito_test/train.parquet"))
    training = training_folds(train.filter(pl.col("split") < 8))
    queries = oof_queries(training, 1000)
    pl.DataFrame(queries).write_parquet(OUT / "all_queries.parquet")
    with np.load(OOF) as data:
        old_x, old_y, old_groups = data["features"], data["labels"], data["group_ids"]
    selected_parts, pool_parts, offsets = [], [], [0]
    used = []
    start = time.perf_counter()
    row_offset = 0
    for fold in range(3):
        history_rows = training.filter(pl.col("fold") != fold).drop("fold")
        retriever = FeatureRetriever(index, TrainSignals(history_rows, metadata),
                                     metadata["in_benchmark"].to_numpy(), items)
        fold_queries = [q for q in queries if q["fold"] == fold]
        for number, query in enumerate(fold_queries):
            truth = {mapping[item] for item in query["relevant"]}
            relevant = np.array(sorted(truth), dtype=np.int32)
            results, scores, allowed = retriever.retrieve_with_scores(query, relevant, False)
            rng = np.random.default_rng(int(query["group_id"][:16], 16) ^ SEED)
            selected, positives = sample_training_rows(results, truth, NEGATIVES_PER_QUERY, rng)
            if not positives or len(selected) == positives:
                continue
            n = len(selected)
            if not np.array_equal(old_groups[row_offset:row_offset+n], np.full(n, len(used), dtype=np.int32)):
                raise ValueError(f"OOF group order changed at {query['group_id']}")
            target = np.array([int(int(item) in truth) for item in selected], dtype=np.uint8)
            if not np.array_equal(old_y[row_offset:row_offset+n], target):
                raise ValueError(f"OOF labels changed at {query['group_id']}")
            original_features = retriever.features(query, selected, results, scores, allowed)
            if not np.array_equal(old_x[row_offset:row_offset+n], original_features):
                difference = np.max(np.abs(old_x[row_offset:row_offset+n] - original_features))
                raise ValueError(f"OOF features changed at {query['group_id']}: {difference}")
            pool = pool_ids(results)
            selected_parts.append(selected)
            pool_parts.append(pool)
            offsets.append(offsets[-1] + len(pool))
            used.append({"group_id": query["group_id"], "search_query": query["search_query"],
                         "fold": fold, "rows": n, "positives": positives})
            row_offset += n
            if (number + 1) % 200 == 0:
                print(f"Replay fold {fold}: {number+1}/{len(fold_queries)}; {time.perf_counter()-start:.1f}s", flush=True)
        del retriever, history_rows
        gc.collect()
    if row_offset != len(old_y):
        raise ValueError(f"OOF replay ended at {row_offset}, expected {len(old_y)}")
    np.save(OUT / "selected_rows.npy", np.concatenate(selected_parts))
    np.save(OUT / "pool_rows.npy", np.concatenate(pool_parts))
    np.save(OUT / "pool_offsets.npy", np.asarray(offsets, dtype=np.int64))
    pl.DataFrame(used).write_parquet(OUT / "used_queries.parquet")
    report = {"queries_requested": len(queries), "queries_used": len(used),
              "sampled_rows": row_offset, "pool_rows": offsets[-1],
              "positive_rows": int(old_y.sum()), "control_oof_sha256": sha256_file(OOF),
              "exact_features_labels_and_groups": True,
              "elapsed_seconds": time.perf_counter()-start}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    replay()
