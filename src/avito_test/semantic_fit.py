"""Controlled CatBoost ablations with a frozen E5 encoder."""
# Hold the E5 encoder fixed while comparing ranker feature and pool changes.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import psutil
from catboost import CatBoostRanker, Pool

from .data import sha256_file
from .ranker import FEATURES, MODEL_PARAMS, TEXT_FEATURES
from .semantic_benchmark import clean_text
from .semantic_embeddings import OUT as EMBEDDINGS, encode_queries
from .semantic_oof import OOF, OUT as REPLAY
from .semantic_search import build_item_vectors

OUT = Path("artifacts/semantic/models")
BASE_SEMANTIC_FEATURES = ["e5_cosine", "e5_rank_in_pool", "e5_gap"]
SOURCE_SEMANTIC_FEATURES = ["e5_global_rank", "e5_global_missing",
                            "e5_local_rank", "e5_local_missing"]


def fit(name: str, matrix: np.ndarray, labels: np.ndarray,
        groups: np.ndarray, feature_names: list[str]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.cbm"
    start = time.perf_counter()
    model = CatBoostRanker(**MODEL_PARAMS)
    model.fit(Pool(matrix, label=labels, group_id=groups,
                   feature_names=feature_names))
    model.save_model(str(path))
    report = {"variant": name, "model_path": str(path), "model_sha256": sha256_file(path),
              "feature_names": feature_names, "params": MODEL_PARAMS,
              "sampled_rows": len(labels), "sampled_groups": int(groups.max()) + 1,
              "positive_rows": int(labels.sum()), "train_seconds": time.perf_counter() - start,
              "process_peak_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset",
                                                 psutil.Process().memory_info().rss)}
    (OUT / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def prepare_b() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (REPLAY / "manifest.json").exists():
        raise ValueError("Replay the original OOF sample before training B")
    OUT.mkdir(parents=True, exist_ok=True)
    item_vectors = np.load(build_item_vectors(EMBEDDINGS), mmap_mode="r")
    selected = np.load(REPLAY / "selected_rows.npy", mmap_mode="r")
    pool = np.load(REPLAY / "pool_rows.npy", mmap_mode="r")
    offsets = np.load(REPLAY / "pool_offsets.npy", mmap_mode="r")
    queries = pl.read_parquet(REPLAY / "used_queries.parquet")
    if len(offsets) != queries.height + 1:
        raise ValueError("OOF pool offsets do not align with used queries")
    embeddings = encode_queries(queries["search_query"].to_list(), out=EMBEDDINGS)
    with np.load(OOF) as data:
        original_x, labels, groups = data["features"], data["labels"], data["group_ids"]
    columns = [FEATURES.index(name) for name in TEXT_FEATURES]
    semantic = np.empty((len(labels), 3), dtype=np.float32)
    row = 0
    start = time.perf_counter()
    for group, query in enumerate(queries.iter_rows(named=True)):
        n = query["rows"]
        ids = selected[row:row+n]
        candidates = pool[offsets[group]:offsets[group+1]]
        positions = np.searchsorted(candidates, ids)
        if np.any(positions == len(candidates)) or not np.array_equal(candidates[positions], ids):
            raise ValueError("OOF sampled row is absent from its original pool")
        q = embeddings[clean_text(query["search_query"])]
        similarities = item_vectors[candidates] @ q
        order = np.lexsort((candidates, -similarities))
        ranks = np.empty(len(candidates), dtype=np.float32)
        ranks[order] = np.arange(1, len(candidates) + 1, dtype=np.float32)
        semantic[row:row+n, 0] = similarities[positions]
        semantic[row:row+n, 1] = ranks[positions]
        semantic[row:row+n, 2] = np.max(similarities) - similarities[positions]
        row += n
        if (group + 1) % 500 == 0:
            print(f"B E5 features: {group+1}/{queries.height}; {time.perf_counter()-start:.1f}s", flush=True)
    if row != len(labels):
        raise ValueError("OOF B row count changed")
    matrix = np.column_stack((original_x[:, columns], semantic)).astype(np.float32)
    np.savez_compressed(OUT / "B_oof.npz", features=matrix, labels=labels, group_ids=groups)
    return matrix, labels, groups


def train_b() -> dict:
    matrix, labels, groups = prepare_b()
    return fit("B_existing_pool_semantic_features", matrix, labels, groups,
               TEXT_FEATURES + BASE_SEMANTIC_FEATURES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["B"])
    args = parser.parse_args()
    if args.variant == "B":
        train_b()
