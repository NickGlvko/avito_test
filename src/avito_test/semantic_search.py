"""Exact E5 inner-product retrieval with the same corpus scopes as lexical search."""
# Filter the eligible corpus before dense top-k, then rank by exact inner product.
from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import polars as pl
from numpy.lib.format import open_memmap

from .semantic_benchmark import clean_text
from .semantic_embeddings import DIM, OUT, encode_queries

METADATA = Path("artifacts/expanded/cache/73915bb3bc082213/items.parquet")
DEPTH = 200


def build_item_vectors(out: Path = OUT) -> Path:
    """Materialize a row-aligned matrix from deduplicated normalized vectors."""
    path = out / "item_vectors.npy"
    state = json.loads((out / "encode_state.json").read_text(encoding="utf-8"))
    if not state.get("complete") or not state.get("vector_sha256"):
        raise ValueError("Item embeddings are not complete and checksummed")
    inverse = np.load(out / "item_to_text.npy", mmap_mode="r")
    unique = np.load(out / "unique_vectors.npy", mmap_mode="r")
    if path.exists():
        check = np.load(path, mmap_mode="r")
        if check.shape == (len(inverse), DIM):
            return path
        raise ValueError("Existing item matrix has the wrong shape")
    partial = out / "item_vectors.partial.npy"
    matrix = open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(inverse), DIM))
    for start in range(0, len(inverse), 8192):
        end = min(start + 8192, len(inverse))
        matrix[start:end] = unique[inverse[start:end]]
    matrix.flush()
    del matrix
    os.replace(partial, path)
    return path


class SemanticIndex:
    """Exact global/local search; augmented scope combines benchmark with current positives."""

    def __init__(self, query_texts: list[str], out: Path = OUT, threads: int = 4):
        self.out = out
        self.item_vectors = np.load(build_item_vectors(out), mmap_mode="r")
        metadata = pl.read_parquet(METADATA)
        self.benchmark_mask = metadata["in_benchmark"].to_numpy()
        self.benchmark_rows = np.flatnonzero(self.benchmark_mask).astype(np.int32)
        self.locations = metadata["item_location_id"].to_numpy()
        self.queries = encode_queries(query_texts, out=out)
        faiss.omp_set_num_threads(threads)
        self.indices: dict[str, tuple[faiss.IndexFlatIP, np.ndarray]] = {}
        self.local_indices: dict[tuple[str, object], tuple[faiss.IndexFlatIP, np.ndarray]] = {}
        self.search_seconds = 0.0

    def vector(self, raw_query: str) -> np.ndarray:
        return self.queries[clean_text(raw_query)]

    def _build_index(self, rows: np.ndarray) -> tuple[faiss.IndexFlatIP, np.ndarray]:
        index = faiss.IndexFlatIP(DIM)
        for start in range(0, len(rows), 8192):
            index.add(np.ascontiguousarray(self.item_vectors[rows[start:start + 8192]],
                                            dtype=np.float32))
        return index, rows

    def _scope_index(self, scope: str, location: object | None = None):
        if location is None:
            if scope not in self.indices:
                rows = np.arange(len(self.item_vectors), dtype=np.int32) if scope == "union" else self.benchmark_rows
                self.indices[scope] = self._build_index(rows)
            return self.indices[scope]
        key = (scope, location)
        if key not in self.local_indices:
            rows = np.flatnonzero(self.locations == location).astype(np.int32)
            if scope == "benchmark":
                rows = rows[self.benchmark_mask[rows]]
            self.local_indices[key] = self._build_index(rows)
        return self.local_indices[key]

    @staticmethod
    def _sorted(ids: np.ndarray, values: np.ndarray, depth: int) -> np.ndarray:
        order = np.lexsort((ids, -values))[:depth]
        return ids[order].astype(np.int32)

    @functools.lru_cache(maxsize=40000)
    def _base_search(self, raw_query: str, location: object, scope: str) -> tuple[np.ndarray, np.ndarray]:
        index, rows = self._scope_index(scope, location)
        if not len(rows):
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        q = np.ascontiguousarray(self.vector(raw_query).reshape(1, DIM), dtype=np.float32)
        scores, local = index.search(q, min(DEPTH, len(rows)))
        ids = rows[local[0]]
        values = scores[0]
        order = np.lexsort((ids, -values))
        return ids[order], values[order]

    def _merge_augmented(self, ids: np.ndarray, scores: np.ndarray,
                         query_vector: np.ndarray, relevant: np.ndarray,
                         location: object | None) -> np.ndarray:
        outside = relevant[~self.benchmark_mask[relevant]]
        if location is not None:
            outside = outside[self.locations[outside] == location]
        if not len(outside):
            return ids
        extra = self.item_vectors[outside] @ query_vector
        combined_ids = np.concatenate((ids, outside))
        combined_scores = np.concatenate((scores, extra))
        return self._sorted(combined_ids, combined_scores, DEPTH)

    def retrieve(self, raw_query: str, location: object,
                 relevant: np.ndarray | None = None,
                 augmented: bool = False) -> dict[str, np.ndarray]:
        start = time.perf_counter()
        scope = "benchmark" if augmented else "union"
        global_ids, global_scores = self._base_search(raw_query, None, scope)
        if location is None:
            local_ids = np.empty(0, dtype=np.int32)
            local_scores = np.empty(0, dtype=np.float32)
        else:
            local_ids, local_scores = self._base_search(raw_query, location, scope)
        if augmented:
            if relevant is None:
                raise ValueError("Augmented corpus requires current positive rows")
            q = self.vector(raw_query)
            global_ids = self._merge_augmented(global_ids, global_scores, q, relevant, None)
            if location is not None:
                local_ids = self._merge_augmented(local_ids, local_scores, q, relevant, location)
        self.search_seconds += time.perf_counter() - start
        return {"semantic_global": global_ids, "semantic_local": local_ids}

    def similarity(self, raw_query: str, ids: np.ndarray) -> np.ndarray:
        return (self.item_vectors[ids] @ self.vector(raw_query)).astype(np.float32)


def semantic_features(index: SemanticIndex, query: dict, ids: np.ndarray,
                      results: dict, extended: bool,
                      reference_ids: np.ndarray | None = None) -> np.ndarray:
    """Cosine, within-pool rank/gap; optionally source positions for expanded pool."""
    reference = ids if reference_ids is None else reference_ids
    if not np.array_equal(reference, np.sort(reference)):
        raise ValueError("Reference candidate rows must be sorted")
    positions = np.searchsorted(reference, ids)
    if np.any(positions == len(reference)) or not np.array_equal(reference[positions], ids):
        raise ValueError("Selected rows must be a subset of the reference pool")
    reference_similarities = index.similarity(query["search_query"], reference)
    order = np.lexsort((reference, -reference_similarities))
    all_ranks = np.empty(len(reference), dtype=np.float32)
    all_ranks[order] = np.arange(1, len(reference) + 1, dtype=np.float32)
    similarities = reference_similarities[positions]
    output = [similarities, all_ranks[positions],
              np.max(reference_similarities) - similarities]
    if extended:
        for name in ("semantic_global", "semantic_local"):
            positions = {int(item): pos for pos, item in enumerate(results[name], start=1)}
            source_ranks = np.fromiter((positions.get(int(item), 0) for item in ids),
                                       dtype=np.float32, count=len(ids))
            output += [source_ranks, (source_ranks == 0).astype(np.float32)]
    return np.column_stack(output).astype(np.float32)
