"""Cache complete expanded D candidate pools and their pool-dependent features."""
# Recompute rank and gap features on the expanded D pool itself.
from __future__ import annotations

import argparse
import json
import os
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
from .ranker import FeatureRetriever, TEXT_FEATURES, group_queries, pool_ids
from .semantic_embeddings import OUT as EMBEDDINGS, encode_queries
from .semantic_fit import BASE_SEMANTIC_FEATURES, SOURCE_SEMANTIC_FEATURES
from .semantic_full_d_oof import OUT
from .semantic_scale_evaluate import INDEX, MODELS, SIGNALS
from .semantic_search import SemanticIndex, build_item_vectors, semantic_features


CHUNK = 64
DIM = len(TEXT_FEATURES) + len(BASE_SEMANTIC_FEATURES) + len(SOURCE_SEMANTIC_FEATURES)
_STATE = None


def cache_signature(stage: str, queries: list[dict]) -> str:
    state = json.loads((EMBEDDINGS / "encode_state.json").read_text(encoding="utf-8"))
    code = {name: sha256_file(Path(__file__).with_name(name)) for name in
            ("semantic_full_d_cache.py", "semantic_search.py", "ranker.py",
             "retrieval.py", "data.py")}
    return digest({"stage": stage, "chunk": CHUNK,
                   "group_ids": [query["group_id"] for query in queries],
                   "code": code, "sources": source_manifest(Path("src/avito_test")),
                   "index_sha256": sha256_file(INDEX),
                   "signals_sha256": sha256_file(SIGNALS),
                   "vectors_sha256": state["vector_sha256"],
                   "D15000_sha256": sha256_file(MODELS["D15000"])})[:16]


def initialize(query_texts: list[str]) -> None:
    global _STATE
    threadpool_limits(limits=1)
    index = joblib.load(INDEX, mmap_mode="r")
    signals = joblib.load(SIGNALS, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    items = pl.read_parquet(INDEX.parent / "features.parquet")
    retriever = FeatureRetriever(index, signals, metadata["in_benchmark"].to_numpy(), items)
    semantic = SemanticIndex(query_texts, threads=1)
    model = CatBoostRanker()
    model.load_model(str(MODELS["D15000"]))
    _STATE = retriever, semantic, model


def cache_chunk(number: int, queries: list[dict], directory: str) -> tuple[int, int, int]:
    retriever, semantic, model = _STATE
    feature_parts, item_parts, old_parts, d15_parts = [], [], [], []
    offsets = [0]
    for query in queries:
        relevant = np.asarray(query["relevant_rows"], dtype=np.int32)
        for augmented in (False, True):
            base, scores, allowed = retriever.retrieve_with_scores(
                query, relevant, augmented)
            b_ids = pool_ids(base)
            dense = semantic.retrieve(
                query["search_query"], query["search_location_id"],
                relevant, augmented)
            sources = {**base, **dense}
            ids = np.unique(np.concatenate((b_ids, *dense.values()))).astype(np.int32)
            lexical = retriever.features(query, ids, base, scores, allowed)
            embedding = semantic_features(semantic, query, ids, sources, True, ids)
            x = np.column_stack((lexical[:, :len(TEXT_FEATURES)], embedding)).astype(np.float32)
            if x.shape != (len(ids), DIM) or len(ids) < 50:
                raise ValueError("Expanded D pool or feature order changed")
            feature_parts.append(x)
            item_parts.append(ids)
            old_parts.append(np.isin(ids, b_ids))
            d15_parts.append(np.asarray(model.predict(x, thread_count=1), dtype=np.float64))
            offsets.append(offsets[-1] + len(ids))
    target = Path(directory) / f"part_{number:05d}.npz"
    temporary = target.with_suffix(".partial")
    with temporary.open("wb") as output:
        np.savez(output, features=np.concatenate(feature_parts),
                 item_rows=np.concatenate(item_parts),
                 in_old_pool=np.concatenate(old_parts),
                 D15000=np.concatenate(d15_parts),
                 offsets=np.asarray(offsets, dtype=np.int32))
    os.replace(temporary, target)
    mark = {"queries": len(queries), "rows": offsets[-1],
            "part_sha256": sha256_file(target),
            "first_group": queries[0]["group_id"],
            "last_group": queries[-1]["group_id"]}
    temporary_mark = target.with_suffix(".tmp")
    temporary_mark.write_text(json.dumps(mark), encoding="utf-8")
    os.replace(temporary_mark, target.with_suffix(".json"))
    memory = psutil.Process().memory_info()
    return len(queries), offsets[-1], getattr(memory, "peak_wset", memory.rss)


def cache(stage: str = "dev", workers: int = 2) -> dict:
    if stage not in ("dev", "test") or workers < 1:
        raise ValueError("Invalid D cache stage or worker count")
    if stage == "test" and not (OUT / "decision.json").exists():
        raise ValueError("Diagnostic test requires the frozen D dev decision")
    start = time.perf_counter()
    train = enrich_train(pl.read_parquet("src/avito_test/train.parquet"))
    queries = group_queries(train, stage, 0)
    index = joblib.load(INDEX, mmap_mode="r")
    mapping = {item: row for row, item in enumerate(index.item_ids)}
    for query in queries:
        query["relevant_rows"] = sorted(mapping[item] for item in query["relevant"])
    encode_queries([query["search_query"] for query in queries])
    build_item_vectors()
    signature = cache_signature(stage, queries)
    folder = OUT / "cache" / stage / signature
    folder.mkdir(parents=True, exist_ok=True)
    parts = [queries[position:position+CHUNK]
             for position in range(0, len(queries), CHUNK)]
    pending = []
    total_rows = 0
    for number, chunk in enumerate(parts):
        path = folder / f"part_{number:05d}.npz"
        sidecar = path.with_suffix(".json")
        if path.exists() and sidecar.exists():
            mark = json.loads(sidecar.read_text(encoding="utf-8"))
            if (mark["queries"] != len(chunk) or
                    mark["first_group"] != chunk[0]["group_id"] or
                    mark["last_group"] != chunk[-1]["group_id"] or
                    sha256_file(path) != mark["part_sha256"]):
                raise ValueError(f"Stale or damaged D feature cache: {path}")
            total_rows += mark["rows"]
        else:
            pending.append((number, chunk))
    print(f"D {stage}: {len(queries)} queries, {len(parts)-len(pending)} cached chunks",
          flush=True)
    peak = 0
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                                 initargs=([query["search_query"] for query in queries],)) as executor:
            futures = [executor.submit(cache_chunk, number, chunk, str(folder))
                       for number, chunk in pending]
            done = len(parts)-len(pending)
            for future in as_completed(futures):
                _, rows, memory = future.result()
                done += 1
                total_rows += rows
                peak = max(peak, memory)
                if done % 20 == 0 or done == len(parts):
                    print(f"D {stage} cache: {min(done*CHUNK, len(queries))}/"
                          f"{len(queries)}; {time.perf_counter()-start:.1f}s",
                          flush=True)
    manifest = {"stage": stage, "signature": signature, "queries": len(queries),
                "chunks": len(parts), "rows": total_rows, "feature_count": DIM,
                "seconds": time.perf_counter()-start, "worker_peak_rss_bytes": peak,
                "D15000_sha256": sha256_file(MODELS["D15000"]),
                "sources": source_manifest(Path("src/avito_test"))}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "test"), default="dev")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cache(args.stage, args.workers)
