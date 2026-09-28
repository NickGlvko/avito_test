"""Full-pool B3000/B15000 score blends on the fixed dev and test scopes."""
# Score both B models on identical candidate IDs for a fair blend.
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
from .ranker import FeatureRetriever, candidate_metrics, group_queries, pool_ids
from .semantic_blend import ALPHAS, blended_scores, rank_all, score_candidate_pool, variant
from .semantic_scale_evaluate import INDEX, MODELS, SIGNALS
from .semantic_search import SemanticIndex, build_item_vectors


OUT = Path("artifacts/semantic_blend")
CHUNK = 64
_STATE = None


def initialize(query_texts: list[str]) -> None:
    global _STATE
    threadpool_limits(limits=1)
    index = joblib.load(INDEX, mmap_mode="r")
    signals = joblib.load(SIGNALS, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    items = pl.read_parquet(INDEX.parent / "features.parquet")
    retriever = FeatureRetriever(index, signals, metadata["in_benchmark"].to_numpy(), items)
    semantic = SemanticIndex(query_texts, threads=1)
    models = {}
    for name in ("B3000", "B15000"):
        model = CatBoostRanker()
        model.load_model(str(MODELS[name]))
        models[name] = model
    _STATE = retriever, semantic, models


def score_chunk(number: int, queries: list[dict], parts_dir: str,
                active: tuple[str, ...]) -> tuple[int, int, int]:
    retriever, semantic, models = _STATE
    rows = []
    for query in queries:
        relevant = np.asarray(query["relevant_rows"], dtype=np.int32)
        truth = set(map(int, relevant))
        for scheme, augmented in (("union", False), ("benchmark_plus_positives", True)):
            ids, sources, old, new = score_candidate_pool(
                retriever, semantic, models, query, relevant, augmented)
            rankings = {"B3000": rank_all(ids, old)[:50],
                        "B15000": rank_all(ids, new)[:50]}
            for alpha in ALPHAS:
                name = variant(alpha)
                if name in active:
                    rankings[name] = rank_all(ids, blended_scores(old, new, alpha))[:50]
            history_only = set(map(int, sources["history"])) - set(
                map(int, pool_ids(sources, False)))
            for name in active:
                row = candidate_metrics(rankings[name], ids, truth, retriever.signals,
                                        query, sources["history"], history_only)
                row.update({"stage": query["stage"], "group_id": query["group_id"],
                            "text": query["text"], "scheme": scheme, "variant": name})
                rows.append(row)
    parts = Path(parts_dir)
    destination = parts / f"metrics_{number:05d}.parquet"
    pl.DataFrame(rows, infer_schema_length=None,
        schema_overrides={f"{name}_recall@50": pl.Float64 for name in
            ("seen_item", "new_item", "same_location", "other_location",
             "history_source", "outside_history_source")}).write_parquet(destination)
    (parts / f"done_{number:05d}.json").write_text(
        json.dumps({"queries": len(queries), "rows": len(rows),
                    "metrics_sha256": sha256_file(destination)}), encoding="utf-8")
    memory = psutil.Process().memory_info()
    return number, len(queries), getattr(memory, "peak_wset", memory.rss)


def archived_frame(stage: str) -> pl.DataFrame:
    analysis = json.loads(Path("artifacts/semantic_scale/analysis.json").read_text(
        encoding="utf-8"))
    run = Path(analysis[f"{stage}_path"])
    frame = pl.read_parquet(run / "per_query.parquet")
    if frame.filter(pl.col("variant") == "B15000")["group_id"].n_unique() != (
            35652 if stage == "dev" else 35376):
        raise ValueError("Missing full frozen B15000 control evaluation")
    return frame


def evaluate(stage: str = "dev", workers: int = 3) -> dict:
    if stage not in ("dev", "test") or workers < 1:
        raise ValueError("Unsupported evaluation stage or worker count")
    if stage == "test":
        decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
        if decision["selected"] == "B15000":
            raise ValueError("No blend selected on dev; test is unnecessary")
        if sha256_file(Path(decision["dev_per_query"])) != decision["dev_sha256"]:
            raise ValueError("Frozen dev decision differs from its evaluation")
        active = ("B3000", "B15000", decision["selected"])
    else:
        active = ("B3000", "B15000", *(variant(alpha) for alpha in ALPHAS))
    train = enrich_train(pl.read_parquet("src/avito_test/train.parquet"))
    queries = group_queries(train, stage, 0)
    index = joblib.load(INDEX, mmap_mode="r")
    mapping = {item: row for row, item in enumerate(index.item_ids)}
    for query in queries:
        query["stage"] = stage
        query["relevant_rows"] = sorted(mapping[item] for item in query["relevant"])
    from .semantic_embeddings import encode_queries
    encode_queries([query["search_query"] for query in queries])
    build_item_vectors()
    code = {name: sha256_file(Path(__file__).parent / name) for name in
            ("semantic_blend.py", "semantic_blend_evaluate.py", "semantic_search.py",
             "ranker.py", "retrieval.py", "data.py")}
    signature = digest({"stage": stage, "active": active, "alphas": ALPHAS,
        "models": {name: sha256_file(MODELS[name]) for name in ("B3000", "B15000")},
        "code": code, "sources": source_manifest(Path("src/avito_test")),
        "vectors": json.loads(Path("artifacts/semantic/embeddings/encode_state.json")
                              .read_text(encoding="utf-8"))["vector_sha256"]})
    run = OUT / "evaluation" / stage / signature[:16]
    parts = run / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    chunks = [queries[position:position+CHUNK] for position in range(0, len(queries), CHUNK)]
    pending = []
    for number, chunk in enumerate(chunks):
        path = parts / f"metrics_{number:05d}.parquet"
        mark = parts / f"done_{number:05d}.json"
        if path.exists() and mark.exists():
            saved = json.loads(mark.read_text(encoding="utf-8"))
            if saved != {"queries": len(chunk), "rows": len(chunk)*2*len(active),
                         "metrics_sha256": sha256_file(path)}:
                raise ValueError(f"Damaged evaluation chunk {number}")
        else:
            pending.append((number, chunk))
    start = time.perf_counter()
    completed = sum(len(chunk) for number, chunk in enumerate(chunks)
                    if number not in {pending_number for pending_number, _ in pending})
    peak = 0
    print(f"Blend {stage}: {len(queries)} queries; {len(chunks)-len(pending)} cached chunks",
          flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
            initargs=([query["search_query"] for query in queries],)) as pool:
            futures = [pool.submit(score_chunk, number, chunk, str(parts), active)
                       for number, chunk in pending]
            for future in as_completed(futures):
                _, count, memory = future.result()
                completed += count
                peak = max(peak, memory)
                if completed % (CHUNK*10) < CHUNK:
                    print(f"Blend {stage}: {min(completed, len(queries))}/{len(queries)}; "
                          f"{time.perf_counter()-start:.1f}s", flush=True)
    frame = pl.read_parquet(str(parts / "metrics_*.parquet"))
    expected = len(queries)*2*len(active)
    if frame.height != expected or frame.unique(["group_id", "scheme", "variant"]).height != expected:
        raise ValueError(f"Incomplete blend evaluation: {frame.height} != {expected}")
    frame = frame.sort("group_id", "scheme", "variant")
    reference = archived_frame(stage)
    for name in ("B3000", "B15000"):
        old = reference.filter(pl.col("variant") == name).select(
            "group_id", "scheme", pl.col("recall@50").alias("old_recall"),
            pl.col("candidate_recall").alias("old_pool"))
        paired = frame.filter(pl.col("variant") == name).join(
            old, on=["group_id", "scheme"], validate="1:1")
        if (paired.height != len(queries)*2 or
                not np.array_equal(paired["recall@50"].to_numpy(),
                                   paired["old_recall"].to_numpy()) or
                not np.array_equal(paired["candidate_recall"].to_numpy(),
                                   paired["old_pool"].to_numpy())):
            raise ValueError(f"Frozen {name} full-pool control changed")
    output = run / "per_query.parquet"
    frame.write_parquet(output)
    summary = frame.group_by("scheme", "variant").agg(
        pl.len().alias("queries"), pl.col("recall@50").mean().alias("recall50"),
        pl.col("candidate_recall").mean().alias("pool_recall"),
        pl.col("lost_at_selection").sum().alias("selection_losses"),
        pl.col("same_location_recall@50").mean().alias("same_location_recall50"),
        pl.col("other_location_recall@50").mean().alias("other_location_recall50")
    ).sort("scheme", "variant")
    summary.write_csv(run / "summary.csv")
    report = {"stage": stage, "queries": len(queries), "active": active,
              "per_query": str(output), "per_query_sha256": sha256_file(output),
              "seconds": time.perf_counter()-start, "worker_peak_rss_bytes": peak,
              "control_predictions_reproduced": True}
    (run / "timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "test"), default="dev")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    evaluate(args.stage, args.workers)
