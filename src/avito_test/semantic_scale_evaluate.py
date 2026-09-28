"""Paired full-pool evaluation of 3k and 15k B/D rankers."""
# Keep dev candidates fixed when measuring the effect of more OOF groups.
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
from .ranker import FeatureRetriever, TEXT_FEATURES, candidate_metrics, group_queries, pool_ids
from .semantic_search import SemanticIndex, build_item_vectors, semantic_features
from .semantic_scale_select import OUT, SELECTED


INDEX = Path("artifacts/expanded/cache/73915bb3bc082213/union_index.joblib")
SIGNALS = Path("artifacts/ranker/full/c392362a99e2c4d1/full_train_signals.joblib")
MODELS = {
    "B3000": Path("artifacts/semantic/models/B_existing_pool_semantic_features.cbm"),
    "D3000": Path("artifacts/semantic/models/D_expanded_pool_semantic_features.cbm"),
    "B15000": Path("artifacts/semantic/models/B_15000_existing_pool_semantic_features.cbm"),
    "D15000": Path("artifacts/semantic/models/D_15000_expanded_pool_semantic_features.cbm"),
}
ARCHIVED = Path("artifacts/semantic/evaluation")
CHUNK = 48
_STATE = None


def initialize(query_texts: list[str], active: tuple[str, ...]) -> None:
    global _STATE
    threadpool_limits(limits=1)
    index = joblib.load(INDEX, mmap_mode="r")
    signals = joblib.load(SIGNALS, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    items = pl.read_parquet(INDEX.parent / "features.parquet")
    retriever = FeatureRetriever(index, signals, metadata["in_benchmark"].to_numpy(), items)
    semantic = SemanticIndex(query_texts, threads=1)
    models = {}
    for variant in active:
        model = CatBoostRanker()
        model.load_model(str(MODELS[variant]))
        models[variant] = model
    _STATE = retriever, semantic, models


def score_chunk(number: int, queries: list[dict], directory: str) -> tuple[int, int, int]:
    retriever, semantic, models = _STATE
    metrics = []
    positive_details = []
    for query in queries:
        relevant = np.asarray(query["relevant_rows"], dtype=np.int32)
        truth = set(map(int, relevant))
        for scheme, augmented in (("union", False), ("benchmark_plus_positives", True)):
            base, scores, allowed = retriever.retrieve_with_scores(query, relevant, augmented)
            old_pool = pool_ids(base)
            dense = semantic.retrieve(query["search_query"], query["search_location_id"],
                                      relevant, augmented)
            results = {**base, **dense}
            expanded = np.unique(np.concatenate((old_pool, *dense.values()))).astype(np.int32)
            lexical = retriever.features(query, expanded, base, scores, allowed)[:, :len(TEXT_FEATURES)]
            old_positions = np.searchsorted(expanded, old_pool)
            b_features = np.column_stack((lexical[old_positions],
                semantic_features(semantic, query, old_pool, results, False)))
            d_features = np.column_stack((lexical,
                semantic_features(semantic, query, expanded, results, True)))
            rankings = {}
            for variant, model in models.items():
                candidates, features = ((old_pool, b_features) if variant.startswith("B")
                                        else (expanded, d_features))
                values = model.predict(features, thread_count=1)
                rankings[variant] = candidates[np.lexsort((candidates, -values))]
            old_set = set(map(int, old_pool))
            semantic_only = truth & (set(map(int, expanded)) - old_set)
            b15_selected = set(map(int, rankings["B15000"][:50])) if "B15000" in rankings else set()
            d15_selected = set(map(int, rankings["D15000"][:50])) if "D15000" in rankings else set()
            history_only = set(map(int, base["history"])) - set(map(int, pool_ids(base, False)))
            for variant, ranking in rankings.items():
                pool = old_pool if variant.startswith("B") else expanded
                selected = ranking[:50]
                row = candidate_metrics(selected, pool, truth, retriever.signals,
                                        query, base["history"], history_only)
                row.update({"stage": query["stage"], "group_id": query["group_id"],
                            "text": query["text"], "scheme": scheme, "variant": variant,
                            "semantic_only_positives": len(semantic_only),
                            "semantic_only_in_top50": len(semantic_only & set(map(int, selected))),
                            "old_positive_lost_vs_B15000": len((truth & old_set & b15_selected) -
                                                               set(map(int, selected)))
                                    if b15_selected else None,
                            "old_positive_gained_vs_B15000": len((truth & old_set & set(map(int, selected))) -
                                                                 b15_selected)
                                    if b15_selected else None})
                metrics.append(row)
            dense_positions = {name: {int(item): position for position, item in
                               enumerate(dense[name], 1)} for name in dense}
            for item in relevant:
                value = int(item)
                detail = {"stage": query["stage"], "scheme": scheme,
                          "group_id": query["group_id"], "text": query["text"],
                          "item_row": value, "in_old_pool": value in old_set,
                          "semantic_only": value in semantic_only,
                          "same_location": bool(retriever.signals.locations[value] ==
                                                query["search_location_id"]),
                          "semantic_global_rank": dense_positions["semantic_global"].get(value, 0),
                          "semantic_local_rank": dense_positions["semantic_local"].get(value, 0)}
                for variant, ranking in rankings.items():
                    positions = np.flatnonzero(ranking == value)
                    detail[variant + "_rank"] = int(positions[0]) + 1 if len(positions) else 0
                positive_details.append(detail)
    part = Path(directory)
    pl.DataFrame(metrics, infer_schema_length=None,
        schema_overrides={f"{name}_recall@50": pl.Float64 for name in
            ("seen_item", "new_item", "same_location", "other_location",
             "history_source", "outside_history_source")}).write_parquet(
                 part / f"metrics_{number:05d}.parquet")
    pl.DataFrame(positive_details, infer_schema_length=None).write_parquet(
        part / f"positives_{number:05d}.parquet")
    (part / f"done_{number:05d}.json").write_text(json.dumps({"queries": len(queries)}),
                                                   encoding="utf-8")
    memory = psutil.Process().memory_info()
    return number, len(queries), getattr(memory, "peak_wset", memory.rss)


def evaluate(stage: str = "dev", workers: int = 2, limit: int = 0) -> dict:
    if stage not in ("dev", "test") or workers < 1 or limit < 0:
        raise ValueError("Invalid stage, worker count, or limit")
    if stage == "test":
        decision_path = OUT / "decision.json"
        if not decision_path.exists():
            raise ValueError("Test requires a full-dev frozen decision")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision["selected"] == "B3000":
            raise ValueError("No new ranker selected; the submitted B control remains")
        dev_paths = list((OUT / "evaluation/dev/full").glob("*/per_query.parquet"))
        if not any(sha256_file(path) == decision["dev_per_query_sha256"] for path in dev_paths):
            raise ValueError("Frozen decision no longer matches the full dev results")
        active = ("B3000", decision["selected"])
    else:
        active = ("B3000", "D3000", "B15000", "D15000")
    training = json.loads((OUT / "training.json").read_text(encoding="utf-8"))
    if training["audit"]["selected_groups"] != 15000 or sha256_file(SELECTED) != training["audit"]["signature"]["selected_sha256"]:
        raise ValueError("The 15,000-group training sample changed")
    train = enrich_train(pl.read_parquet("src/avito_test/train.parquet"))
    queries = group_queries(train, stage, limit)
    index = joblib.load(INDEX, mmap_mode="r")
    mapping = {item: row for row, item in enumerate(index.item_ids)}
    for query in queries:
        query["stage"] = stage
        query["relevant_rows"] = sorted(mapping[item] for item in query["relevant"])
    from .semantic_embeddings import encode_queries
    encode_queries([query["search_query"] for query in queries])
    build_item_vectors()
    code = {name: sha256_file(Path(__file__).parent / name) for name in
            ("semantic_scale_evaluate.py", "semantic_search.py", "ranker.py",
             "retrieval.py", "data.py")}
    signature = digest({"stage": stage, "limit": limit, "active": active,
        "models": {variant: sha256_file(MODELS[variant]) for variant in active},
        "code": code, "sources": source_manifest(Path("src/avito_test")),
        "vectors": json.loads(Path("artifacts/semantic/embeddings/encode_state.json").read_text(
            encoding="utf-8"))["vector_sha256"]})
    run = OUT / "evaluation" / stage / ("full" if not limit else f"slice_{limit}") / signature[:16]
    parts = run / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    chunks = [queries[position:position+CHUNK] for position in range(0, len(queries), CHUNK)]
    pending = [(number, chunk) for number, chunk in enumerate(chunks) if not all(
        (parts / f"{name}_{number:05d}.{extension}").exists() for name, extension in
        (("metrics", "parquet"), ("positives", "parquet"), ("done", "json")))]
    started = time.perf_counter()
    peak = 0
    print(f"Scale {stage}: {len(queries)} queries, {len(chunks)-len(pending)} cached parts", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                             initargs=([query["search_query"] for query in queries], active)) as pool:
        futures = [pool.submit(score_chunk, number, chunk, str(parts)) for number, chunk in pending]
        completed = (len(chunks)-len(pending))*CHUNK
        for future in as_completed(futures):
            _, count, memory = future.result()
            completed += count
            peak = max(peak, memory)
            if completed % (CHUNK*10) < CHUNK:
                print(f"Scale {stage}: {min(completed,len(queries))}/{len(queries)}; "
                      f"{time.perf_counter()-started:.1f}s", flush=True)
    frame = pl.read_parquet(str(parts / "metrics_*.parquet"))
    positives = pl.read_parquet(str(parts / "positives_*.parquet"))
    expected = len(queries)*2*len(active)
    if frame.height != expected or frame.unique(["group_id", "scheme", "variant"]).height != expected:
        raise ValueError(f"Incomplete scaled stage predictions: {frame.height} != {expected}")
    if positives.height != 2*sum(len(query["relevant_rows"]) for query in queries):
        raise ValueError("Incomplete scaled positive ranks")
    frame.sort("group_id", "scheme", "variant").write_parquet(run / "per_query.parquet")
    positives.sort("group_id", "scheme", "item_row").write_parquet(run / "positive_ranks.parquet")
    reference = pl.read_parquet(ARCHIVED / stage / "full" /
        ("d139b03a27932e37" if stage == "dev" else "bff6b3ac7f8e9508") /
        "per_query.parquet")
    for variant, original_name in (("B3000", "B"), ("D3000", "D")):
        if variant not in active:
            continue
        old = reference.filter(pl.col("variant") == original_name).select(
            "group_id", "scheme", pl.col("recall@50").alias("old_recall"),
            pl.col("candidate_recall").alias("old_pool"))
        paired = frame.filter(pl.col("variant") == variant).join(
            old, on=["group_id", "scheme"], validate="1:1")
        if (paired.height != len(queries)*2 or
                not np.array_equal(paired["recall@50"].to_numpy(),
                                   paired["old_recall"].to_numpy()) or
                not np.array_equal(paired["candidate_recall"].to_numpy(),
                                   paired["old_pool"].to_numpy())):
            raise ValueError(f"Frozen {variant} predictions or candidate pools changed")
    # All four variants must see exactly the archived candidate pools.
    for variant, original_name in (("B15000", "B"), ("D15000", "D")):
        if variant not in active:
            continue
        old = reference.filter(pl.col("variant") == original_name).select(
            "group_id", "scheme", pl.col("candidate_recall").alias("old_pool"))
        paired = frame.filter(pl.col("variant") == variant).join(
            old, on=["group_id", "scheme"], validate="1:1")
        if not np.array_equal(paired["candidate_recall"].to_numpy(),
                              paired["old_pool"].to_numpy()):
            raise ValueError(f"{variant} candidate pool changed")
    summary = frame.group_by("scheme", "variant").agg(
        pl.len().alias("queries"), pl.col("recall@50").mean().alias("recall50"),
        pl.col("candidate_recall").mean().alias("pool_recall"),
        pl.col("lost_at_selection").sum().alias("selection_losses"),
        pl.col("same_location_recall@50").mean().alias("same_location_recall50"),
        pl.col("other_location_recall@50").mean().alias("other_location_recall50"),
        pl.col("semantic_only_positives").sum().alias("semantic_only_positives"),
        pl.col("semantic_only_in_top50").sum().alias("semantic_only_in_top50"),
        pl.col("old_positive_lost_vs_B15000").sum().alias("old_positive_lost_vs_B15000"),
        pl.col("old_positive_gained_vs_B15000").sum().alias("old_positive_gained_vs_B15000")
    ).sort("scheme", "variant")
    summary.write_csv(run / "summary.csv")
    report = {"stage": stage, "query_count": len(queries), "seconds": time.perf_counter()-started,
              "worker_peak_rss_bytes": peak, "active": active,
              "per_query": str(run / "per_query.parquet"),
              "positive_ranks": str(run / "positive_ranks.parquet"),
              "control_predictions_reproduced": True}
    (run / "timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary, flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["dev", "test"], default="dev")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()
    evaluate(arguments.stage, arguments.workers, arguments.limit)
