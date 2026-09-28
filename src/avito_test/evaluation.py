"""Macro recall by full query, channel ablations and error diagnostics."""
# Average recall over complete query groups, not over positive training rows.
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .data import evaluation_groups, normalize, split_train, write_json
from .retrieval import DEPTH, EXPERIMENTS, TrainSignals, channels, fuse


def recall(prediction, relevant, k: int) -> float:
    truth = set(relevant)
    if not truth:
        raise ValueError("Recall is undefined for an empty relevance set")
    return len(set(prediction[:k]) & truth) / len(truth)


def evaluate(index, items: pl.DataFrame, train: pl.DataFrame, output: Path,
             protocol: str = "group", stage: str = "dev", limit: int = 0,
             experiments: dict | None = None) -> dict:
    experiments = EXPERIMENTS if experiments is None else experiments
    run_dir = output / f"{protocol}_{stage}"
    run_dir.mkdir(parents=True, exist_ok=True)
    training = split_train(train, protocol)
    signals = TrainSignals(training, items)
    groups, eligibility = evaluation_groups(train, index.item_ids, protocol, stage, limit)
    write_json(run_dir / "eligibility.json", eligibility)
    rows_by_id = {item: i for i, item in enumerate(index.item_ids)}
    titles = items["item_title_raw"].to_list()
    metrics = []
    examples = []
    cached = {}
    start = time.perf_counter()
    print(f"Evaluating {protocol}/{stage}: {groups.height} groups; training rows: {training.height}", flush=True)
    for number, query in enumerate(groups.iter_rows(named=True)):
        relevant = {rows_by_id[x] for x in query["relevant"]}
        results = channels(index, signals, query)
        union = set(int(x) for values in results.values() for x in values)
        for source, values in results.items():
            if source not in cached:
                cached[source] = np.full((groups.height, DEPTH), -1, dtype=np.int32)
            cached[source][number, :len(values)] = values
        for name, weights in experiments.items():
            prediction = fuse(results, weights)
            selected_union = set(int(x) for source in weights for x in results[source])
            metrics.append({"group_id": query["group_id"], "experiment": name,
                            **{f"recall@{k}": recall(prediction, relevant, k) for k in [50, 100, 200, 500]},
                            "hit@50": float(bool(set(prediction[:50]) & relevant)),
                            "union_recall": len(union & relevant) / len(relevant),
                            "selected_union_recall": len(selected_union & relevant) / len(relevant),
                            "known_text": query["text"] in signals.seen_texts,
                            "all_items_seen": all(x in signals.seen_items for x in query["relevant"]),
                            "has_filters": bool(query["search_infm_params_text"]),
                            "category": query["search_category"], "positives": len(relevant),
                            "ceiling@50": min(1., 50 / len(relevant))})
        # Store factual per-source ranks; interpretation is separate from labels.
        for positive in sorted(relevant):
            ranks = {}
            for source, values in results.items():
                found = np.flatnonzero(values == positive)
                ranks[source] = int(found[0] + 1) if len(found) else None
            examples.append({"group_id": query["group_id"], "search_query": query["search_query"],
                             "search_location_id": query["search_location_id"],
                             "search_filters": query["search_infm_params_text"],
                             "item_id": index.item_ids[positive], "title": titles[positive],
                             "same_location": bool(signals.locations[positive] == query["search_location_id"]),
                             "title_token_overlap": len(set(normalize(query["search_query"]).split()) &
                                                        set(normalize(titles[positive]).split())),
                             "source_ranks": ranks})
        if (number + 1) % 100 == 0:
            print(f"{protocol}/{stage}: {number + 1}/{groups.height}, {time.perf_counter()-start:.1f}s", flush=True)
    frame = pl.DataFrame(metrics)
    frame.write_parquet(run_dir / "per_query.parquet")
    metric_columns = ["recall@50", "recall@100", "recall@200", "recall@500", "hit@50",
                      "union_recall", "selected_union_recall", "ceiling@50"]
    summary = frame.group_by("experiment").agg(pl.len().alias("queries"),
        *[pl.col(c).mean() for c in metric_columns]).sort(["recall@50", "experiment"], descending=[True, False])
    summary.write_csv(run_dir / "metrics.csv")
    slices = {}
    for feature in ["known_text", "all_items_seen", "has_filters", "category"]:
        slices[feature] = frame.group_by("experiment", feature).agg(
            pl.len().alias("queries"), pl.col("recall@50").mean()).sort("experiment", feature).to_dicts()
    write_json(run_dir / "slices.json", slices)
    write_json(run_dir / "timing.json", {"search_seconds": time.perf_counter() - start,
                                         "seconds_per_query": (time.perf_counter() - start) / groups.height})
    groups.write_parquet(run_dir / "queries.parquet")
    np.savez_compressed(run_dir / "channels.npz", **cached)
    with (run_dir / "positive_ranks.jsonl").open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    result = {"eligibility": eligibility, "metrics": summary.to_dicts()}
    write_json(run_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def diagnose(output: Path, weights: dict, protocol: str = "group", stage: str = "dev") -> dict:
    """Re-use cached rankings to distinguish candidate misses from top-50 selection misses."""
    run_dir = output / f"{protocol}_{stage}"
    groups = pl.read_parquet(run_dir / "queries.parquet")
    with np.load(run_dir / "channels.npz") as archive:
        cached = {name: archive[name] for name in archive.files}
    counts = {"missed_by_sources": 0, "lost_at_selection": 0, "retrieved": 0}
    corpus_ids = json.loads((output / "item_ids.json").read_text(encoding="utf-8"))
    mapping = {x: i for i, x in enumerate(corpus_ids)}
    errors = []
    examples = [json.loads(line) for line in (run_dir / "positive_ranks.jsonl").read_text(encoding="utf-8").splitlines()]
    by_key = {(x["group_id"], x["item_id"]): x for x in examples}
    for number, query in enumerate(groups.iter_rows(named=True)):
        channels_here = {source: values[number][values[number] >= 0] for source, values in cached.items()}
        prediction = set(fuse(channels_here, weights)[:50])
        union = {int(x) for source in weights for x in channels_here[source]}
        for item in query["relevant"]:
            idx = mapping[item]
            reason = "retrieved" if idx in prediction else "lost_at_selection" if idx in union else "missed_by_sources"
            counts[reason] += 1
            if reason != "retrieved":
                errors.append({**by_key[query["group_id"], item], "failure_stage": reason})
    write_json(run_dir / "errors_summary.json", counts)
    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as stream:
        for error in errors:
            stream.write(json.dumps(error, ensure_ascii=False) + "\n")
    return counts
