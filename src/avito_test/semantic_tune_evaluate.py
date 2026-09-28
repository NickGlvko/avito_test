"""Evaluate CatBoost tree checkpoints on cached, complete B candidate pools."""
# Measure macro Recall@50 on full pools, not training loss on sampled rows.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import psutil
from catboost import CatBoostRanker

from .data import digest, enrich_train, sha256_file
from .ranker import candidate_metrics, group_queries
from .semantic_blend import blended_scores, rank_all
from .semantic_blend_evaluate import OUT as BLEND_OUT
from .semantic_scale_evaluate import INDEX, SIGNALS
from .semantic_tune_cache import OUT, cache_signature
from .semantic_tune_train import CHECKPOINTS


SCHEMES = ("union", "benchmark_plus_positives")
CONTROL = "submitted_blend"


def query_context(stage: str) -> tuple[list[dict], object]:
    queries = group_queries(enrich_train(pl.read_parquet(
        "src/avito_test/train.parquet")), stage, 0)
    index = joblib.load(INDEX, mmap_mode="r")
    mapping = {item: row for row, item in enumerate(index.item_ids)}
    for query in queries:
        query["relevant_rows"] = sorted(mapping[item] for item in query["relevant"])
    signals = joblib.load(SIGNALS, mmap_mode="r")
    return queries, signals


def cached_folder(stage: str, queries: list[dict]) -> Path:
    folder = OUT / "cache" / stage / cache_signature(stage, queries)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["stage"] != stage or manifest["signature"] != folder.name or
            manifest["queries"] != len(queries) or manifest["feature_count"] != 44):
        raise ValueError("Full-pool cache manifest changed")
    return folder


def cached_chunks(folder: Path, queries: list[dict]):
    for number, first in enumerate(range(0, len(queries), 64)):
        chunk = queries[first:first+64]
        path = folder / f"part_{number:05d}.npz"
        mark = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (mark["queries"] != len(chunk) or
                mark["first_group"] != chunk[0]["group_id"] or
                mark["last_group"] != chunk[-1]["group_id"] or
                sha256_file(path) != mark["part_sha256"]):
            raise ValueError(f"Damaged cached pool: {path}")
        with np.load(path) as data:
            arrays = {key: data[key] for key in
                      ("features", "item_rows", "B3000", "B15000", "offsets")}
        if (arrays["features"].shape != (mark["rows"], 44) or
                len(arrays["offsets"]) != 2*len(chunk)+1 or
                arrays["offsets"][-1] != mark["rows"]):
            raise ValueError(f"Cached full query pool shape changed: {path}")
        yield number, chunk, arrays


def controls(arrays: dict) -> dict[str, np.ndarray]:
    blend = np.empty_like(arrays["B3000"])
    offsets = arrays["offsets"]
    for begin, end in zip(offsets[:-1], offsets[1:]):
        blend[begin:end] = blended_scores(arrays["B3000"][begin:end],
                                          arrays["B15000"][begin:end], 0.75)
    return {"B3000": arrays["B3000"], "B15000": arrays["B15000"],
            CONTROL: blend}


def append_metrics(records: list[dict], chunk: list[dict], arrays: dict,
                   scores: dict[str, np.ndarray], signals, stage: str) -> None:
    offsets = arrays["offsets"]
    empty = np.empty(0, dtype=np.int32)
    for position, query in enumerate(chunk):
        truth = set(map(int, query["relevant_rows"]))
        for scheme_index, scheme in enumerate(SCHEMES):
            pair = 2*position + scheme_index
            begin, end = int(offsets[pair]), int(offsets[pair+1])
            candidates = arrays["item_rows"][begin:end]
            for variant, values in scores.items():
                selected = rank_all(candidates, values[begin:end])[:50]
                row = candidate_metrics(selected, candidates, truth, signals,
                                        query, empty, set())
                row.update({"stage": stage, "scheme": scheme, "variant": variant,
                            "group_id": query["group_id"], "text": query["text"]})
                records.append(row)


def validate_controls(frame: pl.DataFrame, stage: str) -> None:
    analysis = json.loads((BLEND_OUT / "analysis.json").read_text(encoding="utf-8"))
    reference = pl.read_parquet(Path(analysis[f"{stage}_path"]) / "per_query.parquet")
    for variant, original in (("B3000", "B3000"), ("B15000", "B15000"),
                              (CONTROL, "blend_0.75")):
        current = frame.filter(pl.col("variant") == variant).select(
            "group_id", "scheme", "recall@50", "candidate_recall",
            "lost_at_selection")
        archived = reference.filter(pl.col("variant") == original).select(
            "group_id", "scheme", pl.col("recall@50").alias("old_recall"),
            pl.col("candidate_recall").alias("old_pool"),
            pl.col("lost_at_selection").alias("old_lost"))
        paired = current.join(archived, on=["group_id", "scheme"], validate="1:1")
        if (paired.height != 2*(35652 if stage == "dev" else 35376) or
                not np.array_equal(paired["recall@50"].to_numpy(),
                                   paired["old_recall"].to_numpy()) or
                not np.array_equal(paired["candidate_recall"].to_numpy(),
                                   paired["old_pool"].to_numpy()) or
                not np.array_equal(paired["lost_at_selection"].to_numpy(),
                                   paired["old_lost"].to_numpy())):
            raise ValueError(f"Cached full-pool {variant} differs from frozen control")


def summary(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.group_by("scheme", "variant").agg(
        pl.len().alias("queries"), pl.col("recall@50").mean().alias("recall50"),
        pl.col("candidate_recall").mean().alias("pool_recall"),
        pl.col("lost_at_selection").sum().alias("selection_losses"),
        pl.col("same_location_recall@50").mean().alias("same_location_recall50"),
        pl.col("other_location_recall@50").mean().alias("other_location_recall50")
    ).sort("scheme", "variant")


def evaluate_checkpoints() -> dict:
    stage = "dev"
    started = time.perf_counter()
    queries, signals = query_context(stage)
    folder = cached_folder(stage, queries)
    models = {}
    for depth in (6, 8):
        report = json.loads((OUT / f"depth_{depth}_960.json").read_text(encoding="utf-8"))
        path = Path(report["model"])
        if sha256_file(path) != report["model_sha256"]:
            raise ValueError(f"Full depth-{depth} training model changed")
        model = CatBoostRanker()
        model.load_model(str(path))
        if model.tree_count_ != 960:
            raise ValueError("Full model has the wrong number of trees")
        models[depth] = model
    output = OUT / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    settings = digest({"cache_manifest_sha256": sha256_file(folder / "manifest.json"),
                       "depth_6_sha256": sha256_file(OUT / "depth_6_960.cbm"),
                       "depth_8_sha256": sha256_file(OUT / "depth_8_960.cbm"),
                       "code_sha256": sha256_file(Path(__file__))})
    prediction_seconds = {f"d{depth}": 0. for depth in models}
    for number, chunk, arrays in cached_chunks(folder, queries):
        part = output / f"part_{number:05d}.parquet"
        mark = part.with_suffix(".json")
        if part.exists() and mark.exists():
            saved = json.loads(mark.read_text(encoding="utf-8"))
            if (saved["part_sha256"] != sha256_file(part) or
                    saved["queries"] != len(chunk) or
                    saved["settings"] != settings):
                raise ValueError(f"Damaged checkpoint metrics: {part}")
            continue
        scores = controls(arrays)
        for depth, model in models.items():
            tick = time.perf_counter()
            stages = model.staged_predict(arrays["features"], ntree_end=960,
                                          eval_period=240, thread_count=6)
            for trees, values in zip(CHECKPOINTS, stages, strict=True):
                scores[f"d{depth}_t{trees}"] = np.asarray(values, dtype=np.float64).copy()
            prediction_seconds[f"d{depth}"] += time.perf_counter()-tick
        rows: list[dict] = []
        append_metrics(rows, chunk, arrays, scores, signals, stage)
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(part)
        mark.write_text(json.dumps({"queries": len(chunk), "settings": settings,
                                    "part_sha256": sha256_file(part)}), encoding="utf-8")
        if (number+1) % 20 == 0:
            print(f"Checkpoint dev: {min((number+1)*64, len(queries))}/"
                  f"{len(queries)}; {time.perf_counter()-started:.1f}s", flush=True)
    frame = pl.read_parquet(str(output / "part_*.parquet"))
    expected = len(queries)*2*(3+8)
    if frame.height != expected or frame.unique(["group_id", "scheme", "variant"]).height != expected:
        raise ValueError("Incomplete full-pool checkpoint evaluation")
    validate_controls(frame, stage)
    checkpoint_240 = frame.filter(pl.col("variant") == "d6_t240").select(
        "group_id", "scheme", pl.col("recall@50").alias("checkpoint_recall"))
    original_240 = frame.filter(pl.col("variant") == "B15000").select(
        "group_id", "scheme", pl.col("recall@50").alias("original_recall"))
    paired_240 = checkpoint_240.join(original_240, on=["group_id", "scheme"],
                                     validate="1:1")
    if not np.array_equal(paired_240["checkpoint_recall"].to_numpy(),
                          paired_240["original_recall"].to_numpy()):
        raise ValueError("Depth-6 240-tree checkpoint failed to reproduce B15000")
    frame.sort("group_id", "scheme", "variant").write_parquet(output / "per_query.parquet")
    table = summary(frame)
    table.write_csv(output / "summary.csv")
    memory = psutil.Process().memory_info()
    report = {"stage": stage, "cache": str(folder), "cache_manifest_sha256":
              sha256_file(folder / "manifest.json"), "per_query_sha256":
              sha256_file(output / "per_query.parquet"),
              "settings": settings,
              "seconds": time.perf_counter()-started,
              "model_prediction_seconds": prediction_seconds,
              "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
              "controls_reproduced_per_query": True}
    (output / "timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(table, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("checkpoints",))
    if parser.parse_args().mode == "checkpoints":
        evaluate_checkpoints()
