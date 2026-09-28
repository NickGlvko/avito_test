"""Compare the selected checkpoint and its fixed 50/50 blend with the submitted blend."""
# Compare one predetermined 50/50 blend after choosing a checkpoint.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import psutil
from catboost import CatBoostRanker

from .data import digest, sha256_file
from .semantic_blend import normalize_scores
from .semantic_report import compare
from .semantic_tune_cache import OUT
from .semantic_tune_evaluate import (CONTROL, append_metrics, cached_chunks,
                                     cached_folder, controls, query_context,
                                     summary, validate_controls)
from .semantic_tune_select import CHECKPOINT_DECISION, SELECTED_MODEL


FINAL_DECISION = OUT / "decision.json"
VARIANTS = ("new_ranker", "half_blend")


def final_scores(arrays: dict, model: CatBoostRanker) -> dict[str, np.ndarray]:
    scores = controls(arrays)
    new = np.asarray(model.predict(arrays["features"], thread_count=6),
                     dtype=np.float64)
    half = np.empty_like(new)
    offsets = arrays["offsets"]
    for begin, end in zip(offsets[:-1], offsets[1:]):
        half[begin:end] = (normalize_scores(new[begin:end]) +
                           normalize_scores(scores[CONTROL][begin:end])) * 0.5
    scores.update({"new_ranker": new, "half_blend": half})
    return scores


def evaluate(stage: str) -> dict:
    if stage not in ("dev", "test"):
        raise ValueError("Only dev or diagnostic test can be evaluated")
    checkpoint = json.loads(CHECKPOINT_DECISION.read_text(encoding="utf-8"))
    if sha256_file(SELECTED_MODEL) != checkpoint["selected_model_sha256"]:
        raise ValueError("Chosen ranker weights changed")
    if stage == "test":
        decision = json.loads(FINAL_DECISION.read_text(encoding="utf-8"))
        if decision["selected"] == CONTROL or sha256_file(
                OUT / "final/dev/per_query.parquet") != decision["dev_sha256"]:
            raise ValueError("Test requires a frozen improving dev choice")
    started = time.perf_counter()
    queries, signals = query_context(stage)
    folder = cached_folder(stage, queries)
    model = CatBoostRanker()
    model.load_model(str(SELECTED_MODEL))
    if model.tree_count_ != checkpoint["trees"]:
        raise ValueError("Shrunk model has the wrong number of trees")
    output = OUT / "final" / stage
    output.mkdir(parents=True, exist_ok=True)
    settings = digest({"cache_manifest_sha256": sha256_file(folder / "manifest.json"),
                       "selected_model_sha256": checkpoint["selected_model_sha256"],
                       "checkpoint_decision_sha256": sha256_file(CHECKPOINT_DECISION),
                       "code_sha256": sha256_file(Path(__file__))})
    predict_seconds = 0.
    for number, chunk, arrays in cached_chunks(folder, queries):
        path = output / f"part_{number:05d}.parquet"
        mark = path.with_suffix(".json")
        if path.exists() and mark.exists():
            saved = json.loads(mark.read_text(encoding="utf-8"))
            if (saved["part_sha256"] != sha256_file(path) or
                    saved["queries"] != len(chunk) or
                    saved["settings"] != settings):
                raise ValueError(f"Damaged final metrics: {path}")
            continue
        tick = time.perf_counter()
        scores = final_scores(arrays, model)
        predict_seconds += time.perf_counter()-tick
        rows: list[dict] = []
        append_metrics(rows, chunk, arrays, scores, signals, stage)
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(path)
        mark.write_text(json.dumps({"queries": len(chunk), "settings": settings,
                                    "part_sha256": sha256_file(path)}), encoding="utf-8")
        if (number+1) % 20 == 0:
            print(f"Final {stage}: {min((number+1)*64, len(queries))}/"
                  f"{len(queries)}; {time.perf_counter()-started:.1f}s", flush=True)
    frame = pl.read_parquet(str(output / "part_*.parquet"))
    expected = len(queries)*2*5
    if frame.height != expected or frame.unique(["group_id", "scheme", "variant"]).height != expected:
        raise ValueError("Incomplete final full-pool evaluation")
    validate_controls(frame, stage)
    frame.sort("group_id", "scheme", "variant").write_parquet(output / "per_query.parquet")
    table = summary(frame)
    table.write_csv(output / "summary.csv")
    memory = psutil.Process().memory_info()
    timing = {"stage": stage, "seconds": time.perf_counter()-started,
              "new_model_predict_seconds": predict_seconds,
              "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
              "cache": str(folder), "selected_model_sha256": checkpoint["selected_model_sha256"],
              "settings": settings,
              "per_query_sha256": sha256_file(output / "per_query.parquet"),
              "controls_reproduced_per_query": True}
    (output / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(table, flush=True)
    return timing


def select() -> dict:
    source = OUT / "final/dev/per_query.parquet"
    frame = pl.read_parquet(source)
    if (frame["stage"].unique().to_list() != ["dev"] or
            frame.filter(pl.col("variant") == CONTROL)["group_id"].n_unique() != 35652):
        raise ValueError("Only full dev can select the final variant")
    table = summary(frame)
    rows = {scheme: {row["variant"]: row for row in
                     table.filter(pl.col("scheme") == scheme).to_dicts()}
            for scheme in ("union", "benchmark_plus_positives")}
    comparisons = {scheme: {name: compare(frame, scheme, name, CONTROL)
                            for name in VARIANTS} for scheme in rows}
    eligible = [name for name in VARIANTS if
                comparisons["union"][name]["ci95"][0] > 0 and
                rows["benchmark_plus_positives"][name]["recall50"] >=
                rows["benchmark_plus_positives"][CONTROL]["recall50"]]
    chosen = max(eligible, key=lambda name: (
        rows["union"][name]["recall50"],
        rows["benchmark_plus_positives"][name]["recall50"])) if eligible else CONTROL
    checkpoint = json.loads(CHECKPOINT_DECISION.read_text(encoding="utf-8"))
    decision = {"selected": chosen, "eligible": eligible,
                "selected_checkpoint": checkpoint["selected_checkpoint"],
                "checkpoint_model_sha256": checkpoint["selected_model_sha256"],
                "dev_sha256": sha256_file(source), "dev_summary": rows,
                "dev_comparisons": comparisons,
                "rule": "positive lower paired text-cluster bootstrap bound versus "
                        "submitted blend on union/dev and no mean loss on "
                        "benchmark_plus_positives/dev; among eligible highest union/dev"}
    FINAL_DECISION.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"selected": chosen, "eligible": eligible,
                      "checkpoint": checkpoint["selected_checkpoint"]}, indent=2), flush=True)
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dev", "test", "select"))
    mode = parser.parse_args().mode
    if mode == "select":
        select()
    else:
        evaluate(mode)
