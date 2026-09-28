"""Evaluate the two full-pool B rankers on frozen complete dev/test candidate pools."""
# Score complete dev pools even when some queries have no retrieved positive.
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
from .semantic_report import compare
from .semantic_tune_cache import OUT as TUNE_OUT
from .semantic_tune_evaluate import (CONTROL, append_metrics, cached_chunks,
                                     cached_folder, controls, query_context,
                                     summary, validate_controls)
from .semantic_full_oof import OUT


VARIANTS = ("full_unweighted", "full_balanced")
DECISION = OUT / "decision.json"


def evaluate(stage: str) -> dict:
    if stage not in ("dev", "test"):
        raise ValueError("Only dev and diagnostic test are supported")
    if stage == "test":
        decision = json.loads(DECISION.read_text(encoding="utf-8"))
        if (decision["selected"] == CONTROL or
                sha256_file(OUT / "evaluation/dev/per_query.parquet") != decision["dev_sha256"]):
            raise ValueError("Test requires a frozen improving dev choice")
    started = time.perf_counter()
    queries, signals = query_context(stage)
    folder = cached_folder(stage, queries)
    models = {}
    hashes = {}
    for name in VARIANTS:
        report = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
        path = Path(report["model"])
        if sha256_file(path) != report["model_sha256"]:
            raise ValueError(f"Full-pool model weights changed: {name}")
        model = CatBoostRanker()
        model.load_model(str(path))
        models[name] = model
        hashes[name] = report["model_sha256"]
    output = OUT / "evaluation" / stage
    output.mkdir(parents=True, exist_ok=True)
    settings = digest({"stage": stage, "cache_sha256": sha256_file(
        folder / "manifest.json"), "models": hashes,
        "code_sha256": sha256_file(Path(__file__))})
    model_seconds = {name: 0. for name in VARIANTS}
    for number, chunk, arrays in cached_chunks(folder, queries):
        path = output / f"part_{number:05d}.parquet"
        mark = path.with_suffix(".json")
        if path.exists() and mark.exists():
            saved = json.loads(mark.read_text(encoding="utf-8"))
            if (saved["queries"] != len(chunk) or saved["settings"] != settings or
                    saved["part_sha256"] != sha256_file(path)):
                raise ValueError(f"Damaged full-pool evaluation part: {path}")
            continue
        scores = controls(arrays)
        for name, model in models.items():
            tick = time.perf_counter()
            scores[name] = np.asarray(model.predict(arrays["features"], thread_count=6),
                                      dtype=np.float64)
            model_seconds[name] += time.perf_counter()-tick
        rows: list[dict] = []
        append_metrics(rows, chunk, arrays, scores, signals, stage)
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(path)
        mark.write_text(json.dumps({"queries": len(chunk), "settings": settings,
                                    "part_sha256": sha256_file(path)}), encoding="utf-8")
        if (number+1) % 20 == 0:
            print(f"Full-pool {stage}: {min((number+1)*64, len(queries))}/"
                  f"{len(queries)}; {time.perf_counter()-started:.1f}s", flush=True)
    frame = pl.read_parquet(str(output / "part_*.parquet"))
    expected = len(queries)*2*(3+len(VARIANTS))
    if frame.height != expected or frame.unique(["group_id", "scheme", "variant"]).height != expected:
        raise ValueError("Incomplete full-pool model evaluation")
    validate_controls(frame, stage)
    frame.sort("group_id", "scheme", "variant").write_parquet(output / "per_query.parquet")
    table = summary(frame)
    table.write_csv(output / "summary.csv")
    memory = psutil.Process().memory_info()
    timing = {"stage": stage, "settings": settings,
              "seconds": time.perf_counter()-started,
              "model_predict_seconds": model_seconds,
              "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
              "cache": str(folder), "models": hashes,
              "per_query_sha256": sha256_file(output / "per_query.parquet"),
              "controls_reproduced_per_query": True}
    (output / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(table, flush=True)
    return timing


def select() -> dict:
    source = OUT / "evaluation/dev/per_query.parquet"
    frame = pl.read_parquet(source)
    if (frame["stage"].unique().to_list() != ["dev"] or
            frame.filter(pl.col("variant") == CONTROL)["group_id"].n_unique() != 35652):
        raise ValueError("Only full dev can select a full-pool ranker")
    table = summary(frame)
    rows = {scheme: {row["variant"]: row for row in
                     table.filter(pl.col("scheme") == scheme).to_dicts()}
            for scheme in ("union", "benchmark_plus_positives")}
    pairs = {scheme: {name: compare(frame, scheme, name, CONTROL)
                      for name in VARIANTS} for scheme in rows}
    eligible = [name for name in VARIANTS if
                pairs["union"][name]["ci95"][0] > 0 and
                rows["benchmark_plus_positives"][name]["recall50"] >=
                rows["benchmark_plus_positives"][CONTROL]["recall50"]]
    chosen = max(eligible, key=lambda name: (
        rows["union"][name]["recall50"],
        rows["benchmark_plus_positives"][name]["recall50"])) if eligible else CONTROL
    decision = {"selected": chosen, "eligible": eligible,
                "dev_sha256": sha256_file(source),
                "dev_summary": rows, "dev_comparisons": pairs,
                "control_answer_sha256": "8124b99e143fcc1091705be80e9bc4c2b62a42c3cf600909ae352afdf54ee707",
                "models": {name: sha256_file(OUT / f"{name}.cbm") for name in VARIANTS},
                "rule": "positive lower paired text-cluster bootstrap bound versus "
                        "submitted blend on union/dev, no mean loss on "
                        "benchmark_plus_positives/dev; among eligible highest union/dev"}
    DECISION.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"selected": chosen, "eligible": eligible}, indent=2), flush=True)
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dev", "test", "select"))
    mode = parser.parse_args().mode
    if mode == "select":
        select()
    else:
        evaluate(mode)
