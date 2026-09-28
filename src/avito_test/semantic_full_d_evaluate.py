"""Compare complete D rankers with the submitted complete-B ranker."""
# Keep the complete D pool unchanged across ranker comparisons.
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
from .ranker import candidate_metrics
from .semantic_blend import rank_all
from .semantic_full_d_cache import CHUNK, DIM, cache_signature
from .semantic_full_d_oof import OUT
from .semantic_report import compare
from .semantic_tune_evaluate import cached_chunks, cached_folder, query_context


CONTROL = "B_full_balanced"
NEW = ("D_full_unweighted", "D_full_balanced")
VARIANTS = (CONTROL, "D15000", *NEW)
SCHEMES = ("union", "benchmark_plus_positives")
DECISION = OUT / "decision.json"
B_MODEL = Path("models/semantic_full_v1/ranker.cbm")
B_REPORT = Path("models/semantic_full_v1/config.json")


def d_folder(stage: str, queries: list[dict]) -> Path:
    folder = OUT / "cache" / stage / cache_signature(stage, queries)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["stage"] != stage or manifest["queries"] != len(queries) or
            manifest["feature_count"] != DIM or manifest["signature"] != folder.name):
        raise ValueError("Frozen D cache manifest changed")
    return folder


def d_chunk(folder: Path, number: int, queries: list[dict]) -> dict:
    path = folder / f"part_{number:05d}.npz"
    mark = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (mark["queries"] != len(queries) or
            mark["first_group"] != queries[0]["group_id"] or
            mark["last_group"] != queries[-1]["group_id"] or
            mark["part_sha256"] != sha256_file(path)):
        raise ValueError(f"Damaged D cache part: {path}")
    with np.load(path) as data:
        result = {name: data[name] for name in
                  ("features", "item_rows", "in_old_pool", "D15000", "offsets")}
    if (result["features"].shape != (mark["rows"], DIM) or
            len(result["offsets"]) != 2*len(queries)+1 or
            result["offsets"][-1] != mark["rows"]):
        raise ValueError("D feature cache shape changed")
    return result


def summarize(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.group_by("scheme", "variant").agg(
        pl.len().alias("queries"),
        pl.col("recall@50").mean().alias("recall50"),
        pl.col("candidate_recall").mean().alias("pool_recall"),
        pl.col("lost_at_selection").sum().alias("selection_losses"),
        pl.col("same_location_recall@50").mean().alias("same_location_recall50"),
        pl.col("other_location_recall@50").mean().alias("other_location_recall50"),
        pl.col("semantic_only_in_top50").sum().alias("new_semantic_positives_selected"),
        pl.col("old_positive_gained").sum().alias("old_positive_regained"),
        pl.col("old_positive_lost").sum().alias("old_positive_lost"),
        pl.col("new_macro_gain").mean().alias("new_macro_gain"),
        pl.col("old_macro_gain").mean().alias("old_macro_gain"),
        pl.col("old_macro_loss").mean().alias("old_macro_loss"),
    ).sort("scheme", "variant")


def validate_controls(frame: pl.DataFrame, stage: str) -> None:
    b_reference = pl.read_parquet(
        Path("artifacts/semantic_full_pool/evaluation") / stage / "per_query.parquet"
    ).filter(pl.col("variant") == "full_balanced")
    current = frame.filter(pl.col("variant") == CONTROL)
    paired = current.select("group_id", "scheme",
                            pl.col("recall@50").alias("new"),
                            pl.col("candidate_recall").alias("new_pool")).join(
        b_reference.select("group_id", "scheme",
                           pl.col("recall@50").alias("old"),
                           pl.col("candidate_recall").alias("old_pool")),
        on=("group_id", "scheme"), validate="1:1"
    )
    expected = (35652 if stage == "dev" else 35376)*2
    if (paired.height != expected or
            not np.array_equal(paired["new"].to_numpy(), paired["old"].to_numpy()) or
            not np.array_equal(paired["new_pool"].to_numpy(),
                               paired["old_pool"].to_numpy())):
        raise ValueError("Submitted B_full_balanced control was not reproduced")
    if stage == "dev":
        analysis = json.loads(Path("artifacts/semantic_scale/analysis.json")
                              .read_text(encoding="utf-8"))
        original = pl.read_parquet(Path(analysis["dev_path"]) / "per_query.parquet")
        original = original.filter(pl.col("variant") == "D15000")
        current = frame.filter(pl.col("variant") == "D15000")
        paired = current.select("group_id", "scheme",
                                pl.col("recall@50").alias("new"),
                                pl.col("candidate_recall").alias("new_pool")).join(
            original.select("group_id", "scheme",
                            pl.col("recall@50").alias("old"),
                            pl.col("candidate_recall").alias("old_pool")),
            on=("group_id", "scheme"), validate="1:1"
        )
        if (paired.height != expected or
                not np.array_equal(paired["new"].to_numpy(), paired["old"].to_numpy()) or
                not np.array_equal(paired["new_pool"].to_numpy(),
                                   paired["old_pool"].to_numpy())):
            raise ValueError("D15000 control or expanded candidate pool changed")


def evaluate(stage: str) -> dict:
    if stage not in ("dev", "test"):
        raise ValueError("Only full dev and frozen diagnostic test are supported")
    if stage == "test":
        decision = json.loads(DECISION.read_text(encoding="utf-8"))
        if (decision["selected"] == CONTROL or
                sha256_file(OUT / "evaluation/dev/per_query.parquet") !=
                decision["dev_sha256"]):
            raise ValueError("D test requires a frozen improving dev choice")
        active_new = (decision["selected"],)
        variants = (CONTROL, *active_new)
    else:
        active_new = NEW
        variants = VARIANTS
    started = time.perf_counter()
    queries, signals = query_context(stage)
    b_cache = cached_folder(stage, queries)
    d_cache = d_folder(stage, queries)
    b_report = json.loads(B_REPORT.read_text(encoding="utf-8"))
    if sha256_file(B_MODEL) != b_report["model_sha256"]:
        raise ValueError("Submitted B_full_balanced weights changed")
    model_files = {CONTROL: B_MODEL}
    for name in active_new:
        path = OUT / f"{name.removeprefix('D_')}.cbm"
        record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if sha256_file(path) != record["model_sha256"]:
            raise ValueError(f"Full D model weights changed: {name}")
        model_files[name] = path
    models = {}
    for name, path in model_files.items():
        model = CatBoostRanker()
        model.load_model(str(path))
        models[name] = model
    output = OUT / "evaluation" / stage
    output.mkdir(parents=True, exist_ok=True)
    settings = digest({
        "stage": stage, "b_cache_sha256": sha256_file(b_cache / "manifest.json"),
        "d_cache_sha256": sha256_file(d_cache / "manifest.json"),
        "models": {name: sha256_file(path) for name, path in model_files.items()},
        "code_sha256": sha256_file(Path(__file__)),
    })
    predict_seconds = {name: 0. for name in model_files}
    for number, chunk, b in cached_chunks(b_cache, queries):
        target = output / f"part_{number:05d}.parquet"
        mark = target.with_suffix(".json")
        if target.exists() and mark.exists():
            saved = json.loads(mark.read_text(encoding="utf-8"))
            if (saved["settings"] != settings or saved["queries"] != len(chunk) or
                    saved["part_sha256"] != sha256_file(target)):
                raise ValueError(f"Stale D evaluation part: {target}")
            continue
        d = d_chunk(d_cache, number, chunk)
        tick = time.perf_counter()
        b_scores = np.asarray(models[CONTROL].predict(
            b["features"], thread_count=6), dtype=np.float64)
        predict_seconds[CONTROL] += time.perf_counter()-tick
        d_scores = {"D15000": d["D15000"]}
        for name in active_new:
            tick = time.perf_counter()
            d_scores[name] = np.asarray(models[name].predict(
                d["features"], thread_count=6), dtype=np.float64)
            predict_seconds[name] += time.perf_counter()-tick
        records = []
        empty = np.empty(0, dtype=np.int32)
        for position, query in enumerate(chunk):
            truth = set(map(int, query["relevant_rows"]))
            relevant_count = len(truth)
            for scheme_index, scheme in enumerate(SCHEMES):
                pair = 2*position+scheme_index
                bb, be = map(int, b["offsets"][pair:pair+2])
                db, de = map(int, d["offsets"][pair:pair+2])
                b_ids = b["item_rows"][bb:be]
                d_ids = d["item_rows"][db:de]
                old_mask = d["in_old_pool"][db:de]
                if not np.array_equal(b_ids, d_ids[old_mask]):
                    raise ValueError("Old B pool differs inside expanded D pool")
                b_selected = set(map(int, rank_all(b_ids, b_scores[bb:be])[:50]))
                old_set = set(map(int, b_ids))
                new_positive = truth & (set(map(int, d_ids)) - old_set)
                for name in variants:
                    if name == CONTROL:
                        selected = b_selected
                        candidates = b_ids
                    else:
                        selected = set(map(int, rank_all(
                            d_ids, d_scores[name][db:de])[:50]))
                        candidates = d_ids
                    new_selected = len(new_positive & selected)
                    old_gained = len((truth & old_set & selected) - b_selected)
                    old_lost = len((truth & b_selected) - selected)
                    baseline_recall = len(truth & b_selected)/relevant_count
                    expected_recall = baseline_recall + (
                        new_selected + old_gained - old_lost)/relevant_count
                    if abs(expected_recall - len(truth & selected)/relevant_count) > 1e-12:
                        raise ValueError("Positive gain/loss decomposition does not close")
                    row = candidate_metrics(
                        np.asarray(sorted(selected), dtype=np.int32), candidates,
                        truth, signals, query, empty, set())
                    row.update({
                        "stage": stage, "scheme": scheme, "variant": name,
                        "group_id": query["group_id"], "text": query["text"],
                        "semantic_only_positives": len(new_positive),
                        "semantic_only_in_top50": new_selected,
                        "old_positive_gained": old_gained,
                        "old_positive_lost": old_lost,
                        "new_macro_gain": new_selected/relevant_count,
                        "old_macro_gain": old_gained/relevant_count,
                        "old_macro_loss": old_lost/relevant_count,
                    })
                    records.append(row)
        pl.DataFrame(records, infer_schema_length=None).write_parquet(target)
        mark.write_text(json.dumps({
            "settings": settings, "queries": len(chunk),
            "part_sha256": sha256_file(target)
        }), encoding="utf-8")
        if (number+1) % 20 == 0:
            print(f"Full D {stage}: {min((number+1)*CHUNK, len(queries))}/"
                  f"{len(queries)}; {time.perf_counter()-started:.1f}s",
                  flush=True)
    frame = pl.read_parquet(str(output / "part_*.parquet"))
    expected = len(queries)*2*len(variants)
    if (frame.height != expected or
            frame.unique(("group_id", "scheme", "variant")).height != expected):
        raise ValueError("Incomplete full D evaluation")
    validate_controls(frame, stage)
    frame.sort("group_id", "scheme", "variant").write_parquet(
        output / "per_query.parquet")
    table = summarize(frame)
    table.write_csv(output / "summary.csv")
    memory = psutil.Process().memory_info()
    timing = {
        "stage": stage, "seconds": time.perf_counter()-started,
        "model_predict_seconds": predict_seconds,
        "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
        "b_cache": str(b_cache), "d_cache": str(d_cache),
        "per_query_sha256": sha256_file(output / "per_query.parquet"),
        "controls_reproduced_per_query": True,
    }
    (output / "timing.json").write_text(json.dumps(timing, indent=2),
                                         encoding="utf-8")
    print(table, flush=True)
    return timing


def select() -> dict:
    source = OUT / "evaluation/dev/per_query.parquet"
    frame = pl.read_parquet(source)
    if (frame["stage"].unique().to_list() != ["dev"] or
            frame.filter(pl.col("variant") == CONTROL)["group_id"].n_unique() != 35652):
        raise ValueError("Only complete dev can select a full D ranker")
    table = summarize(frame)
    rows = {scheme: {row["variant"]: row for row in
                     table.filter(pl.col("scheme") == scheme).to_dicts()}
            for scheme in SCHEMES}
    pairs = {scheme: {name: compare(frame, scheme, name, CONTROL)
                      for name in NEW} for scheme in SCHEMES}
    eligible = [name for name in NEW if
                pairs["union"][name]["ci95"][0] > 0 and
                rows["benchmark_plus_positives"][name]["recall50"] >=
                rows["benchmark_plus_positives"][CONTROL]["recall50"]]
    if eligible:
        best_union = max(rows["union"][name]["recall50"] for name in eligible)
        near_best = [name for name in eligible if
                     best_union - rows["union"][name]["recall50"] <= 0.001]
        chosen = max(near_best, key=lambda name: (
            rows["benchmark_plus_positives"][name]["recall50"],
            rows["union"][name]["recall50"]))
    else:
        chosen = CONTROL
    decision = {
        "selected": chosen, "eligible": eligible,
        "dev_sha256": sha256_file(source),
        "dev_summary": rows, "dev_comparisons": pairs,
        "control_answer_sha256":
            "e8b467534723c5ed556fd0ae2dbef6e22a673b3c6acb2ad13028b97080b86dfd",
        "models": {name: sha256_file(OUT / f"{name.removeprefix('D_')}.cbm")
                   for name in NEW},
        "rule": "positive lower paired text-cluster bootstrap bound versus "
                "submitted B_full_balanced on union/dev, no mean loss on "
                "benchmark_plus_positives/dev; among eligible models within "
                "0.001 union/dev Recall@50 of the best, highest benchmark_plus_positives/dev",
        "near_best_union_tolerance": 0.001,
    }
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
