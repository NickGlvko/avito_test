"""Evaluate six frozen B/D combinations on complete cached candidate pools."""
# Check the unchanged B and D outputs before comparing six fixed fusions.
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
from .semantic_final import ALPHAS, BETAS, CONTROL, VARIANTS, blend_d, rrf_bd, select
from .semantic_full_d_evaluate import SCHEMES, d_chunk, d_folder
from .semantic_report import compare
from .semantic_tune_evaluate import cached_chunks, cached_folder, query_context


OUT = Path("artifacts/semantic_final")
DECISION = OUT / "decision.json"
B_MODEL = Path("models/semantic_full_v1/ranker.cbm")
D_MODEL = Path("models/semantic_full_D_v1/ranker.cbm")
DB_MODEL = Path("artifacts/semantic_full_d/full_balanced.cbm")
B_CONFIG = Path("models/semantic_full_v1/config.json")
D_CONFIG = Path("models/semantic_full_D_v1/config.json")
DB_CONFIG = DB_MODEL.with_suffix(".json")
SUBMITTED = Path("answer_semantic_full_D.csv")


def frozen_models() -> dict[str, CatBoostRanker]:
    b_config = json.loads(B_CONFIG.read_text(encoding="utf-8"))
    d_config = json.loads(D_CONFIG.read_text(encoding="utf-8"))
    db_config = json.loads(DB_CONFIG.read_text(encoding="utf-8"))
    if (sha256_file(B_MODEL) != b_config["model_sha256"] or
            sha256_file(D_MODEL) != d_config["model_sha256"] or
            sha256_file(DB_MODEL) != db_config["model_sha256"] or
            sha256_file(SUBMITTED) != d_config["answer_sha256"] or
            d_config["selected"] != CONTROL):
        raise ValueError("Frozen B or D control weights or submitted CSV changed")
    result = {}
    for name, path in (("B", B_MODEL), ("D", D_MODEL), ("DB", DB_MODEL)):
        model = CatBoostRanker()
        model.load_model(str(path))
        result[name] = model
    return result


def choices(b_ids: np.ndarray, b_scores: np.ndarray, d_ids: np.ndarray,
            u_scores: np.ndarray, v_scores: np.ndarray,
            all_variants: bool) -> dict[str, np.ndarray]:
    """Select from complete lists; check both RRF endpoints on every query."""
    b_top = select(b_ids, b_scores)
    d_top = select(d_ids, u_scores)
    if (not np.array_equal(select(d_ids, rrf_bd(
            b_ids, b_scores, d_ids, u_scores, 0.0)), b_top) or
            not np.array_equal(select(d_ids, rrf_bd(
                b_ids, b_scores, d_ids, u_scores, 1.0)), d_top) or
            not np.array_equal(select(d_ids, blend_d(
                u_scores, v_scores, 1.0)), d_top)):
        raise ValueError("B/D RRF or D blend endpoint differs from frozen model")
    result = {CONTROL: d_top, "B_extreme": b_top}
    if all_variants:
        for alpha in ALPHAS:
            result[f"d_blend_{alpha:.2f}"] = select(
                d_ids, blend_d(u_scores, v_scores, alpha))
        for beta in BETAS:
            result[f"rrf_bd_{beta:.2f}"] = select(
                d_ids, rrf_bd(b_ids, b_scores, d_ids, u_scores, beta))
    return result


def summarize(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.group_by("scheme", "variant").agg(
        pl.len().alias("queries"),
        pl.col("recall@50").mean().alias("recall50"),
        pl.col("candidate_recall").mean().alias("pool_recall"),
        pl.col("lost_at_selection").sum().alias("selection_losses"),
        pl.col("same_location_recall@50").mean().alias("same_location_recall50"),
        pl.col("other_location_recall@50").mean().alias("other_location_recall50"),
        pl.col("semantic_only_in_top50").sum().alias("semantic_only_in_top50"),
        pl.col("positive_regained").sum().alias("positive_regained"),
        pl.col("positive_lost").sum().alias("positive_lost"),
        pl.col("regained_macro").mean().alias("regained_macro"),
        pl.col("lost_macro").mean().alias("lost_macro"),
    ).sort("scheme", "variant")


def validate_controls(frame: pl.DataFrame, stage: str) -> None:
    reference = pl.read_parquet(Path("artifacts/semantic_full_d/evaluation") /
                                stage / "per_query.parquet")
    for name, archived in ((CONTROL, "D_full_unweighted"),
                           ("B_extreme", "B_full_balanced")):
        current = frame.filter(pl.col("variant") == name)
        original = reference.filter(pl.col("variant") == archived)
        paired = current.select("group_id", "scheme",
                                pl.col("recall@50").alias("actual"),
                                pl.col("candidate_recall").alias("actual_pool")).join(
            original.select("group_id", "scheme",
                            pl.col("recall@50").alias("expected"),
                            pl.col("candidate_recall").alias("expected_pool")),
            on=("group_id", "scheme"), validate="1:1")
        expected_rows = 2 * (35652 if stage == "dev" else 35376)
        if (paired.height != expected_rows or
                not np.array_equal(paired["actual"].to_numpy(),
                                   paired["expected"].to_numpy()) or
                not np.array_equal(paired["actual_pool"].to_numpy(),
                                   paired["expected_pool"].to_numpy())):
            raise ValueError(f"Frozen {name} was not reproduced on {stage}")


def evaluate(stage: str) -> dict:
    if stage not in ("dev", "test"):
        raise ValueError("Only complete dev and frozen diagnostic test are supported")
    if stage == "test":
        decision = json.loads(DECISION.read_text(encoding="utf-8"))
        if (decision["selected"] == CONTROL or
                sha256_file(OUT / "evaluation/dev/per_query.parquet") !=
                decision["dev_sha256"]):
            raise ValueError("Final test requires a frozen improving dev choice")
    started = time.perf_counter()
    models = frozen_models()
    queries, signals = query_context(stage)
    b_folder = cached_folder(stage, queries)
    d_folder_path = d_folder(stage, queries)
    output = OUT / "evaluation" / stage
    output.mkdir(parents=True, exist_ok=True)
    settings = digest({
        "stage": stage,
        "b_cache": sha256_file(b_folder / "manifest.json"),
        "d_cache": sha256_file(d_folder_path / "manifest.json"),
        "models": {name: sha256_file(path) for name, path in
                   (("B", B_MODEL), ("D", D_MODEL), ("DB", DB_MODEL))},
        "code": {name: sha256_file(Path(__file__).with_name(name)) for name in
                 ("semantic_final.py", "semantic_final_evaluate.py")},
    })
    predict_seconds = {name: 0.0 for name in models}
    expected_names = (CONTROL, "B_extreme", *(VARIANTS[1:] if stage == "dev" else
                      (decision["selected"],)))
    for number, chunk, b in cached_chunks(b_folder, queries):
        target = output / f"part_{number:05d}.parquet"
        mark = target.with_suffix(".json")
        if target.exists() and mark.exists():
            saved = json.loads(mark.read_text(encoding="utf-8"))
            if (saved["settings"] != settings or saved["queries"] != len(chunk) or
                    saved["part_sha256"] != sha256_file(target)):
                raise ValueError(f"Stale final evaluation part: {target}")
            continue
        d = d_chunk(d_folder_path, number, chunk)
        score = {}
        for name, model, features in (("B", models["B"], b["features"]),
                                      ("D", models["D"], d["features"]),
                                      ("DB", models["DB"], d["features"])):
            tick = time.perf_counter()
            score[name] = np.asarray(model.predict(features, thread_count=6),
                                     dtype=np.float64)
            predict_seconds[name] += time.perf_counter() - tick
        rows = []
        empty = np.empty(0, dtype=np.int32)
        for position, query in enumerate(chunk):
            truth = set(map(int, query["relevant_rows"]))
            for scheme_index, scheme in enumerate(SCHEMES):
                pair = 2 * position + scheme_index
                bb, be = map(int, b["offsets"][pair:pair+2])
                db, de = map(int, d["offsets"][pair:pair+2])
                b_ids = b["item_rows"][bb:be]
                d_ids = d["item_rows"][db:de]
                if not np.array_equal(b_ids, d_ids[d["in_old_pool"][db:de]]):
                    raise ValueError("B and D candidate pools changed")
                ranked = choices(b_ids, score["B"][bb:be], d_ids,
                                 score["D"][db:de], score["DB"][db:de],
                                 stage == "dev")
                if stage == "test":
                    selected = decision["selected"]
                    if selected not in ranked:
                        # Compute just the one frozen combination on test.
                        if selected.startswith("d_blend_"):
                            alpha = float(selected.removeprefix("d_blend_"))
                            ranked[selected] = select(d_ids, blend_d(
                                score["D"][db:de], score["DB"][db:de], alpha))
                        elif selected.startswith("rrf_bd_"):
                            beta = float(selected.removeprefix("rrf_bd_"))
                            ranked[selected] = select(d_ids, rrf_bd(
                                b_ids, score["B"][bb:be], d_ids,
                                score["D"][db:de], beta))
                        else:
                            raise ValueError("Unknown frozen final choice")
                d_top = set(map(int, ranked[CONTROL]))
                semantic_only = truth & (set(map(int, d_ids)) - set(map(int, b_ids)))
                for name in expected_names:
                    selected = set(map(int, ranked[name]))
                    pool = b_ids if name == "B_extreme" else d_ids
                    gained = len((truth & selected) - d_top)
                    lost = len((truth & d_top) - selected)
                    metrics = candidate_metrics(ranked[name], pool, truth, signals,
                                                query, empty, set())
                    metrics.update({
                        "stage": stage, "scheme": scheme, "variant": name,
                        "group_id": query["group_id"], "text": query["text"],
                        "semantic_only_in_top50": len(semantic_only & selected),
                        "positive_regained": gained, "positive_lost": lost,
                        "regained_macro": gained / len(truth),
                        "lost_macro": lost / len(truth),
                    })
                    rows.append(metrics)
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(target)
        mark.write_text(json.dumps({"settings": settings,
                                    "queries": len(chunk),
                                    "part_sha256": sha256_file(target)}),
                        encoding="utf-8")
        if (number + 1) % 20 == 0:
            print(f"Final {stage}: {min((number+1)*64, len(queries))}/"
                  f"{len(queries)}; {time.perf_counter()-started:.1f}s",
                  flush=True)
    frame = pl.read_parquet(str(output / "part_*.parquet"))
    expected = 2 * len(queries) * len(expected_names)
    if (frame.height != expected or
            frame.unique(("group_id", "scheme", "variant")).height != expected):
        raise ValueError("Incomplete final evaluation")
    validate_controls(frame, stage)
    frame.sort("group_id", "scheme", "variant").write_parquet(
        output / "per_query.parquet")
    table = summarize(frame)
    table.write_csv(output / "summary.csv")
    memory = psutil.Process().memory_info()
    timing = {"stage": stage, "seconds": time.perf_counter()-started,
              "predict_seconds": predict_seconds,
              "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
              "controls_reproduced_per_query": True,
              "per_query_sha256": sha256_file(output / "per_query.parquet")}
    (output / "timing.json").write_text(json.dumps(timing, indent=2),
                                         encoding="utf-8")
    print(table, flush=True)
    return timing


def choose() -> dict:
    source = OUT / "evaluation/dev/per_query.parquet"
    frame = pl.read_parquet(source)
    if (frame["stage"].unique().to_list() != ["dev"] or
            frame.filter(pl.col("variant") == CONTROL)["group_id"].n_unique() !=
            35652):
        raise ValueError("Only complete dev can choose the final combination")
    summary = summarize(frame)
    rows = {scheme: {r["variant"]: r for r in
                     summary.filter(pl.col("scheme") == scheme).to_dicts()}
            for scheme in SCHEMES}
    comparisons = {scheme: {name: compare(frame, scheme, name, CONTROL)
                            for name in VARIANTS[1:]} for scheme in SCHEMES}
    eligible = [name for name in VARIANTS[1:] if
                comparisons["union"][name]["ci95"][0] > 0 and
                rows["benchmark_plus_positives"][name]["recall50"] >=
                rows["benchmark_plus_positives"][CONTROL]["recall50"]]
    selected = (max(eligible, key=lambda name: rows["union"][name]["recall50"])
                if eligible else CONTROL)
    result = {"selected": selected, "eligible": eligible,
              "dev_sha256": sha256_file(source), "summary": rows,
              "comparisons": comparisons,
              "submitted_answer_sha256": sha256_file(SUBMITTED),
              "models": {name: sha256_file(path) for name, path in
                         (("B", B_MODEL), ("D", D_MODEL), ("DB", DB_MODEL))},
              "rule": "Positive lower paired text-cluster bootstrap bound "
                      "versus submitted D on union/dev; no mean loss on "
                      "benchmark_plus_positives/dev; select highest "
                      "union/dev among eligible six fixed variants"}
    OUT.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "eligible": eligible}, indent=2),
          flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dev", "choose", "test"))
    mode = parser.parse_args().mode
    choose() if mode == "choose" else evaluate(mode)
