"""Freeze the dev-chosen tree checkpoint before comparing final blends."""
# Freeze tree depth and count before inspecting diagnostic test.
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from catboost import CatBoostRanker

from .data import sha256_file
from .semantic_tune_cache import OUT
from .semantic_tune_evaluate import CONTROL, summary


CHECKPOINT_FRAME = OUT / "checkpoints/per_query.parquet"
CHECKPOINT_DECISION = OUT / "checkpoint_decision.json"
SELECTED_MODEL = OUT / "selected_new_ranker.cbm"


def select_checkpoint() -> dict:
    frame = pl.read_parquet(CHECKPOINT_FRAME)
    if (frame.filter(pl.col("variant") == CONTROL)["group_id"].n_unique() != 35652 or
            frame["stage"].unique().to_list() != ["dev"]):
        raise ValueError("Checkpoint must be chosen on the complete dev split")
    rows = {scheme: {row["variant"]: row for row in
                     summary(frame).filter(pl.col("scheme") == scheme).to_dicts()}
            for scheme in ("union", "benchmark_plus_positives")}
    variants = [f"d{depth}_t{trees}" for depth in (6, 8)
                for trees in (240, 480, 720, 960)]
    for scheme, expected in (("union", 0.881835),
                             ("benchmark_plus_positives", 0.926889)):
        if any(abs(rows[scheme][name]["pool_recall"] - expected) > 0.00001
               for name in variants + [CONTROL]):
            raise ValueError("Frozen B pool completeness changed")
    eligible = [name for name in variants if
                rows["benchmark_plus_positives"][name]["recall50"] >=
                rows["benchmark_plus_positives"][CONTROL]["recall50"]]
    choices = eligible or variants
    # Resolve exact ties toward the smaller, shallower model.
    chosen = max(choices, key=lambda name: (
        rows["union"][name]["recall50"],
        rows["benchmark_plus_positives"][name]["recall50"],
        -int(name.split("_t")[1]), -int(name.split("_t")[0][1:])))
    depth, trees = (int(chosen.split("_t")[0][1:]),
                    int(chosen.split("_t")[1]))
    training = json.loads((OUT / f"depth_{depth}_960.json").read_text(encoding="utf-8"))
    if sha256_file(Path(training["model"])) != training["model_sha256"]:
        raise ValueError("Chosen 960-tree checkpoint source changed")
    model = CatBoostRanker()
    model.load_model(training["model"])
    model.shrink(ntree_end=trees)
    model.save_model(str(SELECTED_MODEL))
    decision = {"selected_checkpoint": chosen, "depth": depth, "trees": trees,
                "eligible_augmented": eligible,
                "rule": "highest union/dev Recall@50 among checkpoints with no mean "
                        "benchmark_plus_positives/dev loss versus submitted blend; "
                        "if none qualify, highest union/dev; exact ties prefer fewer trees and depth",
                "checkpoint_frame_sha256": sha256_file(CHECKPOINT_FRAME),
                "full_model_sha256": training["model_sha256"],
                "selected_model_sha256": sha256_file(SELECTED_MODEL),
                "union_dev": rows["union"][chosen]["recall50"],
                "augmented_dev": rows["benchmark_plus_positives"][chosen]["recall50"],
                "control_union_dev": rows["union"][CONTROL]["recall50"],
                "control_augmented_dev":
                    rows["benchmark_plus_positives"][CONTROL]["recall50"]}
    CHECKPOINT_DECISION.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    return decision


if __name__ == "__main__":
    select_checkpoint()
