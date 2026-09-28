"""Freeze the submitted E5 control and select 15,000 leakage-free OOF groups."""
# Assign OOF folds deterministically for both B and D training.
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from .data import SEARCH, SEED, enrich_train, sha256_file, source_manifest
from .ranker import oof_queries, training_folds


OUT = Path("artifacts/semantic_scale")
CONTROL = OUT / "control_3000"
SELECTED = OUT / "selected_queries.parquet"
SELECTION_SEED = SEED + 15000


def freeze_control() -> dict:
    CONTROL.mkdir(parents=True, exist_ok=True)
    source_answer = Path("answer_semantic.csv")
    source_package = Path("models/semantic_v1")
    config = json.loads((source_package / "config.json").read_text(encoding="utf-8"))
    if (config["variant"] != "B" or
            sha256_file(source_answer) != config["answer_sha256"]):
        raise ValueError("Submitted B answer differs from its frozen package")
    target_answer = CONTROL / "answer_semantic.csv"
    if not target_answer.exists():
        shutil.copy2(source_answer, target_answer)
    if sha256_file(target_answer) != config["answer_sha256"]:
        raise ValueError("Existing control answer has a different SHA256")
    target_package = CONTROL / "package"
    if not target_package.exists():
        shutil.copytree(source_package, target_package)
    for name in ("config.json", "ranker.cbm", "training.json",
                 "item_id_to_vector_row.parquet"):
        if sha256_file(target_package / name) != sha256_file(source_package / name):
            raise ValueError(f"Control package changed: {name}")
    manifest = {"platform_recall50_user_reported": 0.732865,
                "answer_sha256": config["answer_sha256"],
                "package": {name: sha256_file(target_package / name) for name in
                            ("config.json", "ranker.cbm", "training.json",
                             "item_id_to_vector_row.parquet")},
                "model_b_3000_sha256": sha256_file(Path(
                    "artifacts/semantic/models/B_existing_pool_semantic_features.cbm")),
                "model_d_3000_sha256": sha256_file(Path(
                    "artifacts/semantic/models/D_expanded_pool_semantic_features.cbm")),
                "e5_revision": config["e5_revision"],
                "e5_weights_sha256": config["e5_sha256"],
                "union_dev_3000": {"B": 0.7381082997631614, "D": 0.7361839862687883},
                "augmented_dev_3000": {"B": 0.8034454402890765, "D": 0.8026496903998372}}
    (CONTROL / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def select_queries() -> dict:
    freeze_control()
    OUT.mkdir(parents=True, exist_ok=True)
    training = training_folds(enrich_train(pl.read_parquet(
        "src/avito_test/train.parquet")).filter(pl.col("split") < 8))
    grouped = training.group_by("group_id", *SEARCH).agg(
        pl.col("item_id").unique().sort().alias("relevant"),
        pl.col("text").first(), pl.col("fold").first())
    old = pl.DataFrame(oof_queries(training, 1000))
    saved = pl.read_parquet("artifacts/semantic/oof_replay/all_queries.parquet")
    if old.sort("group_id").to_dicts() != saved.sort("group_id").to_dicts():
        raise ValueError("The previous 3,000 OOF queries did not reproduce")
    if old.height != 3000 or grouped["group_id"].n_unique() != grouped.height:
        raise ValueError("Unexpected original OOF group count or duplicate group_id")
    old_ids = set(old["group_id"].to_list())
    remaining = grouped.filter(~pl.col("group_id").is_in(old_ids)).sort("group_id")
    generator = np.random.default_rng(SELECTION_SEED)
    selected_ids = set(remaining["group_id"].to_numpy()[
        generator.choice(remaining.height, size=12000, replace=False)].tolist())
    added = remaining.filter(pl.col("group_id").is_in(selected_ids))
    selected = pl.concat([old.with_columns(pl.lit(True).alias("is_control_3000")),
                          added.with_columns(pl.lit(False).alias("is_control_3000"))],
                         how="vertical_relaxed")
    selected = selected.sort("fold", "text", "group_id")
    if (selected.height != 15000 or selected["group_id"].n_unique() != 15000 or
            selected.filter(pl.col("is_control_3000")).height != 3000 or
            selected.filter(~pl.col("is_control_3000")).height != 12000):
        raise ValueError("The 15,000-group selection is incomplete")
    selected.write_parquet(SELECTED)
    report = {"selection_seed": SELECTION_SEED, "groups": selected.height,
              "reused_control_groups": 3000, "new_groups": 12000,
              "by_fold": {str(key): value for key, value in
                          selected.group_by("fold").len().iter_rows()},
              "selected_sha256": sha256_file(SELECTED),
              "original_oof_sha256": sha256_file(Path(
                  "artifacts/semantic/oof_replay/all_queries.parquet")),
              "sources": source_manifest(Path("src/avito_test"))}
    (OUT / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    select_queries()
