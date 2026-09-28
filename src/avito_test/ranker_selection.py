"""Freeze a transfer-aware ranker choice using dev only, before reading test."""
# Freeze the decision on dev before opening diagnostic test.
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .data import sha256_file, write_json

MODELS = ["catboost_text_pool", "catboost_full_no_history_features",
          "catboost_full_with_history_features"]
SECONDARY_TOLERANCE = .01


def select_robust_dev(run: Path) -> dict:
    path = run / "dev_decision.json"
    if path.exists():
        raise FileExistsError("Dev decision is already frozen")
    dev_path = run / "dev" / "per_query.parquet"
    frame = pl.read_parquet(dev_path).filter(pl.col("variant").is_in(MODELS))
    groups = frame["group_id"].n_unique()
    if not groups or frame.height != groups * 2 * len(MODELS):
        raise ValueError("Dev evaluation is incomplete")
    scores = frame.group_by("variant", "scheme").agg(pl.col("recall@50").mean())
    values = {(row["variant"], row["scheme"]): row["recall@50"] for row in scores.iter_rows(named=True)}
    secondary = "benchmark_plus_positives"
    best_secondary = max(values[(name, secondary)] for name in MODELS)
    eligible = [name for name in MODELS if values[(name, secondary)] >= best_secondary - SECONDARY_TOLERANCE]
    chosen = max(eligible, key=lambda name: (values[(name, "union")], name))
    union_only = max(MODELS, key=lambda name: (values[(name, "union")], name))
    decision = {"selection_basis": "dev only; test was not read",
                "rule": "maximize union/dev among models within 0.01 Recall@50 of the best model on benchmark_plus_positives/dev",
                "secondary_tolerance": SECONDARY_TOLERANCE,
                "selected_variant": chosen, "union_only_variant": union_only,
                "selected_union_dev_recall50": values[(chosen, "union")],
                "selected_augmented_dev_recall50": values[(chosen, secondary)],
                "union_only_union_dev_recall50": values[(union_only, "union")],
                "union_only_augmented_dev_recall50": values[(union_only, secondary)],
                "model_dev_scores": {name: {"union": values[(name, "union")],
                                             secondary: values[(name, secondary)]} for name in MODELS},
                "eligible_variants": eligible, "dev_groups": groups,
                "dev_metrics_sha256": sha256_file(dev_path),
                "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(path, decision)
    return decision
