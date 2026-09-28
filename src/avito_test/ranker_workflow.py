"""Run dev, freeze the transfer-aware choice, then open diagnostic test."""
# Run selection and diagnostic test in that order to avoid test-driven tuning.
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .ranker import run_ranker
from .ranker_selection import select_robust_dev


def run_ranker_workflow(data_dir: Path, output: Path, train: pl.DataFrame,
                        train_queries: int, dev_limit: int, workers: int) -> dict:
    run_ranker(data_dir, output, train, train_queries, dev_limit, workers, include_test=False)
    run = Path(json.loads((output / "ranker" / "latest.json").read_text(encoding="utf-8"))["run"])
    if "smoke" not in run.parts and not (run / "dev_decision.json").exists():
        select_robust_dev(run)
    return run_ranker(data_dir, output, train, train_queries, dev_limit, workers, include_test=True)
