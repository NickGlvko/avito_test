"""Save the selected ranker and its supervised signals outside ephemeral caches."""
# Freeze model, signals and input hashes together for repeatable prediction.
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .data import SEED, sha256_file, source_manifest, write_json
from .ranker import FEATURES, FEATURE_VERSION, POOL_SOURCES, TEXT_FEATURES
from .retrieval import DEPTH, INDEX_CONFIG


def save_ranker_bundle(run: Path, destination: Path, answer: Path, data_dir: Path) -> dict:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    decision = json.loads((run / "dev_decision.json").read_text(encoding="utf-8"))
    models = json.loads((run / "models.json").read_text(encoding="utf-8"))
    prediction = json.loads((run / (answer.stem + "_manifest.json")).read_text(encoding="utf-8"))
    if sha256_file(run / "dev" / "per_query.parquet") != decision["dev_metrics_sha256"]:
        raise ValueError("Dev decision does not match its evaluation data")
    if source_manifest(data_dir) != summary["data"] or prediction["sources"] != summary["data"]:
        raise ValueError("Source Parquet files differ from the selected run")
    if prediction["variant"] != decision["selected_variant"] or sha256_file(answer) != prediction["answer_sha256"]:
        raise ValueError("Answer does not match the frozen dev choice")
    for filename, expected in summary["code"].items():
        if sha256_file(Path(__file__).with_name(filename)) != expected:
            raise ValueError(f"Feature or retrieval code changed: {filename}")
    if models["feature_sets"]["text"] != TEXT_FEATURES or models["feature_sets"]["history"] != FEATURES:
        raise ValueError("Saved model features differ from current code")
    if decision["selected_variant"] != "catboost_full_no_history_features":
        raise ValueError("This bundle is for the submitted text-feature model on the full pool")
    artifacts = {"text_model.cbm": models["model_sha256"]["text"],
                 "history_model.cbm": models["model_sha256"]["history"],
                 "all_train_signals.joblib": prediction["all_train_signals_sha256"]}
    for filename, expected in artifacts.items():
        if sha256_file(run / filename) != expected:
            raise ValueError(f"Training artifact changed: {filename}")
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected in artifacts.items():
        target = destination / filename
        if target.exists() and sha256_file(target) != expected:
            raise ValueError(f"Bundle already has conflicting {filename}")
        if not target.exists():
            shutil.copy2(run / filename, target)
        if sha256_file(target) != expected:
            raise ValueError(f"Copied artifact hash mismatch: {filename}")
    config = {"format_version": 1, "selected_variant": decision["selected_variant"],
              "selection": decision, "answer_sha256": prediction["answer_sha256"],
              "query_count": prediction["validation"]["queries"],
              "source_manifest": summary["data"], "code_sha256": summary["code"],
              "prediction_code_sha256": sha256_file(Path(__file__).with_name("ranker_predict.py")),
              "uv_lock_sha256": sha256_file(Path("uv.lock")),
              "artifacts_sha256": artifacts, "model_params": models["params"],
              "model_features": models["feature_sets"], "feature_version": FEATURE_VERSION,
              "retrieval": {"index_config": INDEX_CONFIG, "source_depth": DEPTH,
                            "pool_sources": POOL_SOURCES}, "seed": SEED,
              "training_signals": "all_train_signals.joblib; built from all train rows",
              "corpus": "benchmark_items for candidates; union index for text statistics"}
    write_json(destination / "config.json", config)
    return config
