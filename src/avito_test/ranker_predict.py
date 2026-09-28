"""Reproduce a selected ranker's submission against the benchmark corpus only."""
# Map ranked corpus rows back to original item_id before writing the answer.
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import psutil

from .data import sha256_file, source_manifest, write_json
from .extended_validation import prepare_union
from .ranker import FEATURES, HISTORY_FEATURES, TEXT_FEATURES, FeatureRetriever, load_feature_items, model_ranking, pool_ids
from .retrieval import TrainSignals
from .submission import write_answer


def predict_ranker(data_dir: Path, output: Path, train: pl.DataFrame | None,
                   answer: Path, run: Path | None = None,
                   package: Path | None = None) -> dict:
    total_start = time.perf_counter()
    if package is not None and run is not None:
        raise ValueError("Use either a saved package or a training run")
    saved = None
    if package is not None:
        saved_path = package / "config.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        variant = saved["selected_variant"]
        if saved["format_version"] != 1 or saved["feature_version"] != 1:
            raise ValueError("Unsupported ranker package version")
        if saved["model_features"]["text"] != TEXT_FEATURES or saved["model_features"]["history"] != FEATURES:
            raise ValueError("Package feature order differs from current code")
        if sha256_file(Path("uv.lock")) != saved["uv_lock_sha256"]:
            raise ValueError("uv.lock differs from the saved ranker environment")
        if sha256_file(Path(__file__)) != saved["prediction_code_sha256"]:
            raise ValueError("Prediction code differs from the saved ranker package")
        model_path = package / ("history_model.cbm" if variant == "catboost_full_with_history_features"
                                else "text_model.cbm")
        signals_path = package / "all_train_signals.joblib"
        for filename, expected in saved["artifacts_sha256"].items():
            if sha256_file(package / filename) != expected:
                raise ValueError(f"Package artifact hash differs: {filename}")
        expected_sources, expected_code = saved["source_manifest"], saved["code_sha256"]
        selection = saved["selection"]
        manifest_dir = output / "ranker" / "portable"
    else:
        if run is None:
            run = Path(json.loads((output / "ranker" / "latest.json").read_text(encoding="utf-8"))["run"])
        if "smoke" in run.parts:
            raise ValueError("A smoke-test model cannot create the submission")
        config = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        decision_path = run / "dev_decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else None
        if decision and sha256_file(run / "dev" / "per_query.parquet") != decision["dev_metrics_sha256"]:
            raise ValueError("Dev metrics differ from the frozen decision")
        variant = decision["selected_variant"] if decision else config["variant"]
        if config["text_features"] != TEXT_FEATURES or config["history_features"] != HISTORY_FEATURES:
            raise ValueError("Feature order differs from the selected run")
        model_path = run / ("history_model.cbm" if variant == "catboost_full_with_history_features"
                            else "text_model.cbm")
        signals_path = run / "all_train_signals.joblib"
        model_manifest = json.loads((run / "models.json").read_text(encoding="utf-8"))
        expected_hash = model_manifest["model_sha256"]["history" if variant == "catboost_full_with_history_features" else "text"]
        if sha256_file(model_path) != expected_hash:
            raise ValueError("Model weights differ from the selected run")
        expected_sources, expected_code = summary["data"], summary["code"]
        selection = decision or config
        manifest_dir = run
    if variant not in {"catboost_text_pool", "catboost_full_no_history_features",
                       "catboost_full_with_history_features"}:
        raise ValueError(f"Unsupported ranker variant: {variant}")
    sources = source_manifest(data_dir)
    if sources != expected_sources:
        raise ValueError("Data differ from the selected ranker run")
    for filename, expected in expected_code.items():
        if sha256_file(Path(__file__).with_name(filename)) != expected:
            raise ValueError(f"Retrieval or feature code differs from the selected run: {filename}")
    index_path, metadata_path, _ = prepare_union(data_dir, output / "expanded", sources)
    index = joblib.load(index_path, mmap_mode="r")
    metadata = pl.read_parquet(metadata_path)
    items = load_feature_items(data_dir, index_path.parent, index.item_ids, sources)
    if not signals_path.exists():
        if train is None:
            raise ValueError("Saved ranker package lacks its training signals")
        signals = TrainSignals(train, metadata)
        joblib.dump(signals, signals_path)
    else:
        signals = joblib.load(signals_path, mmap_mode="r")
    if len(signals.locations) != metadata.height or not np.array_equal(
            signals.locations, metadata["item_location_id"].to_numpy()):
        raise ValueError("Training signals do not align with the union corpus")
    retriever = FeatureRetriever(index, signals, metadata["in_benchmark"].to_numpy(), items)
    from catboost import CatBoostRanker
    model = CatBoostRanker()
    model.load_model(str(model_path))
    queries = pl.read_parquet(data_dir / "benchmark_queries.parquet")
    predictions = []
    text_columns = [FEATURES.index(feature) for feature in TEXT_FEATURES]
    start = time.perf_counter()
    for number, query in enumerate(queries.iter_rows(named=True), start=1):
        # Empty relevant rows make the diagnostic augmented scope exactly the
        # benchmark corpus. Neither query_id nor any label is used in scoring.
        results, scores, allowed = retriever.retrieve_with_scores(
            query, np.empty(0, dtype=np.int32), augmented=True)
        candidates = pool_ids(results, include_history=variant != "catboost_text_pool")
        x = retriever.features(query, candidates, results, scores, allowed)
        if variant == "catboost_full_with_history_features":
            values = model.predict(x, thread_count=1)
        else:
            values = model.predict(x[:, text_columns], thread_count=1)
        ranked = model_ranking(candidates, values)
        predictions.append([index.item_ids[int(item)] for item in ranked])
        if number % 250 == 0:
            print(f"Ranker prediction: {number}/{queries.height}, {time.perf_counter()-start:.1f}s", flush=True)
    benchmark_ids = pl.read_parquet(data_dir / "benchmark_items.parquet", columns=["item_id"])["item_id"].to_list()
    validation = write_answer(answer, queries["query_id"].to_list(), predictions, benchmark_ids)
    answer_hash = sha256_file(answer)
    if saved and (answer_hash != saved["answer_sha256"] or validation["queries"] != saved["query_count"]):
        raise ValueError("Portable prediction does not reproduce the frozen answer")
    manifest = {"run": str(run) if run else None, "package": str(package) if package else None,
                "variant": variant, "model_sha256": sha256_file(model_path),
                "all_train_signals_sha256": sha256_file(signals_path),
                "answer_sha256": answer_hash, "validation": validation,
                "prediction_seconds": time.perf_counter() - start,
                "total_seconds": time.perf_counter() - total_start,
                "sources": sources, "selection": selection,
                "prediction_code_sha256": sha256_file(Path(__file__))}
    memory = psutil.Process().memory_info()
    manifest["peak_rss_bytes"] = getattr(memory, "peak_wset", memory.rss)
    write_json(manifest_dir / (answer.stem + "_manifest.json"), manifest)
    if run is not None:
        from .ranker_report import write_ranker_report
        frame = pl.concat([pl.read_parquet(run / stage / "per_query.parquet") for stage in ("dev", "test")])
        write_ranker_report(run, Path("reports/ranker.md"), frame)
    print(json.dumps({"answer": str(answer), "sha256": manifest["answer_sha256"], **validation}), flush=True)
    return manifest
