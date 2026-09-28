"""Entry points: audit, index, evaluate, select, predict, validate, run."""
# Each command uses the same paths and data contract as the saved experiment.
from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import polars as pl

from .audit import audit, preparation_signature
from .data import SEARCH, digest, load_items, sha256_file, source_manifest, write_json
from .evaluation import diagnose, evaluate
from .retrieval import EXPERIMENTS, INDEX_CONFIG, TrainSignals, channels, fuse, get_index
from .reporting import build_report
from .submission import validate_answer, write_answer


def prepare(data_dir: Path, output: Path) -> pl.DataFrame:
    manifest = source_manifest(data_dir)
    provenance = output / "sources.json"
    preparation = output / "preparation.json"
    if (not provenance.exists() or json.loads(provenance.read_text(encoding="utf-8")) != manifest
            or not preparation.exists()
            or json.loads(preparation.read_text(encoding="utf-8")) != preparation_signature()):
        audit(data_dir, output)
    train = pl.read_parquet(output / "train_pairs.parquet")
    if "item_category_id" not in train.columns:
        audit(data_dir, output)
        train = pl.read_parquet(output / "train_pairs.parquet")
    return train


def select(output: Path, config_path: Path) -> dict:
    summary = json.loads((output / "group_dev" / "summary.json").read_text(encoding="utf-8"))
    best = summary["metrics"][0]
    config = {"experiment": best["experiment"], "weights": EXPERIMENTS[best["experiment"]],
              "rrf_k": 60, "depth": 500, "index": INDEX_CONFIG,
              "selection": "highest macro Recall@50 on group/dev; tie broken by experiment name",
              "dev_recall50": best["recall@50"], "dev_queries": best["queries"]}
    write_json(config_path, config)
    diagnose(output, config["weights"])
    return config


def predict(index, items: pl.DataFrame, train: pl.DataFrame, data_dir: Path,
            output: Path, config: dict, answer: Path) -> dict:
    if config["index"] != INDEX_CONFIG or config["depth"] != 500 or config["rrf_k"] != 60:
        raise ValueError("Config does not match this implementation")
    queries = pl.read_parquet(data_dir / "benchmark_queries.parquet")
    signals = TrainSignals(train, items)
    predictions = []
    start = time.perf_counter()
    for number, query in enumerate(queries.iter_rows(named=True)):
        ranking = fuse(channels(index, signals, query), config["weights"], depth=50)
        predictions.append([index.item_ids[i] for i in ranking])
        if (number + 1) % 100 == 0:
            print(f"Predict: {number + 1}/{queries.height}, {time.perf_counter()-start:.1f}s", flush=True)
    check = write_answer(answer, queries["query_id"].to_list(), predictions, index.item_ids)
    manifest = {"config": config, "validation": check, "sources": source_manifest(data_dir),
                "answer_sha256": sha256_file(answer), "prediction_seconds": time.perf_counter() - start,
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: version(name) for name in ["numpy", "polars", "scipy", "scikit-learn", "joblib"]},
                "code": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "uv_lock_sha256": sha256_file(Path("uv.lock")) if Path("uv.lock").exists() else None}
    write_json(output / (answer.stem + "_manifest.json"), manifest)
    print(json.dumps({"answer": str(answer), **check, "sha256": manifest["answer_sha256"]}), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "index", "evaluate", "evaluate-expanded", "ranker", "geo-diagnose", "geo-train", "geo-evaluate", "geo-report", "save-ranker-bundle", "predict-ranker", "select", "predict", "validate", "report", "run"])
    parser.add_argument("--data-dir", type=Path, default=Path("src/avito_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--answer", type=Path, default=Path("answer.csv"))
    parser.add_argument("--ranker-answer", type=Path, default=Path("answer_ranker.csv"))
    parser.add_argument("--ranker-run", type=Path, default=None,
                        help="Frozen full CatBoost run; defaults to artifacts/ranker/latest.json")
    parser.add_argument("--ranker-package", type=Path, default=None,
                        help="Portable weights and configuration directory")
    parser.add_argument("--geo-model-run", type=Path, default=None,
                        help="Expanded geography CatBoost run")
    parser.add_argument("--geo-diagnostics", type=Path, default=None,
                        help="Stage directory containing quota metrics and global probes")
    parser.add_argument("--geo-evaluation", type=Path, default=None,
                        help="Stage directory containing retrained global-pool metrics")
    parser.add_argument("--protocol", choices=["group", "unseen"], default="group")
    parser.add_argument("--stage", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=0, help="0 = all eligible groups; use only for smoke tests")
    parser.add_argument("--workers", type=int, default=4, help="Processes for expanded corpus validation")
    parser.add_argument("--expanded-stages", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    parser.add_argument("--train-queries", type=int, default=3000, help="OOF query groups for CatBoost")
    args = parser.parse_args()
    data_dir, output = args.data_dir, args.output_dir
    if args.command == "evaluate-expanded":
        from .extended_validation import run_extended
        train = prepare(data_dir, output)
        answer_before = sha256_file(args.answer) if args.answer.exists() else None
        run_extended(data_dir, output / "expanded", train, source_manifest(data_dir),
                     args.expanded_stages, args.workers, args.limit)
        if answer_before is not None and sha256_file(args.answer) != answer_before:
            raise RuntimeError("Expanded validation changed the submission file")
        write_json(output / "expanded" / "submission_unchanged.json", {"answer_sha256": answer_before,
                   "platform_recall50_user_reported": 0.666981, "unchanged": True})
        return
    if args.command == "ranker":
        from .ranker_workflow import run_ranker_workflow
        train = prepare(data_dir, output)
        answer_before = sha256_file(args.answer) if args.answer.exists() else None
        run_ranker_workflow(data_dir, output, train, args.train_queries, args.limit, args.workers)
        if answer_before is not None and sha256_file(args.answer) != answer_before:
            raise RuntimeError("Ranker validation changed the submitted answer")
        return
    if args.command == "geo-diagnose":
        from .geography import diagnose as geo_diagnose

        train = prepare(data_dir, output)
        result = geo_diagnose(data_dir, output, train, args.stage, args.workers,
                              args.limit, ranker_run=args.ranker_run)
        print(json.dumps({"geography_diagnostics": str(result)}), flush=True)
        return
    if args.command == "geo-train":
        from .geography_expanded import train as geo_train

        train = prepare(data_dir, output)
        result = geo_train(data_dir, output, train, args.train_queries)
        print(json.dumps({"geography_model": str(result)}), flush=True)
        return
    if args.command == "geo-evaluate":
        from .geography_expanded import evaluate as geo_evaluate

        train = prepare(data_dir, output)
        model = args.geo_model_run or Path(json.loads((output / "geography" / "expanded_model_latest.json").read_text(encoding="utf-8"))["run"])
        ranker = args.ranker_run or Path(json.loads((output / "ranker" / "latest.json").read_text(encoding="utf-8"))["run"])
        result = geo_evaluate(data_dir, output, train, model, ranker, args.stage,
                              args.workers, args.limit)
        print(json.dumps({"geography_expanded_evaluation": str(result)}), flush=True)
        return
    if args.command == "geo-report":
        from .geography_report import analyze, freeze_dev_decision, markdown

        if args.geo_diagnostics is None:
            raise ValueError("--geo-diagnostics is required")
        ranker = args.ranker_run or Path(json.loads((output / "ranker" / "latest.json").read_text(encoding="utf-8"))["run"])
        summary_path = output / "geography" / ("dev_summary.json" if args.stage == "dev" else "test_summary.json")
        summary = analyze(args.geo_diagnostics, ranker, args.geo_evaluation,
                          summary_path, args.stage)
        decision_path = output / "geography" / "dev_decision.json"
        decision = (freeze_dev_decision(summary, decision_path) if args.stage == "dev"
                    else json.loads(decision_path.read_text(encoding="utf-8")))
        if args.stage == "test" and sha256_file(output / "geography" / "dev_summary.json") != decision["dev_summary_sha256"]:
            raise ValueError("Frozen dev decision does not match the saved dev summary")
        markdown(summary, decision,
                 Path("reports/geography.md") if args.stage == "dev" else Path("reports/geography_test.md"))
        print(json.dumps({"summary": str(summary_path), "selected": decision["selected"]}), flush=True)
        return
    if args.command == "predict-ranker":
        from .ranker_predict import predict_ranker
        train = None if args.ranker_package else prepare(data_dir, output)
        predict_ranker(data_dir, output, train, args.ranker_answer, args.ranker_run,
                       args.ranker_package)
        return
    if args.command == "save-ranker-bundle":
        from .ranker_bundle import save_ranker_bundle
        run = args.ranker_run or Path(json.loads((output / "ranker" / "latest.json").read_text(encoding="utf-8"))["run"])
        bundle = args.ranker_package or Path("models/ranker_v1")
        result = save_ranker_bundle(run, bundle, args.ranker_answer, data_dir)
        print(json.dumps({"package": str(bundle), "variant": result["selected_variant"],
                          "answer_sha256": result["answer_sha256"]}), flush=True)
        return
    if args.command == "report":
        build_report(output, args.config, Path("reports/baseline.md"))
        return
    if args.command == "audit":
        audit(data_dir, output)
        return
    items = load_items(data_dir)
    if args.command == "validate":
        queries = pl.read_parquet(data_dir / "benchmark_queries.parquet")
        print(validate_answer(args.answer, queries["query_id"].to_list(), items["item_id"].to_list()))
        return
    if args.command == "select":
        print(select(output, args.config))
        return
    train = prepare(data_dir, output)
    write_json(output / "item_ids.json", items["item_id"].to_list())
    index = get_index(items, data_dir, output)
    if args.command == "index":
        return
    if args.command == "run":
        evaluate(index, items, train, output, limit=args.limit)
        config = select(output, args.config)
        # Selection is already frozen before accessing test or unseen-text diagnostics.
        frozen = {config["experiment"]: config["weights"]}
        evaluate(index, items, train, output, stage="test", limit=args.limit, experiments=frozen)
        diagnose(output, config["weights"], stage="test")
        evaluate(index, items, train, output, protocol="unseen", limit=args.limit, experiments=frozen)
        diagnose(output, config["weights"], protocol="unseen")
        predict(index, items, train, data_dir, output, config, args.answer)
        build_report(output, args.config, Path("reports/baseline.md"))
    elif args.command == "evaluate":
        experiments = None
        if args.stage == "test" or args.protocol == "unseen":
            config = json.loads(args.config.read_text(encoding="utf-8"))
            experiments = {config["experiment"]: config["weights"]}
        evaluate(index, items, train, output, args.protocol, args.stage, args.limit, experiments)
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        predict(index, items, train, data_dir, output, config, args.answer)


if __name__ == "__main__":
    main()
