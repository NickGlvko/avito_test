"""Reproduce the fixed two-training capacity experiment and optional answer offline."""
# Evaluate tree checkpoints only after the shared feature cache exists.
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/semantic_tune"
LOG = OUT / "pipeline.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(module: str, *args: str) -> None:
    command = ["uv", "--cache-dir", ".uv-cache", "run", "--offline", "--frozen",
               "python", "-X", "utf8", "-m", f"avito_test.{module}", *args]
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONIOENCODING": "utf-8"})
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(command)}\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def trained(depth: int) -> bool:
    report = OUT / f"depth_{depth}_960.json"
    if not report.exists():
        return False
    record = json.loads(report.read_text(encoding="utf-8"))
    return sha256(ROOT / record["model"]) == record["model_sha256"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run("semantic_tune_control")
    for depth in (6, 8):
        if not trained(depth):
            run("semantic_tune_train", "--depth", str(depth))
    if not list((OUT / "cache/dev").glob("*/manifest.json")):
        run("semantic_tune_cache", "--stage", "dev", "--workers", "3")
    run("semantic_tune_evaluate", "checkpoints")
    run("semantic_tune_select")
    run("semantic_tune_final", "dev")
    run("semantic_tune_final", "select")
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    if decision["selected"] == "submitted_blend":
        run("semantic_tune_report")
        with LOG.open("a", encoding="utf-8") as log:
            log.write("\nNo dev-confirmed improvement; submitted blend preserved.\n")
        return
    if not list((OUT / "cache/test").glob("*/manifest.json")):
        run("semantic_tune_cache", "--stage", "test", "--workers", "3")
    run("semantic_tune_final", "test")
    run("semantic_tune_predict")
    answer = ROOT / "answer_semantic_tuned.csv"
    first = sha256(answer)
    run("semantic_tune_predict")
    if sha256(answer) != first:
        raise ValueError("Two offline tuned predictions differ")
    run("semantic_tune_verify")
    run("semantic_tune_report")
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\nSelected {decision['selected']}; answer SHA256={first}.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\nPipeline stopped: {type(error).__name__}: {error}\n")
        raise
