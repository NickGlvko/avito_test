"""Evaluate fixed blends, select on dev, then reproduce the chosen CSV offline."""
# The wrapper executes frozen stages in order and stops on a failing stage.
from __future__ import annotations

import hashlib
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/semantic_blend"
LOG = OUT / "pipeline.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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


def latest(stage: str) -> Path:
    paths = list((OUT / "evaluation" / stage).glob("*/per_query.parquet"))
    if not paths:
        raise FileNotFoundError(f"No complete {stage} evaluation")
    return max(paths, key=lambda path: path.stat().st_mtime).parent


def main(skip_dev: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run("semantic_blend_control")
    if not skip_dev:
        run("semantic_blend_evaluate", "--stage", "dev", "--workers", "3")
    dev = latest("dev")
    run("semantic_blend_report", str(dev.relative_to(ROOT)), "--select")
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    if decision["selected"] == "B15000":
        before = sha256(ROOT / "answer_semantic_15000.csv")
        run("semantic_blend_predict")
        after = sha256(OUT / "answer_B15000_repeat.csv")
        if after != before:
            raise ValueError("Offline B15000 repeat differs from the saved answer")
        run("semantic_scale_verify")
        run("semantic_blend_report", str(dev.relative_to(ROOT)))
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\nNo eligible blend. B15000 preserved: {after}.\n")
        return
    run("semantic_blend_evaluate", "--stage", "test", "--workers", "3")
    test = latest("test")
    run("semantic_blend_report", str(dev.relative_to(ROOT)),
        "--test", str(test.relative_to(ROOT)))
    run("semantic_blend_predict")
    answer = ROOT / "answer_semantic_blend.csv"
    first = sha256(answer)
    run("semantic_blend_predict")
    if sha256(answer) != first:
        raise ValueError("Two offline blended CSV predictions differ")
    run("semantic_blend_verify")
    run("semantic_blend_report", str(dev.relative_to(ROOT)),
        "--test", str(test.relative_to(ROOT)))
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\nSelected {decision['selected']}; answer SHA256={first}.\n")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--skip-dev", action="store_true",
                            help="use an already completed full-dev evaluation")
        main(parser.parse_args().skip_dev)
    except Exception as error:
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\nPipeline stopped: {type(error).__name__}: {error}\n")
        raise
