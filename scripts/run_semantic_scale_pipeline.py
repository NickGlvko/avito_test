"""Continue the 15k OOF experiment through dev selection and offline prediction."""
# Reuse the original 3k controls while extending OOF training to 15k.
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/semantic_scale"
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
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(command)}\n")
        log.flush()
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "PYTHONIOENCODING": "utf-8"})
        subprocess.run(command, cwd=ROOT, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def latest(stage: str) -> Path:
    paths = list((OUT / "evaluation" / stage / "full").glob("*/per_query.parquet"))
    if not paths:
        raise FileNotFoundError(f"No full {stage} evaluation")
    return max(paths, key=lambda path: path.stat().st_mtime).parent


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    while not (OUT / "oof_audit.json").exists():
        time.sleep(30)
    run("semantic_scale_oof", "fit")
    run("semantic_scale_evaluate", "--stage", "dev", "--workers", "2")
    dev = latest("dev")
    run("semantic_scale_report", str(dev.relative_to(ROOT)), "--select")
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    if decision["selected"] == "B3000":
        with LOG.open("a", encoding="utf-8") as log:
            log.write("\nNo dev-confirmed improvement. Keep submitted B and its CSV.\n")
        return
    run("semantic_scale_evaluate", "--stage", "test", "--workers", "2")
    test = latest("test")
    run("semantic_scale_report", str(dev.relative_to(ROOT)),
        "--test", str(test.relative_to(ROOT)))
    run("semantic_scale_predict")
    answer = ROOT / "answer_semantic_15000.csv"
    first = sha256(answer)
    run("semantic_scale_predict")
    if sha256(answer) != first:
        raise ValueError("Two offline predictions produced different CSV files")
    run("semantic_scale_verify")
    run("semantic_scale_report", str(dev.relative_to(ROOT)),
        "--test", str(test.relative_to(ROOT)))
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\nCompleted. Selected={decision['selected']}; answer SHA256={first}.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\nPipeline stopped: {type(error).__name__}: {error}\n")
        raise
