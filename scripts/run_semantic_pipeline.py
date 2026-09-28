"""Resume the long offline E5 experiment after the two embedding caches finish."""
# Resume saved E5 work before evaluating any semantic ablation.
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/semantic"
LOG = ARTIFACTS / "pipeline.log"


def ready() -> bool:
    state = ARTIFACTS / "embeddings/encode_state.json"
    queries = ARTIFACTS / "embeddings/query_vectors_manifest.json"
    return queries.exists() and state.exists() and json.loads(state.read_text(encoding="utf-8")).get("complete", False)


def run(module: str, *args: str) -> None:
    command = ["uv", "--cache-dir", ".uv-cache", "run", "--offline", "--frozen",
               "python", "-X", "utf8", "-m", f"avito_test.{module}", *args]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n[{stamp}] {' '.join(command)}\n")
        log.flush()
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "PYTHONIOENCODING": "utf-8"})
        subprocess.run(command, cwd=ROOT, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def newest(stage: str) -> Path:
    paths = list((ARTIFACTS / "evaluation" / stage / "full").glob("*/per_query.parquet"))
    if not paths:
        raise FileNotFoundError(f"No full {stage} results")
    return max(paths, key=lambda path: path.stat().st_mtime).parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    while not ready():
        time.sleep(30)
    run("semantic_finalize_embeddings")
    run("semantic_fit", "B")
    run("semantic_extended_oof")
    run("semantic_evaluate", "--stage", "dev", "--limit", "3000", "--workers", "2")
    run("semantic_evaluate", "--stage", "dev", "--limit", "0", "--workers", "2")
    dev = newest("dev")
    run("semantic_report", str(dev.relative_to(ROOT)), "--select")
    run("semantic_error_examples", str(dev.relative_to(ROOT)))
    run("semantic_evaluate", "--stage", "test", "--limit", "0", "--workers", "2")
    test = newest("test")
    run("semantic_report", str(dev.relative_to(ROOT)), "--test", str(test.relative_to(ROOT)))
    analysis = json.loads((ARTIFACTS / "analysis.json").read_text(encoding="utf-8"))
    selected = analysis["decision"]["selected"]
    control = analysis["test_summary"]
    confirmed = selected != "A" and all(
        control[scope][selected]["recall50"] > control[scope]["A"]["recall50"]
        for scope in ("union", "benchmark_plus_positives"))
    if confirmed:
        run("semantic_predict")
        answer = ROOT / "answer_semantic.csv"
        first = sha256(answer)
        run("semantic_predict")
        if sha256(answer) != first:
            raise ValueError("Semantic answer did not reproduce byte-for-byte")
        run("semantic_report", str(dev.relative_to(ROOT)), "--test", str(test.relative_to(ROOT)))
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\nConfirmed semantic answer SHA256: {first}\n")
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\nSemantic evaluation complete. Selected={selected}, "
                  f"test_confirmed={confirmed}.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\nPipeline stopped: {type(error).__name__}: {error}\n")
        raise
