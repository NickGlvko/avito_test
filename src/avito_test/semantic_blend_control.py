"""Freeze the B15000 answer and the two rankers before blending."""
# Record both unchanged B models before any blend comparison.
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .data import sha256_file
from .semantic_scale_evaluate import MODELS


OUT = Path("artifacts/semantic_blend")
CONTROL = OUT / "control_B15000"
ANSWER = Path("answer_semantic_15000.csv")
PACKAGE = Path("models/semantic_15000_v1")


def freeze() -> dict:
    config = json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))
    if config["variant"] != "B15000" or sha256_file(ANSWER) != config["answer_sha256"]:
        raise ValueError("B15000 CSV or package differs from the selected dev answer")
    CONTROL.mkdir(parents=True, exist_ok=True)
    snapshot = CONTROL / ANSWER.name
    if not snapshot.exists():
        shutil.copy2(ANSWER, snapshot)
    if sha256_file(snapshot) != config["answer_sha256"]:
        raise ValueError("Frozen B15000 CSV changed")
    target_package = CONTROL / "package"
    if not target_package.exists():
        shutil.copytree(PACKAGE, target_package)
    package_names = ("config.json", "ranker.cbm", "training.json",
                     "item_id_to_vector_row.parquet")
    for name in package_names:
        if sha256_file(target_package / name) != sha256_file(PACKAGE / name):
            raise ValueError(f"Frozen B15000 package changed: {name}")
    manifest = {"answer_sha256": sha256_file(snapshot),
                "B3000_model_sha256": sha256_file(MODELS["B3000"]),
                "B15000_model_sha256": sha256_file(MODELS["B15000"]),
                "package": {name: sha256_file(target_package / name)
                            for name in package_names}}
    (CONTROL / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                              encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
