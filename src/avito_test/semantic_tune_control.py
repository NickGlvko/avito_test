"""Freeze the submitted B3000/B15000 blend as this experiment's control."""
# Keep the submitted blend fixed as the capacity experiment's baseline.
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .data import sha256_file


OUT = Path("artifacts/semantic_tune/control_submitted_blend")
ANSWER = Path("answer_semantic_blend.csv")
PACKAGE = Path("models/semantic_blend_v1")
EXPECTED_SHA256 = "8124b99e143fcc1091705be80e9bc4c2b62a42c3cf600909ae352afdf54ee707"


def freeze() -> dict:
    config = json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))
    if (sha256_file(ANSWER) != EXPECTED_SHA256 or
            config["answer_sha256"] != EXPECTED_SHA256 or
            config["selected"] != "blend_0.75" or config["alpha"] != 0.75):
        raise ValueError("Submitted blend or its frozen package changed")
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / ANSWER.name
    if not destination.exists():
        shutil.copy2(ANSWER, destination)
    package = OUT / "package"
    if not package.exists():
        shutil.copytree(PACKAGE, package)
    files = ("config.json", "B3000.cbm", "B15000.cbm",
             "item_id_to_vector_row.parquet")
    hashes = {name: sha256_file(PACKAGE / name) for name in files}
    if sha256_file(destination) != EXPECTED_SHA256 or any(
            sha256_file(package / name) != digest for name, digest in hashes.items()):
        raise ValueError("Frozen submitted blend differs from its source")
    manifest = {"platform_recall50_user_reported": 0.744280,
                "answer_sha256": EXPECTED_SHA256,
                "model_package": hashes, "dev_union_recall50": 0.741444175,
                "dev_augmented_recall50": 0.807314351}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    freeze()
