"""Fit the two predeclared CatBoost capacities on the frozen B15000 OOF rows."""
# Train the two declared depths sequentially on identical OOF data.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
from catboost import CatBoostRanker, Pool

from .data import sha256_file
from .ranker import MODEL_PARAMS, TEXT_FEATURES
from .semantic_fit import BASE_SEMANTIC_FEATURES


OUT = Path("artifacts/semantic_tune")
OOF = Path("artifacts/semantic_scale/B_15000_oof.npz")
TRAINING = Path("artifacts/semantic_scale/training.json")
BASE = Path("models/semantic_15000_v1/ranker.cbm")
CHECKPOINTS = (240, 480, 720, 960)
NAMES = TEXT_FEATURES + BASE_SEMANTIC_FEATURES


def train(depth: int) -> dict:
    if depth not in (6, 8):
        raise ValueError("Only the two predeclared depths may be trained")
    OUT.mkdir(parents=True, exist_ok=True)
    record = json.loads(TRAINING.read_text(encoding="utf-8"))
    expected = record["B_oof_sha256"]
    if sha256_file(OOF) != expected:
        raise ValueError("Frozen B15000 OOF table changed")
    baseline = json.loads(Path("models/semantic_15000_v1/training.json")
                          .read_text(encoding="utf-8"))
    if (baseline["feature_names"] != NAMES or
            baseline["params"] != MODEL_PARAMS or
            sha256_file(BASE) != baseline["model_sha256"]):
        raise ValueError("B15000 control model or parameters changed")
    with np.load(OOF) as data:
        features = data["features"]
        labels = data["labels"]
        groups = data["group_ids"]
    if (features.shape != (2022658, len(NAMES)) or len(labels) != len(groups) or
            int(labels.sum()) != 17758 or len(np.unique(groups)) != 13366 or
            not np.array_equal(np.unique(groups), np.arange(13366))):
        raise ValueError("Frozen OOF rows, labels, group order or features changed")
    params = {**MODEL_PARAMS, "depth": depth, "iterations": 960}
    model = CatBoostRanker(**params)
    started = time.perf_counter()
    model.fit(Pool(features, label=labels, group_id=groups, feature_names=NAMES))
    elapsed = time.perf_counter() - started
    path = OUT / f"depth_{depth}_960.cbm"
    model.save_model(str(path))
    memory = psutil.Process().memory_info()
    result = {"model": str(path), "model_sha256": sha256_file(path),
              "oof": str(OOF), "oof_sha256": expected, "feature_names": NAMES,
              "params": params, "checkpoints": CHECKPOINTS,
              "rows": len(labels), "groups": 13366, "positive_rows": int(labels.sum()),
              "unlabeled_rows": len(labels)-int(labels.sum()),
              "train_seconds": elapsed,
              "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss)}
    (OUT / f"depth_{depth}_960.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, choices=(6, 8), required=True)
    train(parser.parse_args().depth)
