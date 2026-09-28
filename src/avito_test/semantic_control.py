"""Retrain the unchanged CatBoost control from the saved OOF rows."""
# The unchanged baseline isolates the effect of newly added E5 features.
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRanker, Pool

from .data import sha256_file
from .ranker import FEATURES, MODEL_PARAMS, TEXT_FEATURES

OOF = Path("artifacts/ranker/full/c392362a99e2c4d1/oof_1000.npz")
ORIGINAL = Path("models/ranker_v1/text_model.cbm")
OUTPUT = Path("artifacts/semantic/control_A.cbm")


if __name__ == "__main__":
    start = time.perf_counter()
    with np.load(OOF) as data:
        x = data["features"]
        labels = data["labels"]
        groups = data["group_ids"]
    columns = [FEATURES.index(name) for name in TEXT_FEATURES]
    pool = Pool(x[:, columns], label=labels, group_id=groups,
                feature_names=TEXT_FEATURES)
    model = CatBoostRanker(**MODEL_PARAMS)
    model.fit(pool)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT))
    original = CatBoostRanker()
    original.load_model(str(ORIGINAL))
    sample = x[:20000, columns]
    difference = np.max(np.abs(model.predict(sample) - original.predict(sample)))
    report = {"training_rows": len(labels), "training_groups": int(groups.max()) + 1,
              "positive_rows": int(labels.sum()), "elapsed_seconds": time.perf_counter() - start,
              "original_sha256": sha256_file(ORIGINAL), "retrained_sha256": sha256_file(OUTPUT),
              "max_prediction_difference_on_20000_saved_rows": float(difference),
              "byte_identical": sha256_file(ORIGINAL) == sha256_file(OUTPUT)}
    Path("artifacts/semantic/control_A.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
