"""Train the two fixed PairLogit variants on the identical full B OOF pool."""
# Group weights affect PairLogit pairs rather than individual row labels.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
from catboost import CatBoostRanker, Pool

from .data import sha256_file
from .ranker import MODEL_PARAMS
from .semantic_full_pool_data import find_folder
from .semantic_full_oof import OUT


VARIANTS = ("unweighted", "balanced")


def train(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError("Only the two fixed full-pool weighting variants are allowed")
    folder = find_folder(0)
    source = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assembled = json.loads((folder / "assembled.json").read_text(encoding="utf-8"))
    quantized = json.loads((folder / "quantized.json").read_text(encoding="utf-8"))
    if (source["audit"]["used_groups"] != 13366 or
            source["audit"]["found_positives"] != 17758 or
            source["audit"]["old_negative_rows"] != 2004900 or
            quantized["rows"] != assembled["rows"] or
            sha256_file(Path(quantized["path"])) != quantized["sha256"]):
        raise ValueError("Full OOF pool differs from B15000 or quantized data changed")
    params = dict(MODEL_PARAMS)
    if (params["loss_function"] != "PairLogit" or params["iterations"] != 240 or
            params["depth"] != 6 or params["learning_rate"] != 0.06 or
            "max_pairs" in params):
        raise ValueError("The fixed B15000 training parameters changed")
    pool = Pool("quantized://" + quantized["path"])
    if variant == "balanced":
        weights_path = folder / "adjusted_group_weights.npy"
        if sha256_file(weights_path) != assembled["arrays"]["adjusted_group_weights"]["sha256"]:
            raise ValueError("Balanced group weights changed")
        weights = np.load(weights_path, mmap_mode="r")
        pool.set_group_weight(weights)
    started = time.perf_counter()
    model = CatBoostRanker(**params)
    model.fit(pool)
    elapsed = time.perf_counter()-started
    model_path = OUT / f"full_{variant}.cbm"
    model.save_model(str(model_path))
    memory = psutil.Process().memory_info()
    report = {"variant": variant, "model": str(model_path),
              "model_sha256": sha256_file(model_path),
              "params": params, "quantized_pool_sha256": quantized["sha256"],
              "group_weight_rule": "1" if variant == "unweighted" else
                                   "150 / number_of_full_unlabeled_candidates",
              "rows": assembled["rows"], "groups": assembled["groups"],
              "positive_rows": assembled["positives"],
              "unlabeled_rows": assembled["rows"]-assembled["positives"],
              "automatically_generated_pairs": assembled["full_pairs"],
              "previous_pairs": assembled["old_pairs"],
              "max_pairs": None,
              "train_seconds": elapsed,
              "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss)}
    (OUT / f"full_{variant}.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    train(parser.parse_args().variant)
