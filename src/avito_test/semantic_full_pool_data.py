"""Assemble full B OOF parts into memory maps and one reusable quantized Pool."""
# Quantize the full B Pool once and reuse it for the two training runs.
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
from catboost import Pool
from numpy.lib.format import open_memmap

from .data import sha256_file
from .semantic_fit import BASE_SEMANTIC_FEATURES
from .ranker import TEXT_FEATURES
from .semantic_full_oof import OUT


NAMES = TEXT_FEATURES + BASE_SEMANTIC_FEATURES


def find_folder(pilot: int) -> Path:
    root = OUT / (f"pilot_{pilot}" if pilot else "full")
    manifests = list(root.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"Expected exactly one complete full-pool OOF run in {root}")
    return manifests[0].parent


def assemble(pilot: int = 0) -> dict:
    started = time.perf_counter()
    folder = find_folder(pilot)
    source = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    audit = source["audit"]
    rows = audit["full_rows"]
    paths = sorted(folder.glob("fold_*_part_*.npz"))
    if len(paths) != source["parts"]:
        raise ValueError("Full OOF parts are missing")
    arrays = {"features": ((rows, len(NAMES)), np.float32),
              "labels": ((rows,), np.uint8),
              "group_ids": ((rows,), np.int32),
              "adjusted_group_weights": ((rows,), np.float32)}
    targets = {}
    for name, (shape, dtype) in arrays.items():
        target = folder / f"{name}.npy"
        if target.exists():
            matrix = np.load(target, mmap_mode="r")
            if matrix.shape != shape or matrix.dtype != dtype:
                raise ValueError(f"Existing full matrix has wrong shape: {target}")
            targets[name] = target
    if len(targets) != len(arrays):
        if targets:
            # A Windows process can retain the last memmap briefly after a completed
            # write. Recover only fully shaped, flushed files from that final rename.
            for name, (shape, dtype) in arrays.items():
                if name in targets:
                    continue
                partial = folder / f"{name}.partial.npy"
                if not partial.exists():
                    raise ValueError("Incomplete assembled full-pool matrices")
                check = np.load(partial, mmap_mode="r")
                if check.shape != shape or check.dtype != dtype:
                    raise ValueError(f"Incomplete assembled full-pool matrix: {partial}")
                del check
                gc.collect()
                os.replace(partial, folder / f"{name}.npy")
                targets[name] = folder / f"{name}.npy"
        else:
            partials = {name: folder / f"{name}.partial.npy" for name in arrays}
            maps = {name: open_memmap(partials[name], mode="w+", dtype=dtype, shape=shape)
                    for name, (shape, dtype) in arrays.items()}
            position = 0
            for number, path in enumerate(paths, 1):
                record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                if sha256_file(path) != record["part_sha256"]:
                    raise ValueError(f"Damaged full-pool OOF part: {path}")
                with np.load(path) as part:
                    count = len(part["labels"])
                    for name in arrays:
                        maps[name][position:position+count] = part[name]
                position += count
                if number % 30 == 0:
                    print(f"Assembled {number}/{len(paths)} parts; {position}/{rows} rows; "
                          f"{time.perf_counter()-started:.1f}s", flush=True)
            if position != rows:
                raise ValueError("Assembled full-pool row count differs from audit")
            for matrix in maps.values():
                matrix.flush()
            del matrix, maps
            gc.collect()
            for name in arrays:
                os.replace(partials[name], folder / f"{name}.npy")
                targets[name] = folder / f"{name}.npy"
    labels = np.load(targets["labels"], mmap_mode="r")
    groups = np.load(targets["group_ids"], mmap_mode="r")
    weights = np.load(targets["adjusted_group_weights"], mmap_mode="r")
    if (int(labels.sum()) != audit["found_positives"] or
            len(np.unique(groups)) != audit["used_groups"] or
            not np.all(np.diff(groups) >= 0) or
            not np.isfinite(weights).all() or
            not np.all((weights > 0) & (weights <= 1))):
        raise ValueError("Assembled labels, group order or group weights changed")
    result = {"folder": str(folder), "pilot": pilot, "rows": rows,
              "groups": audit["used_groups"], "positives": audit["found_positives"],
              "full_pairs": audit["full_pairs"], "old_pairs": audit["old_pairs"],
              "arrays": {name: {"path": str(path), "sha256": sha256_file(path),
                                "bytes": path.stat().st_size}
                         for name, path in targets.items()},
              "seconds": time.perf_counter()-started,
              "peak_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset",
                                        psutil.Process().memory_info().rss)}
    (folder / "assembled.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def quantize(pilot: int = 0) -> dict:
    folder = find_folder(pilot)
    record = json.loads((folder / "assembled.json").read_text(encoding="utf-8"))
    for part in record["arrays"].values():
        if sha256_file(Path(part["path"])) != part["sha256"]:
            raise ValueError("Assembled full-pool matrix changed before quantization")
    destination = folder / "full_quantized.pool"
    if destination.exists():
        result = json.loads((folder / "quantized.json").read_text(encoding="utf-8"))
        if sha256_file(destination) != result["sha256"]:
            raise ValueError("Saved quantized Pool changed")
        return result
    started = time.perf_counter()
    features = np.load(folder / "features.npy", mmap_mode="r")
    labels = np.load(folder / "labels.npy", mmap_mode="r")
    groups = np.load(folder / "group_ids.npy", mmap_mode="r")
    pool = Pool(features, label=labels, group_id=groups, feature_names=NAMES,
                thread_count=6)
    pool.quantize()
    pool.save(str(destination))
    del pool
    gc.collect()
    check = Pool("quantized://" + str(destination))
    if check.num_row() != record["rows"] or check.num_col() != len(NAMES):
        raise ValueError("Quantized full-pool dimensions changed")
    memory = psutil.Process().memory_info()
    result = {"path": str(destination), "sha256": sha256_file(destination),
              "bytes": destination.stat().st_size, "rows": check.num_row(),
              "features": check.num_col(), "seconds": time.perf_counter()-started,
              "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss)}
    (folder / "quantized.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("assemble", "quantize"))
    parser.add_argument("--pilot", type=int, default=0)
    args = parser.parse_args()
    (assemble if args.mode == "assemble" else quantize)(args.pilot)
