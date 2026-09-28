"""Encode a disjoint E5 item range with its own resumable checkpoint."""
# Segment boundaries must not overlap or leave gaps in the item mapping.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .data import sha256_file
from .semantic_benchmark import MAX_TOKENS
from .semantic_embeddings import DIM, MODEL, OUT, load_encoder


def encode_segment(start: int, end: int, batch_size: int = 32,
                   threads: int = 4, out: Path = OUT) -> dict:
    total = json.loads((out / "items_manifest.json").read_text(encoding="utf-8"))["unique_texts"]
    if not 0 <= start < end <= total:
        raise ValueError("Segment outside the saved unique-text matrix")
    vector_path = out / "unique_vectors.npy"
    vectors = np.load(vector_path, mmap_mode="r+")
    if vectors.shape != (total, DIM):
        raise ValueError("Saved embedding matrix has the wrong shape")
    state_path = out / f"segment_{start}_{end}.json"
    signature = {"start": start, "end": end, "model_sha256": sha256_file(MODEL / "onnx/model.onnx"),
                 "texts_sha256": sha256_file(out / "unique_texts.parquet"),
                 "batch_size": batch_size, "max_tokens": MAX_TOKENS, "dim": DIM}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise ValueError("Segment checkpoint signature differs")
        position = state["next_row"]
    else:
        position = start
        state = {"signature": signature, "next_row": start, "complete": False,
                 "elapsed_seconds": 0.}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if position == end:
        return state
    texts = pl.scan_parquet(out / "unique_texts.parquet").slice(position, end-position).collect()["text"].to_list()
    model = load_encoder(threads)
    window_start = time.perf_counter()
    for local in range(0, len(texts), batch_size):
        ids = position + local
        upper = min(ids + batch_size, end)
        batch = model.encode(texts[local:local + (upper-ids)], batch_size=batch_size,
                             normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False).astype(np.float32, copy=False)
        if batch.shape != (upper - ids, DIM) or not np.isfinite(batch).all():
            raise ValueError(f"Invalid E5 segment vectors at {ids}:{upper}")
        vectors[ids:upper] = batch
        if upper == end or (upper - start) % 800 == 0:
            vectors.flush()
            state["next_row"] = upper
            state["complete"] = upper == end
            state["elapsed_seconds"] += time.perf_counter() - window_start
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temporary.replace(state_path)
            print(f"E5 segment {start}:{end}: {upper}/{end}; {state['elapsed_seconds']:.1f}s", flush=True)
            window_start = time.perf_counter()
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(encode_segment(args.start, args.end, args.batch_size, args.threads),
                     indent=2), flush=True)
