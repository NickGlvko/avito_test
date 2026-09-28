"""Verify complete disjoint segment coverage and freeze the E5 item vector cache."""
# Freeze vectors only after all encoded ranges cover the corpus.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import sha256_file
from .semantic_embeddings import DIM, OUT

SEGMENTS = [(144000, 268000), (268000, 392000), (392000, 515836)]


def finalize(out: Path = OUT, segments: list[tuple[int, int]] = SEGMENTS) -> dict:
    state_path = out / "encode_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest = json.loads((out / "items_manifest.json").read_text(encoding="utf-8"))
    expected = manifest["unique_texts"]
    if not ((state["next_row"] == segments[0][0] and not state["complete"]) or
            (state["next_row"] == expected and state["complete"])):
        raise ValueError("Original sequential checkpoint no longer matches segment coverage")
    if segments[-1][1] != expected or any(a[1] != b[0] for a, b in zip(segments, segments[1:])):
        raise ValueError("Segments do not cover the remaining vector rows contiguously")
    records = []
    for start, end in segments:
        segment = json.loads((out / f"segment_{start}_{end}.json").read_text(encoding="utf-8"))
        if not segment["complete"] or segment["next_row"] != end:
            raise ValueError(f"Segment {start}:{end} is unfinished")
        signature = segment["signature"]
        if (signature["start"], signature["end"]) != (start, end):
            raise ValueError("Segment range signature changed")
        if signature["model_sha256"] != state["signature"]["weights_sha256"]:
            raise ValueError("Segment model differs from the original E5 model")
        if signature["texts_sha256"] != state["signature"]["item_text_sha256"]:
            raise ValueError("Segment input text differs from the original cache")
        records.append({"start": start, "end": end, "elapsed_seconds": segment["elapsed_seconds"]})
    vectors = np.load(out / "unique_vectors.npy", mmap_mode="r")
    if vectors.shape != (expected, DIM):
        raise ValueError("Final vector shape is invalid")
    max_error = 0.
    for start in range(0, expected, 8192):
        values = vectors[start:min(start+8192, expected)]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite item vectors at {start}")
        norms = np.linalg.norm(values, axis=1)
        max_error = max(max_error, float(np.max(np.abs(norms - 1))))
    if max_error > 2e-4:
        raise ValueError(f"Final vectors are missing or unnormalized: {max_error}")
    state["next_row"] = expected
    state["complete"] = True
    state["item_count"] = manifest["item_count"]
    state["unique_texts"] = expected
    state["vector_sha256"] = sha256_file(out / "unique_vectors.npy")
    state["inverse_sha256"] = sha256_file(out / "item_to_text.npy")
    state["parallel_segments"] = records
    state["all_vectors_norm_max_error"] = max_error
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2), flush=True)
    return state


if __name__ == "__main__":
    finalize()
