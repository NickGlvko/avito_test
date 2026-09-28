"""Reproducible CPU throughput probe for frozen multilingual-e5-small."""
# Measure CPU throughput on short and long texts before a full encoding run.
from __future__ import annotations

import html
import hashlib
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
import psutil

from .data import sha256_file, write_json

MODEL = Path("models/multilingual-e5-small-614241f6")
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MAX_TOKENS = 256

_HTML = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


def clean_text(value: str | None) -> str:
    """Keep natural casing and punctuation while removing markup and extra space."""
    return _SPACES.sub(" ", _HTML.sub(" ", html.unescape(value or ""))).strip()


def item_text(title: str | None, params: str | None,
              description: str | None) -> str:
    fields = [clean_text(value) for value in (title, params, description)]
    return "passage: " + "\n".join(value for value in fields if value)


def query_text(query: str | None) -> str:
    return "query: " + clean_text(query)


def sample_items(path: Path, per_bucket: int = 800) -> tuple[list[str], dict]:
    items = pl.read_parquet(path, columns=["item_id", "item_title_raw",
                                           "item_infm_params_text", "item_description_raw"])
    items = items.with_columns(pl.col("item_description_raw").fill_null("").str.len_chars().alias("length"))
    buckets = [("short", pl.col("length") < 500),
               ("medium", (pl.col("length") >= 500) & (pl.col("length") < 2000)),
               ("long", pl.col("length") >= 2000)]
    chosen = []
    counts = {}
    for name, condition in buckets:
        group = items.filter(condition).sort("item_id")
        counts[name] = {"available": group.height, "sampled": min(per_bucket, group.height)}
        chosen.extend(group.head(per_bucket).iter_rows(named=True))
    texts = [item_text(row["item_title_raw"], row["item_infm_params_text"],
                       row["item_description_raw"]) for row in chosen]
    return texts, {"buckets": counts, "sampled_items": len(texts),
                   "unique_inputs": len(set(texts)), "sample_item_ids_sha256":
                   hashlib.sha256("\n".join(row["item_id"] for row in chosen).encode()).hexdigest()}


def benchmark(model_path: Path = MODEL, sample_path: Path = Path("src/avito_test/benchmark_items.parquet"),
              output: Path = Path("artifacts/semantic/throughput.json"),
              per_bucket: int = 800, batch_size: int = 32, threads: int = 8,
              backend: str = "torch") -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    texts, sample = sample_items(sample_path, per_bucket)
    kwargs = {}
    if backend == "onnx":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        kwargs = {"backend": "onnx", "model_kwargs": {"provider": "CPUExecutionProvider",
                  "session_options": options, "export": False}}
    elif backend != "torch":
        raise ValueError(f"Unknown backend: {backend}")
    model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True, **kwargs)
    model.max_seq_length = MAX_TOKENS
    if model.get_embedding_dimension() != 384:
        raise ValueError("Unexpected E5 embedding size")
    model.encode(texts[:min(8, len(texts))], batch_size=8,
                 normalize_embeddings=True, show_progress_bar=False)
    started = time.perf_counter()
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    seconds = time.perf_counter() - started
    if vectors.shape != (len(texts), 384) or not np.isfinite(vectors).all():
        raise ValueError("Invalid benchmark vectors")
    norms = np.linalg.norm(vectors, axis=1)
    if np.max(np.abs(norms - 1)) > 1e-4:
        raise ValueError("E5 vectors are not normalized")
    memory = psutil.Process().memory_info()
    weights = model_path / ("onnx/model.onnx" if backend == "onnx" else "model.safetensors")
    report = {"model": str(model_path), "backend": backend, "revision": REVISION,
              "model_sha256": sha256_file(weights),
              "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
              "sample": sample, "max_tokens": MAX_TOKENS,
              "batch_size": batch_size, "torch_threads": threads,
              "seconds": seconds, "items_per_second": len(texts) / seconds,
              "estimated_union_hours_for_515895_items": 515895 * seconds / len(texts) / 3600,
              "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
              "torch_version": torch.__version__}
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--per-bucket", type=int, default=800)
    args = parser.parse_args()
    benchmark(model_path=args.model or (MODEL if args.backend == "torch" else Path(str(MODEL) + "-onnx")),
              output=args.output or Path(f"artifacts/semantic/throughput_{args.backend}.json"),
              backend=args.backend, batch_size=args.batch_size, threads=args.threads,
              per_bucket=args.per_bucket)
