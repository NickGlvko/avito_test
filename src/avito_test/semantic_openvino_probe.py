"""Probe exact OpenVINO CPU inference directly, bypassing incompatible Optimum wrapper."""
# Benchmark an alternate local backend without changing the reference model.
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .semantic_benchmark import MAX_TOKENS, sample_items


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OPENVINO_TELEMETRY_DISABLE"] = "1"
    import openvino as ov
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    texts, details = sample_items(Path("src/avito_test/benchmark_items.parquet"), 200)
    original = Path("models/multilingual-e5-small-614241f6")
    onnx_path = Path("models/multilingual-e5-small-614241f6-onnx/onnx/model.onnx")
    tokenizer = AutoTokenizer.from_pretrained(str(original), local_files_only=True)
    core = ov.Core()
    started = time.perf_counter()
    compiled = core.compile_model(str(onnx_path), "CPU", {"INFERENCE_NUM_THREADS": "8"})
    compile_seconds = time.perf_counter() - started

    def encode(values: list[str], batch_size: int = 32) -> np.ndarray:
        batches = []
        for position in range(0, len(values), batch_size):
            inputs = tokenizer(values[position:position+batch_size], truncation=True,
                               padding=True, max_length=MAX_TOKENS, return_tensors="np")
            payload = {name: inputs[name] for name in ("input_ids", "attention_mask")}
            payload["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
            hidden = compiled(payload)[compiled.output(0)]
            mask = inputs["attention_mask"][..., None]
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
            batches.append(pooled.astype(np.float32))
        return np.concatenate(batches)

    encode(texts[:32])
    started = time.perf_counter()
    vectors = encode(texts)
    encode_seconds = time.perf_counter() - started
    reference = SentenceTransformer(str(original), device="cpu", local_files_only=True)
    reference.max_seq_length = MAX_TOKENS
    truth = reference.encode(texts[:32], normalize_embeddings=True, batch_size=32,
                             show_progress_bar=False)
    cos = np.sum(vectors[:32] * truth, axis=1)
    report = {"items": len(texts), "batch_size": 32, "model_compile_seconds": compile_seconds,
              "encode_seconds": encode_seconds, "items_per_second": len(texts)/encode_seconds,
              "estimated_union_hours": 515895*encode_seconds/len(texts)/3600,
              "mean_cosine_to_torch": float(cos.mean()), "min_cosine_to_torch": float(cos.min())}
    Path("artifacts/semantic/openvino_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
