"""Check embedding drift before using a quantized CPU backend."""
# Check embedding drift before replacing the reference CPU encoder.
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from .semantic_benchmark import MAX_TOKENS, item_text, query_text


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(8)
    items = pl.read_parquet("src/avito_test/benchmark_items.parquet",
                            columns=["item_title_raw", "item_infm_params_text", "item_description_raw"])
    queries = pl.read_parquet("src/avito_test/benchmark_queries.parquet",
                              columns=["search_query"])
    texts = [item_text(*row) for row in items.head(128).iter_rows()]
    qtexts = [query_text(q) for q in queries["search_query"].head(64)]
    inputs = texts + qtexts
    models = [Path("models/multilingual-e5-small-614241f6"),
              Path("models/multilingual-e5-small-614241f6-onnx"),
              Path("models/multilingual-e5-small-614241f6-onnx-int8")]
    vectors = []
    for backend, path in zip(("torch", "onnx", "onnx"), models):
        kwargs = {"backend": "onnx", "model_kwargs": {"provider": "CPUExecutionProvider", "export": False}} if backend == "onnx" else {}
        model = SentenceTransformer(str(path), device="cpu", local_files_only=True, **kwargs)
        model.max_seq_length = MAX_TOKENS
        vectors.append(model.encode(inputs, batch_size=32, normalize_embeddings=True,
                                    show_progress_bar=False))
        del model
    result = {"inputs": len(inputs), "comparisons": {}}
    sim1 = vectors[0][128:] @ vectors[0][:128].T
    for name, vector in zip(("float_onnx", "int8_onnx"), vectors[1:]):
        row_cos = np.sum(vectors[0] * vector, axis=1)
        sim2 = vector[128:] @ vector[:128].T
        overlap = [len(set(np.argsort(-a)[:20]) & set(np.argsort(-b)[:20])) / 20
                   for a, b in zip(sim1, sim2)]
        result["comparisons"][name] = {"mean_same_input_cosine": float(row_cos.mean()),
            "min_same_input_cosine": float(row_cos.min()),
            "pair_score_mean_absolute_error": float(np.mean(np.abs(sim1 - sim2))),
            "top20_overlap": float(np.mean(overlap))}
    print(json.dumps(result, indent=2))
    Path("artifacts/semantic/backend_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
