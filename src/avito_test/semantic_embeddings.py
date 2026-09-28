"""Offline, resumable embeddings for the frozen E5 model and union item snapshot."""
# Reuse vectors for identical item text while retaining every item_id row.
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from numpy.lib.format import open_memmap

from .data import sha256_file
from .semantic_benchmark import MAX_TOKENS, REVISION, clean_text, item_text, query_text

DATA = Path("src/avito_test")
INDEX = Path("artifacts/expanded/cache/73915bb3bc082213/union_index.joblib")
MODEL = Path("models/multilingual-e5-small-614241f6-onnx")
OUT = Path("artifacts/semantic/embeddings")
DIM = 384


def load_encoder(threads: int = 8):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import onnxruntime as ort
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    model = SentenceTransformer(str(MODEL), backend="onnx", device="cpu",
        local_files_only=True, model_kwargs={"provider": "CPUExecutionProvider",
                                              "session_options": options, "export": False})
    model.max_seq_length = MAX_TOKENS
    if model.get_embedding_dimension() != DIM:
        raise ValueError("Unexpected E5 embedding dimension")
    return model


def prepare_item_texts(out: Path = OUT) -> tuple[Path, Path, dict]:
    """Deduplicate exact encoder inputs while retaining every item_id in index order."""
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "items_manifest.json"
    text_path = out / "unique_texts.parquet"
    inverse_path = out / "item_to_text.npy"
    index_ids = joblib.load(INDEX, mmap_mode="r").item_ids
    sources = {name: sha256_file(DATA / name) for name in ("train.parquet", "benchmark_items.parquet")}
    signature = {"sources": sources, "revision": REVISION,
                 "text_policy": "html-unescape; strip-tags; collapse-whitespace; title-params-description; 256 tokens; globally length-sorted for batching"}
    if manifest_path.exists() and text_path.exists() and inverse_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
            raise ValueError("Existing item text cache has a different data or cleaning signature")
        if manifest["item_count"] != len(index_ids):
            raise ValueError("Item text cache no longer aligns with index")
        return text_path, inverse_path, manifest
    columns = ["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"]
    benchmark = pl.read_parquet(DATA / "benchmark_items.parquet", columns=columns)
    training = pl.read_parquet(DATA / "train.parquet", columns=columns).unique()
    if training["item_id"].n_unique() != training.height:
        raise ValueError("Conflicting train item snapshots")
    extra = training.join(benchmark.select("item_id"), on="item_id", how="anti")
    items = pl.concat([benchmark, extra]).sort("item_id")
    if items["item_id"].to_list() != index_ids:
        raise ValueError("E5 item order does not align with text index")
    del benchmark, training, extra
    gc.collect()
    lookup: dict[str, int] = {}
    texts: list[str] = []
    inverse = np.empty(items.height, dtype=np.int32)
    for n, row in enumerate(items.select(columns[1:]).iter_rows()):
        text = item_text(*row)
        idx = lookup.get(text)
        if idx is None:
            idx = len(texts)
            lookup[text] = idx
            texts.append(text)
        inverse[n] = idx
    del lookup, items
    gc.collect()
    order = sorted(range(len(texts)), key=lambda pos: (len(texts[pos]), pos))
    reverse = np.empty(len(order), dtype=np.int32)
    reverse[np.asarray(order, dtype=np.int32)] = np.arange(len(order), dtype=np.int32)
    inverse = reverse[inverse]
    texts = [texts[pos] for pos in order]
    del order, reverse
    pl.DataFrame({"text": texts}).write_parquet(text_path, compression="zstd")
    np.save(inverse_path, inverse)
    manifest = {"signature": signature, "item_count": len(index_ids),
                "unique_texts": len(texts), "duplicates": len(index_ids) - len(texts),
                "text_sha256": sha256_file(text_path),
                "inverse_sha256": sha256_file(inverse_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return text_path, inverse_path, manifest


def encode_items(out: Path = OUT, batch_size: int = 64, threads: int = 8) -> dict:
    text_path, inverse_path, items_manifest = prepare_item_texts(out)
    vector_path, state_path = out / "unique_vectors.npy", out / "encode_state.json"
    unique = items_manifest["unique_texts"]
    weights_sha = sha256_file(MODEL / "onnx/model.onnx")
    signature = {"item_text_sha256": items_manifest["text_sha256"],
                 "weights_sha256": weights_sha, "tokenizer_sha256": sha256_file(MODEL / "tokenizer.json"),
                 "max_tokens": MAX_TOKENS, "batch_size": batch_size, "dim": DIM}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise ValueError("Existing vector cache has a different model or data signature")
        next_row = state["next_row"]
        vectors = np.load(vector_path, mmap_mode="r+")
        if vectors.shape != (unique, DIM):
            raise ValueError("Incorrect vector cache shape")
    else:
        next_row = 0
        vectors = open_memmap(vector_path, mode="w+", dtype=np.float32, shape=(unique, DIM))
        state = {"signature": signature, "next_row": 0, "complete": False,
                 "elapsed_seconds": 0.0}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if next_row == unique:
        if not state.get("vector_sha256"):
            state["item_count"] = items_manifest["item_count"]
            state["unique_texts"] = unique
            state["vector_sha256"] = sha256_file(vector_path)
            state["inverse_sha256"] = sha256_file(inverse_path)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state
    model = load_encoder(threads)
    texts = pl.read_parquet(text_path)["text"].to_list()
    start = time.perf_counter()
    for position in range(next_row, unique, batch_size):
        end = min(position + batch_size, unique)
        batch = model.encode(texts[position:end], batch_size=batch_size,
                             normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False).astype(np.float32, copy=False)
        if batch.shape != (end - position, DIM) or not np.isfinite(batch).all():
            raise ValueError("Invalid E5 item vectors")
        vectors[position:end] = batch
        if end % (batch_size * 25) == 0 or end == unique:
            vectors.flush()
            state["next_row"] = end
            state["complete"] = end == unique
            state["elapsed_seconds"] += time.perf_counter() - start
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temporary.replace(state_path)
            print(f"E5 embeddings: {end}/{unique}; {state['elapsed_seconds']:.1f}s", flush=True)
            start = time.perf_counter()
    del vectors, texts, model
    gc.collect()
    state["item_count"] = items_manifest["item_count"]
    state["unique_texts"] = unique
    state["vector_sha256"] = sha256_file(vector_path)
    state["inverse_sha256"] = sha256_file(inverse_path)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def encode_queries(queries: list[str], out: Path = OUT, batch_size: int = 64,
                   threads: int = 8) -> dict[str, np.ndarray]:
    """Cache the requested raw query texts; no labels or context enter the encoder."""
    path = out / "query_vectors.parquet"
    manifest_path = out / "query_vectors_manifest.json"
    signature = {"revision": REVISION, "model_sha256": sha256_file(MODEL / "onnx/model.onnx"),
                 "tokenizer_sha256": sha256_file(MODEL / "tokenizer.json"),
                 "max_tokens": MAX_TOKENS, "query_policy": "query: + clean natural text"}
    if path.exists() != manifest_path.exists():
        raise ValueError("Incomplete query embedding cache")
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != signature:
            raise ValueError("Query embedding cache has a different model or text policy")
    unique = sorted(set(clean_text(query) for query in queries))
    if path.exists():
        existing = pl.read_parquet(path)
        mapping = {row[0]: np.asarray(row[1], dtype=np.float32)
                   for row in existing.iter_rows()}
    else:
        mapping = {}
    missing = [query for query in unique if query not in mapping]
    if missing:
        model = load_encoder(threads)
        for pos in range(0, len(missing), batch_size):
            part = missing[pos:pos + batch_size]
            vectors = model.encode([query_text(q) for q in part], batch_size=batch_size,
                                   normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False)
            mapping.update(zip(part, vectors))
            if (pos // batch_size + 1) % 25 == 0:
                print(f"E5 queries: {pos + len(part)}/{len(missing)}", flush=True)
        keys = sorted(mapping)
        pl.DataFrame({"query": keys, "vector": [mapping[key].tolist() for key in keys]}).write_parquet(path)
        manifest_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return {query: mapping[query] for query in unique}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(encode_items(batch_size=args.batch_size, threads=args.threads), indent=2))
