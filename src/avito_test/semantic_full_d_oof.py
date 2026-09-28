"""Build complete D candidate pools for the frozen 15k OOF queries."""
# Compute click-derived signals outside each held-out OOF fold.
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
import psutil

from .data import SEED, digest, enrich_train, sha256_file, source_manifest
from .ranker import (FeatureRetriever, NEGATIVES_PER_QUERY, TEXT_FEATURES,
                     load_feature_items, pool_ids, training_folds)
from .retrieval import TrainSignals
from .semantic_extended_oof import sample_expanded
from .semantic_fit import BASE_SEMANTIC_FEATURES, SOURCE_SEMANTIC_FEATURES
from .semantic_scale_oof import D_FULL as OLD_OOF
from .semantic_scale_select import SELECTED
from .semantic_search import SemanticIndex, semantic_features
from .semantic_scale_evaluate import INDEX
from .semantic_full_oof import choose_queries


OUT = Path("artifacts/semantic_full_d")
CHUNK = 100
DIM = len(TEXT_FEATURES) + len(BASE_SEMANTIC_FEATURES) + len(SOURCE_SEMANTIC_FEATURES)


def signature(pilot: int) -> str:
    state = json.loads(Path("artifacts/semantic/embeddings/encode_state.json")
                       .read_text(encoding="utf-8"))
    return digest({"pilot": pilot, "chunk": CHUNK,
                   "selected_sha256": sha256_file(SELECTED),
                   "old_oof_sha256": sha256_file(OLD_OOF),
                   "sources": source_manifest(Path("src/avito_test")),
                   "vectors_sha256": state["vector_sha256"],
                   "code": {name: sha256_file(Path(__file__).with_name(name))
                            for name in ("semantic_full_d_oof.py", "semantic_full_oof.py",
                                         "semantic_extended_oof.py", "ranker.py",
                                         "retrieval.py", "semantic_search.py",
                                         "data.py")}})[:16]


def _counter() -> dict:
    return {"queries": 0, "known_positives": 0, "found_positives": 0,
            "used_groups": 0, "old_negative_rows": 0, "full_negative_rows": 0,
            "full_rows": 0, "old_pairs": 0, "full_pairs": 0,
            "b_found": 0, "semantic_only_found": 0,
            "semantic_only_groups": 0, "max_pool_size": 0}


def process_chunk(queries: list[dict], retriever: FeatureRetriever,
                  semantic: SemanticIndex, item_rows: dict[str, int],
                  pilot: bool) -> tuple[dict, pl.DataFrame, dict]:
    x_parts, y_parts, g_parts, w_parts = [], [], [], []
    rows = []
    audit = _counter()
    for query in queries:
        truth = {item_rows[item] for item in query["relevant"]}
        relevant = np.asarray(sorted(truth), dtype=np.int32)
        base, scores, allowed = retriever.retrieve_with_scores(query, relevant, False)
        dense = semantic.retrieve(query["search_query"], query["search_location_id"])
        sources = {**base, **dense}
        base_pool = pool_ids(base)
        pool = np.unique(np.concatenate((base_pool, *dense.values()))).astype(np.int32)
        base_positives = truth & set(map(int, base_pool))
        positives = truth & set(map(int, pool))
        old_selected, old_found = sample_expanded(
            sources, truth,
            np.random.default_rng(int(query["group_id"][:16], 16) ^ SEED))
        if old_found != len(positives):
            raise ValueError("The old D sample lost a known positive")
        old_negatives = len(old_selected) - old_found
        full_negatives = len(pool) - len(positives)
        usable = bool(positives and full_negatives)
        audit["queries"] += 1
        audit["known_positives"] += len(truth)
        audit["found_positives"] += len(positives)
        audit["b_found"] += len(base_positives)
        audit["semantic_only_found"] += len(positives - base_positives)
        audit["semantic_only_groups"] += int(bool(positives - base_positives))
        audit["max_pool_size"] = max(audit["max_pool_size"], len(pool))
        if usable:
            lexical = retriever.features(query, pool, base, scores, allowed)
            similarity = semantic_features(semantic, query, pool, sources, True, pool)
            features = np.column_stack((lexical[:, :len(TEXT_FEATURES)], similarity))
            if features.shape != (len(pool), DIM) or features.dtype != np.float32:
                raise ValueError("Full D feature shape or dtype changed")
            labels = np.isin(pool, relevant).astype(np.uint8)
            if int(labels.sum()) != old_found or old_negatives != NEGATIVES_PER_QUERY:
                raise ValueError("Previous D15000 OOF sample size changed")
            # Recalculate pool-dependent ranks on the entire D pool, as D15000 did.
            if pilot:
                selected_positions = np.searchsorted(pool, old_selected)
                sampled_lexical = retriever.features(query, old_selected,
                                                     base, scores, allowed)
                sampled_semantic = semantic_features(semantic, query, old_selected,
                                                     sources, True, pool)
                sampled_features = np.column_stack((
                    sampled_lexical[:, :len(TEXT_FEATURES)], sampled_semantic))
                if not np.allclose(features[selected_positions], sampled_features,
                                   rtol=1e-5, atol=1e-6):
                    raise ValueError("Full and sampled D feature definitions differ")
            weight = np.float32(old_negatives / full_negatives)
            x_parts.append(features)
            y_parts.append(labels)
            g_parts.append(np.full(len(pool), query["selected_index"], dtype=np.int32))
            w_parts.append(np.full(len(pool), weight, dtype=np.float32))
            audit["used_groups"] += 1
            audit["old_negative_rows"] += old_negatives
            audit["full_negative_rows"] += full_negatives
            audit["full_rows"] += len(pool)
            audit["old_pairs"] += old_found * old_negatives
            audit["full_pairs"] += old_found * full_negatives
        rows.append({"group_id": query["group_id"], "fold": query["fold"],
                     "is_control_3000": query["is_control_3000"],
                     "selected_index": query["selected_index"],
                     "known_positives": len(truth), "found_positives": old_found,
                     "b_found": len(base_positives),
                     "semantic_only_found": len(positives - base_positives),
                     "pool_size": len(pool), "used_for_training": usable,
                     "old_negatives": old_negatives, "full_negatives":
                         full_negatives if usable else 0,
                     "old_pairs": old_found*old_negatives if usable else 0,
                     "full_pairs": old_found*full_negatives if usable else 0,
                     "new_group_weight": old_negatives/full_negatives if usable else None})
    arrays = {"features": np.concatenate(x_parts) if x_parts else
              np.empty((0, DIM), dtype=np.float32),
              "labels": np.concatenate(y_parts) if y_parts else np.empty(0, dtype=np.uint8),
              "group_ids": np.concatenate(g_parts) if g_parts else np.empty(0, dtype=np.int32),
              "adjusted_group_weights": np.concatenate(w_parts) if w_parts else
              np.empty(0, dtype=np.float32)}
    return arrays, pl.DataFrame(rows), audit


def _combine(audits: list[dict]) -> dict:
    result = _counter()
    for part in audits:
        for key, value in part.items():
            if key == "max_pool_size":
                result[key] = max(result[key], value)
            else:
                result[key] += value
    return result


def prepare(pilot: int = 0) -> dict:
    queries = choose_queries(pilot)
    sig = signature(pilot)
    folder = OUT / (f"pilot_{pilot}" if pilot else "full") / sig
    folder.mkdir(parents=True, exist_ok=True)
    index = joblib.load(INDEX, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    feature_items = load_feature_items(Path("src/avito_test"), INDEX.parent,
                                       index.item_ids, source_manifest(Path("src/avito_test")))
    item_rows = {item: row for row, item in enumerate(index.item_ids)}
    semantic = SemanticIndex([query["search_query"] for query in queries], threads=1)
    training = training_folds(enrich_train(pl.read_parquet(
        "src/avito_test/train.parquet")).filter(pl.col("split") < 8))
    started = time.perf_counter()
    audits = []
    part_paths = []
    for fold in range(3):
        history = training.filter(pl.col("fold") != fold).drop("fold")
        retriever = FeatureRetriever(index, TrainSignals(history, metadata),
                                     metadata["in_benchmark"].to_numpy(), feature_items)
        fold_queries = [query for query in queries if query["fold"] == fold]
        for start in range(0, len(fold_queries), CHUNK):
            chunk = fold_queries[start:start+CHUNK]
            stem = f"fold_{fold}_part_{start//CHUNK:03d}"
            path = folder / f"{stem}.npz"
            stats = folder / f"{stem}.parquet"
            sidecar = folder / f"{stem}.json"
            if path.exists() and stats.exists() and sidecar.exists():
                saved = json.loads(sidecar.read_text(encoding="utf-8"))
                if (saved["signature"] != sig or saved["queries"] != len(chunk) or
                        saved["first_group"] != chunk[0]["group_id"] or
                        saved["last_group"] != chunk[-1]["group_id"] or
                        saved["part_sha256"] != sha256_file(path) or
                        saved["stats_sha256"] != sha256_file(stats)):
                    raise ValueError(f"Damaged full OOF part: {path}")
            else:
                arrays, frame, audit = process_chunk(chunk, retriever, semantic,
                                                     item_rows, bool(pilot))
                temporary = path.with_suffix(".partial")
                with temporary.open("wb") as stream:
                    np.savez(stream, **arrays)
                os.replace(temporary, path)
                frame.write_parquet(stats)
                saved = {"signature": sig, "queries": len(chunk),
                         "first_group": chunk[0]["group_id"],
                         "last_group": chunk[-1]["group_id"],
                         "part_sha256": sha256_file(path),
                         "stats_sha256": sha256_file(stats), "audit": audit}
                temporary_sidecar = sidecar.with_suffix(".tmp")
                temporary_sidecar.write_text(json.dumps(saved), encoding="utf-8")
                os.replace(temporary_sidecar, sidecar)
            audits.append(saved["audit"])
            part_paths.append(path)
            if len(part_paths) % 10 == 0:
                print(f"Full D OOF: {sum(x['queries'] for x in audits)}/{len(queries)}; "
                      f"{time.perf_counter()-started:.1f}s", flush=True)
        del retriever, history
        gc.collect()
    audit = _combine(audits)
    if audit["queries"] != len(queries):
        raise ValueError("Incomplete full D OOF query audit")
    if not pilot and (audit["known_positives"] != 19996 or
                      audit["found_positives"] != 18392 or
                      audit["used_groups"] != 13869 or
                      audit["b_found"] != 17758 or
                      audit["semantic_only_found"] != 634 or
                      audit["old_negative_rows"] != 13869 * NEGATIVES_PER_QUERY):
        raise ValueError("Full D OOF no longer matches D15000 positives or groups")
    stats = pl.read_parquet(str(folder / "fold_*_part_*.parquet"))
    if stats.height != len(queries) or stats["group_id"].n_unique() != len(queries):
        raise ValueError("Incomplete full D OOF per-query statistics")
    stats.sort("selected_index").write_parquet(folder / "queries.parquet")
    memory = psutil.Process().memory_info()
    result = {"pilot": pilot, "signature": sig, "folder": str(folder),
              "parts": len(part_paths), "audit": audit,
              "seconds": time.perf_counter()-started,
              "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
              "part_bytes": sum(path.stat().st_size for path in part_paths),
              "queries_sha256": sha256_file(folder / "queries.parquet")}
    (folder / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=int, default=0,
                        help="if set, use this many total groups divided across three folds")
    prepare(parser.parse_args().pilot)
