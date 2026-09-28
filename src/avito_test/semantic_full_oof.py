"""Build all B candidates for the frozen 15k OOF queries, with sampling diagnostics."""
# Do not insert known positives that retrieval failed to find.
from __future__ import annotations

import argparse
from collections import Counter
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
from .ranker import (FeatureRetriever, NEGATIVES_PER_QUERY, POOL_RRF,
                     POOL_SOURCES, TEXT_FEATURES, load_feature_items, pool_ids,
                     sample_training_rows, training_folds)
from .retrieval import TrainSignals, fuse
from .semantic_fit import BASE_SEMANTIC_FEATURES
from .semantic_scale_oof import B_FULL as OLD_OOF
from .semantic_scale_select import SELECTED
from .semantic_search import SemanticIndex, semantic_features
from .semantic_scale_evaluate import INDEX


OUT = Path("artifacts/semantic_full_pool")
CHUNK = 100
TOPS = (10, 50, 100, 500)
DIM = len(TEXT_FEATURES) + len(BASE_SEMANTIC_FEATURES)


def choose_queries(pilot: int) -> list[dict]:
    selected = pl.read_parquet(SELECTED)
    if (selected.height != 15000 or selected["group_id"].n_unique() != 15000 or
            selected.filter(pl.col("is_control_3000")).height != 3000):
        raise ValueError("Frozen 15k OOF query selection changed")
    selected = selected.sort("fold", "text", "group_id")
    if pilot:
        if pilot < 3 or pilot % 3:
            raise ValueError("Pilot size must be divisible by three folds")
        selected = pl.concat([selected.filter(pl.col("fold") == fold).head(pilot // 3)
                              for fold in range(3)])
    rows = selected.to_dicts()
    for index, row in enumerate(rows):
        row["selected_index"] = index
    return rows


def signature(pilot: int) -> str:
    state = json.loads(Path("artifacts/semantic/embeddings/encode_state.json")
                       .read_text(encoding="utf-8"))
    return digest({"pilot": pilot, "chunk": CHUNK,
                   "selected_sha256": sha256_file(SELECTED),
                   "old_oof_sha256": sha256_file(OLD_OOF),
                   "sources": source_manifest(Path("src/avito_test")),
                   "vectors_sha256": state["vector_sha256"],
                   "code": {name: sha256_file(Path(__file__).with_name(name))
                            for name in ("semantic_full_oof.py", "ranker.py",
                                         "retrieval.py", "semantic_search.py",
                                         "data.py")}})[:16]


def _counter() -> dict:
    return {"queries": 0, "known_positives": 0, "found_positives": 0,
            "used_groups": 0, "old_negative_rows": 0, "full_negative_rows": 0,
            "full_rows": 0, "old_pairs": 0, "full_pairs": 0,
            "sample_top70": 0, "sample_single_source": 0, "sample_random": 0,
            "top50_any_source_missed": 0, "e5_top50_missed": 0,
            "max_pool_size": 0, "coverage": {}}


def _add_coverage(audit: dict, name: str, ids: np.ndarray, selected: set[int],
                  truth: set[int], usable: bool) -> tuple[int, int]:
    unlabelled = set(map(int, ids)) - truth
    found = len(unlabelled & selected)
    for cohort in ("all", "usable") if usable else ("all",):
        key = f"{cohort}/{name}"
        values = audit["coverage"].setdefault(key, [0, 0])
        values[0] += found
        values[1] += len(unlabelled)
    return found, len(unlabelled)


def process_chunk(queries: list[dict], retriever: FeatureRetriever,
                  semantic: SemanticIndex, item_rows: dict[str, int],
                  pilot: bool) -> tuple[dict, pl.DataFrame, dict]:
    x_parts, y_parts, g_parts, w_parts = [], [], [], []
    rows = []
    audit = _counter()
    for query in queries:
        truth = {item_rows[item] for item in query["relevant"]}
        relevant = np.asarray(sorted(truth), dtype=np.int32)
        sources, scores, allowed = retriever.retrieve_with_scores(query, relevant, False)
        pool = pool_ids(sources)
        positives = truth & set(map(int, pool))
        old_selected, old_found = sample_training_rows(
            sources, truth, NEGATIVES_PER_QUERY,
            np.random.default_rng(int(query["group_id"][:16], 16) ^ SEED))
        if old_found != len(positives):
            raise ValueError("The 150-alternative sample lost a known positive")
        sample_set = set(map(int, old_selected)) - truth
        old_negatives = len(sample_set)
        full_negatives = len(pool) - len(positives)
        usable = bool(positives and full_negatives)
        similarity = semantic_features(semantic, query, pool, sources, False, pool)
        audit["queries"] += 1
        audit["known_positives"] += len(truth)
        audit["found_positives"] += len(positives)
        audit["max_pool_size"] = max(audit["max_pool_size"], len(pool))
        memberships = Counter(int(item) for name in POOL_SOURCES for item in sources[name])
        ranking = fuse(sources, POOL_RRF, depth=len(pool))
        top70 = []
        for item in ranking:
            if int(item) not in truth:
                top70.append(int(item))
            if len(top70) == min(70, NEGATIVES_PER_QUERY):
                break
        top70 = set(top70)
        singles = []
        for item in ranking:
            value = int(item)
            if memberships[value] == 1 and value not in truth and value not in top70:
                singles.append(value)
            if len(singles) == min(30, NEGATIVES_PER_QUERY-len(top70)):
                break
        singles = set(singles)
        if usable:
            audit["sample_top70"] += len(sample_set & top70)
            audit["sample_single_source"] += len(sample_set & singles)
            audit["sample_random"] += len(sample_set - top70 - singles)
        strong = set()
        for name in POOL_SOURCES:
            for top in TOPS:
                candidates = sources[name][:top]
                _add_coverage(audit, f"{name}@{top}", candidates,
                              sample_set, truth, usable)
                if top == 50:
                    strong.update(map(int, candidates))
        for top in TOPS:
            dense = pool[similarity[:, 1] <= top]
            _add_coverage(audit, f"e5@{top}", dense, sample_set, truth, usable)
        audit["top50_any_source_missed"] += len((strong - truth) - sample_set)
        e5_top = set(map(int, pool[similarity[:, 1] <= 50])) - truth
        audit["e5_top50_missed"] += len(e5_top - sample_set)
        if usable:
            lexical = retriever.features(query, pool, sources, scores, allowed)
            features = np.column_stack((lexical[:, :len(TEXT_FEATURES)], similarity))
            if features.shape != (len(pool), DIM) or features.dtype != np.float32:
                raise ValueError("Full B feature shape or dtype changed")
            labels = np.isin(pool, relevant).astype(np.uint8)
            if int(labels.sum()) != old_found or old_negatives != 150:
                raise ValueError("Previous B15000 OOF sample size changed")
            # A pilot verifies that the old sampled objects receive the same features.
            if pilot:
                selected_positions = np.searchsorted(pool, old_selected)
                sampled_lexical = retriever.features(query, old_selected,
                                                     sources, scores, allowed)
                sampled_semantic = semantic_features(semantic, query, old_selected,
                                                     sources, False, pool)
                sampled_features = np.column_stack((
                    sampled_lexical[:, :len(TEXT_FEATURES)], sampled_semantic))
                if not np.allclose(features[selected_positions], sampled_features,
                                   rtol=1e-5, atol=1e-6):
                    raise ValueError("Full and sampled B feature definitions differ")
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
            if key == "coverage":
                for name, counts in value.items():
                    total = result["coverage"].setdefault(name, [0, 0])
                    total[0] += counts[0]
                    total[1] += counts[1]
            elif key == "max_pool_size":
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
                print(f"Full B OOF: {sum(x['queries'] for x in audits)}/{len(queries)}; "
                      f"{time.perf_counter()-started:.1f}s", flush=True)
        del retriever, history
        gc.collect()
    audit = _combine(audits)
    if audit["queries"] != len(queries):
        raise ValueError("Incomplete full B OOF query audit")
    if not pilot and (audit["known_positives"] != 19996 or
                      audit["found_positives"] != 17758 or
                      audit["used_groups"] != 13366 or
                      audit["old_negative_rows"] != 2004900 or
                      audit["old_pairs"] != 2663700):
        raise ValueError("Full B OOF no longer matches B15000 positives or groups")
    stats = pl.read_parquet(str(folder / "fold_*_part_*.parquet"))
    if stats.height != len(queries) or stats["group_id"].n_unique() != len(queries):
        raise ValueError("Incomplete full B OOF per-query statistics")
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
