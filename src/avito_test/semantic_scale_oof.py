"""Extend the frozen B/D OOF samples from 3,000 to 15,000 train groups."""
# The 15k selection includes the original 3k query groups.
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import psutil

from .data import SEED, enrich_train, sha256_file, source_manifest
from .ranker import (FeatureRetriever, NEGATIVES_PER_QUERY, TEXT_FEATURES,
                     load_feature_items, pool_ids, sample_training_rows,
                     training_folds)
from .retrieval import TrainSignals
from .semantic_extended_oof import sample_expanded
from .semantic_fit import BASE_SEMANTIC_FEATURES, SOURCE_SEMANTIC_FEATURES, fit
from .semantic_oof import INDEX
from .semantic_scale_select import OUT, SELECTED
from .semantic_search import SemanticIndex, semantic_features


PARTS = OUT / "oof_parts"
CHUNK = 100
B_OLD = Path("artifacts/semantic/models/B_oof.npz")
D_OLD = Path("artifacts/semantic/models/expanded_oof.npz")
B_FULL = OUT / "B_15000_oof.npz"
D_FULL = OUT / "D_15000_oof.npz"


def _save_part(path: Path, arrays: dict, report: dict) -> None:
    partial = path.with_suffix(".partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(partial, path)
    report["part_sha256"] = sha256_file(path)
    sidecar = path.with_suffix(".json")
    temporary = sidecar.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, sidecar)


def _existing(path: Path, signature: dict) -> dict | None:
    sidecar = path.with_suffix(".json")
    if not path.exists() or not sidecar.exists():
        return None
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    if saved.get("signature") != signature or sha256_file(path) != saved.get("part_sha256"):
        raise ValueError(f"Stale or damaged OOF part: {path}")
    return saved


def _process_chunk(queries: list[dict], retriever: FeatureRetriever,
                   semantic: SemanticIndex, mapping: dict[str, int],
                   signature: dict) -> tuple[dict, dict]:
    b_x, b_y, b_g, d_x, d_y, d_g = [], [], [], [], [], []
    audit = {"queries": len(queries), "known_positives": 0,
             "b_found": 0, "d_found": 0, "semantic_only_found": 0,
             "b_macro_pool_sum": 0., "d_macro_pool_sum": 0.,
             "b_groups": 0, "d_groups": 0, "b_positive_rows": 0,
             "d_positive_rows": 0, "b_rows": 0, "d_rows": 0}
    for query in queries:
        truth = {mapping[item] for item in query["relevant"]}
        relevant = np.array(sorted(truth), dtype=np.int32)
        base, scores, allowed = retriever.retrieve_with_scores(query, relevant, False)
        dense = semantic.retrieve(query["search_query"], query["search_location_id"])
        results = {**base, **dense}
        old_pool = pool_ids(base)
        expanded = np.unique(np.concatenate((old_pool, *dense.values()))).astype(np.int32)
        old_truth = truth & set(map(int, old_pool))
        expanded_truth = truth & set(map(int, expanded))
        audit["known_positives"] += len(truth)
        audit["b_found"] += len(old_truth)
        audit["d_found"] += len(expanded_truth)
        audit["semantic_only_found"] += len(expanded_truth - old_truth)
        audit["b_macro_pool_sum"] += len(old_truth) / len(truth)
        audit["d_macro_pool_sum"] += len(expanded_truth) / len(truth)
        group_seed = int(query["group_id"][:16], 16) ^ SEED
        b_selected, b_positives = sample_training_rows(
            base, truth, NEGATIVES_PER_QUERY, np.random.default_rng(group_seed))
        d_selected, d_positives = sample_expanded(
            results, truth, np.random.default_rng(group_seed))
        if b_positives and len(b_selected) > b_positives:
            lexical = retriever.features(query, b_selected, base, scores, allowed)
            embedding = semantic_features(semantic, query, b_selected,
                                          results, False, old_pool)
            b_x.append(np.column_stack((lexical[:, :len(TEXT_FEATURES)], embedding)))
            target = np.isin(b_selected, relevant).astype(np.uint8)
            b_y.append(target)
            b_g.append(np.full(len(target), audit["b_groups"], dtype=np.int32))
            audit["b_groups"] += 1
            audit["b_positive_rows"] += int(target.sum())
            audit["b_rows"] += len(target)
        if d_positives and len(d_selected) > d_positives:
            lexical = retriever.features(query, d_selected, base, scores, allowed)
            embedding = semantic_features(semantic, query, d_selected,
                                          results, True, expanded)
            d_x.append(np.column_stack((lexical[:, :len(TEXT_FEATURES)], embedding)))
            target = np.isin(d_selected, relevant).astype(np.uint8)
            d_y.append(target)
            d_g.append(np.full(len(target), audit["d_groups"], dtype=np.int32))
            audit["d_groups"] += 1
            audit["d_positive_rows"] += int(target.sum())
            audit["d_rows"] += len(target)
    empty_b = np.empty((0, len(TEXT_FEATURES) + len(BASE_SEMANTIC_FEATURES)), dtype=np.float32)
    empty_d = np.empty((0, len(TEXT_FEATURES) + len(BASE_SEMANTIC_FEATURES) +
                         len(SOURCE_SEMANTIC_FEATURES)), dtype=np.float32)
    arrays = {"b_x": np.concatenate(b_x).astype(np.float32) if b_x else empty_b,
              "b_y": np.concatenate(b_y) if b_y else np.empty(0, dtype=np.uint8),
              "b_g": np.concatenate(b_g) if b_g else np.empty(0, dtype=np.int32),
              "d_x": np.concatenate(d_x).astype(np.float32) if d_x else empty_d,
              "d_y": np.concatenate(d_y) if d_y else np.empty(0, dtype=np.uint8),
              "d_g": np.concatenate(d_g) if d_g else np.empty(0, dtype=np.int32)}
    return arrays, audit


def prepare() -> dict:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    if sha256_file(SELECTED) != selection["selected_sha256"]:
        raise ValueError("Selected OOF groups changed")
    queries = pl.read_parquet(SELECTED).filter(~pl.col("is_control_3000")).to_dicts()
    if len(queries) != 12000:
        raise ValueError("Expected exactly 12,000 new OOF groups")
    PARTS.mkdir(parents=True, exist_ok=True)
    signature = {"selected_sha256": selection["selected_sha256"],
                 "e5_vectors_sha256": json.loads(Path(
                     "artifacts/semantic/embeddings/encode_state.json").read_text(
                         encoding="utf-8"))["vector_sha256"],
                 "negative_limit": NEGATIVES_PER_QUERY, "seed": SEED,
                 "code": {name: sha256_file(Path(__file__).parent / name) for name in
                          ("semantic_scale_oof.py", "semantic_extended_oof.py",
                           "semantic_search.py", "ranker.py", "retrieval.py")}}
    index = joblib.load(INDEX, mmap_mode="r")
    metadata = pl.read_parquet(INDEX.parent / "items.parquet")
    mapping = {item: position for position, item in enumerate(index.item_ids)}
    feature_items = load_feature_items(Path("src/avito_test"), INDEX.parent,
                                       index.item_ids, source_manifest(Path("src/avito_test")))
    semantic = SemanticIndex([query["search_query"] for query in queries])
    training = training_folds(enrich_train(pl.read_parquet(
        "src/avito_test/train.parquet")).filter(pl.col("split") < 8))
    start = time.perf_counter()
    audits = []
    for fold in range(3):
        history = training.filter(pl.col("fold") != fold).drop("fold")
        retriever = FeatureRetriever(index, TrainSignals(history, metadata),
                                     metadata["in_benchmark"].to_numpy(), feature_items)
        fold_queries = [query for query in queries if query["fold"] == fold]
        for start_row in range(0, len(fold_queries), CHUNK):
            part_queries = fold_queries[start_row:start_row+CHUNK]
            part = PARTS / f"fold_{fold}_part_{start_row//CHUNK:03d}.npz"
            part_signature = {**signature, "fold": fold, "first_group":
                              part_queries[0]["group_id"], "last_group":
                              part_queries[-1]["group_id"], "queries": len(part_queries)}
            cached = _existing(part, part_signature)
            if cached is None:
                arrays, audit = _process_chunk(part_queries, retriever, semantic,
                                               mapping, part_signature)
                cached = {"signature": part_signature, "audit": audit}
                _save_part(part, arrays, cached)
            audits.append(cached["audit"])
            if len(audits) % 10 == 0:
                print(f"OOF 15k: {sum(a['queries'] for a in audits)}/12000 new; "
                      f"{time.perf_counter()-start:.1f}s", flush=True)
        del retriever, history
        gc.collect()
    summary = {key: sum(part[key] for part in audits) for key in audits[0]}
    old_b = json.loads(Path("artifacts/ranker/full/c392362a99e2c4d1/oof_1000.json").read_text(encoding="utf-8"))
    old_d = json.loads(Path("artifacts/semantic/models/expanded_oof.json").read_text(encoding="utf-8"))
    report = {"selected_groups": 15000, "new_groups": summary["queries"],
              "b_used_groups": old_b["queries_used_for_training"] + summary["b_groups"],
              "d_used_groups": old_d["groups_with_positive_in_pool"] + summary["d_groups"],
              "known_positives": old_b["known_positives"] + summary["known_positives"],
              "b_found": old_b["retrieved_positives"] + summary["b_found"],
              "d_found": old_d["expanded_found"] + summary["d_found"],
              "semantic_only_found": old_d["expanded_found"] - old_b["retrieved_positives"] +
                                     summary["semantic_only_found"],
              "b_pool_macro_recall": (old_b["macro_pool_recall_sum"] +
                                      summary["b_macro_pool_sum"]) / 15000,
              "d_pool_macro_recall": (old_d["pool_macro_recall_sum"] +
                                      summary["d_macro_pool_sum"]) / 15000,
              "b_rows": old_b["sampled_rows"] + summary["b_rows"],
              "d_rows": old_d["sampled_rows"] + summary["d_rows"],
              "b_positives": old_b["sampled_positives"] + summary["b_positive_rows"],
              "d_positives": old_d["positive_rows"] + summary["d_positive_rows"],
              "new_parts": len(audits), "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset",
                                        psutil.Process().memory_info().rss),
              "signature": signature}
    (OUT / "oof_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def assemble_and_fit() -> dict:
    audit = json.loads((OUT / "oof_audit.json").read_text(encoding="utf-8"))
    paths = sorted(PARTS.glob("fold_*_part_*.npz"))
    if len(paths) != audit["new_parts"]:
        raise ValueError("OOF parts are missing or duplicated")
    with np.load(B_OLD) as old:
        bx, by, bg = old["features"], old["labels"], old["group_ids"]
    with np.load(D_OLD) as old:
        dx, dy, dg = old["features"], old["labels"], old["group_ids"]
    b_x, b_y, b_g = [bx], [by], [bg]
    d_x, d_y, d_g = [dx], [dy], [dg]
    next_b, next_d = int(bg.max())+1, int(dg.max())+1
    for path in paths:
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if sha256_file(path) != sidecar["part_sha256"]:
            raise ValueError(f"Damaged OOF part: {path}")
        with np.load(path) as part:
            b_x.append(part["b_x"])
            b_y.append(part["b_y"])
            b_g.append(part["b_g"] + next_b)
            d_x.append(part["d_x"])
            d_y.append(part["d_y"])
            d_g.append(part["d_g"] + next_d)
        next_b += sidecar["audit"]["b_groups"]
        next_d += sidecar["audit"]["d_groups"]
    b = np.concatenate(b_x).astype(np.float32), np.concatenate(b_y), np.concatenate(b_g)
    d = np.concatenate(d_x).astype(np.float32), np.concatenate(d_y), np.concatenate(d_g)
    for label, (x, y, groups), expected_groups, expected_rows in (
        ("B", b, audit["b_used_groups"], audit["b_rows"]),
        ("D", d, audit["d_used_groups"], audit["d_rows"])):
        if (len(y) != expected_rows or len(np.unique(groups)) != expected_groups or
                not np.array_equal(np.unique(groups), np.arange(expected_groups))):
            raise ValueError(f"Incomplete {label} assembled OOF matrix")
        with (B_FULL if label == "B" else D_FULL).open("wb") as stream:
            np.savez_compressed(stream, features=x, labels=y, group_ids=groups)
    b_model = fit("B_15000_existing_pool_semantic_features", *b,
                  TEXT_FEATURES + BASE_SEMANTIC_FEATURES)
    d_model = fit("D_15000_expanded_pool_semantic_features", *d,
                  TEXT_FEATURES + BASE_SEMANTIC_FEATURES + SOURCE_SEMANTIC_FEATURES)
    result = {"audit": audit, "B": b_model, "D": d_model,
              "B_oof_sha256": sha256_file(B_FULL), "D_oof_sha256": sha256_file(D_FULL)}
    (OUT / "training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "fit"])
    args = parser.parse_args()
    if args.step == "prepare":
        prepare()
    else:
        assemble_and_fit()
