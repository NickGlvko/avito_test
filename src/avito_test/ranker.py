"""Leakage-controlled CatBoost selection on the frozen union retrieval index."""
# Pool indices are temporary; only item_id identifies the final advertisement.
from __future__ import annotations

import functools
import gc
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import psutil
from catboost import CatBoostRanker, Pool
from threadpoolctl import threadpool_limits

from .data import SEARCH, SEED, digest, normalize, sha256_file, source_manifest, write_json
from .extended_validation import ScopedRetriever, augmented_scope, prepare_union, rank_in_scope
from .retrieval import DEPTH, INDEX_CONFIG, TrainSignals, fuse

POOL_SOURCES = ("bm25_geo16", "char_geo16", "bm25_stats", "char_stats", "history")
TEXT_SOURCES = POOL_SOURCES[:-1]
POOL_RRF = {name: 1. for name in POOL_SOURCES}
TEXT_RRF = {name: 1. for name in TEXT_SOURCES}
CONTROLS = {"rrf_history_half": {"bm25_stats": 1., "char_stats": 1., "history": .5},
            "rrf_geo16": {"bm25_geo16": 1., "char_geo16": 1.}}
MODEL_PARAMS = {"loss_function": "PairLogit", "iterations": 240, "depth": 6,
                "learning_rate": .06, "l2_leaf_reg": 4., "random_seed": SEED,
                "task_type": "CPU", "thread_count": 6, "allow_writing_files": False,
                "verbose": False}

# All features are numeric float32. IDs, corpus membership and labels never enter X.
TEXT_FEATURES = [
    "bm25_title", "bm25_params", "bm25_description", "bm25_raw", "char_raw",
    "title_ratio", "params_ratio", "description_ratio", "bm25_ratio", "char_ratio",
    "title_gap", "params_gap", "description_gap", "bm25_gap", "char_gap",
    "geo_bm25_rank", "geo_char_rank", "geo_bm25_missing", "geo_char_missing",
    "geo_best_rank", "geo_source_count", "geo_bm25_ratio", "geo_char_ratio",
    "same_location", "same_category", "delivery_query", "query_token_count",
    "query_title_coverage", "query_params_coverage", "query_in_title", "query_in_params",
    "has_filters", "filter_token_count", "filter_params_coverage", "filter_in_params",
    "rating", "rating_missing", "reviews_log", "reviews_missing", "price_log", "price_missing",
]
HISTORY_FEATURES = [
    "stats_bm25_rank", "stats_char_rank", "history_rank", "stats_bm25_missing",
    "stats_char_missing", "history_missing", "all_best_rank", "all_source_count",
    "stats_bm25_ratio", "stats_char_ratio", "history_raw", "history_ratio",
    "history_nearest_similarity", "train_clicks_log", "category_prior",
]
FEATURES = TEXT_FEATURES + HISTORY_FEATURES
FEATURE_VERSION = 1
NEGATIVES_PER_QUERY = 150
TRAIN_QUERIES = 3000
CHUNK_SIZE = 64


def canonical_fold(row: dict) -> int:
    """All raw variants of one canonical full query stay in one OOF fold."""
    key = [normalize(row["search_query"]), row["search_location_id"],
           row["search_is_delivery_search"], normalize(row["search_infm_params_text"]),
           row["search_category"]]
    return int(digest([SEED, key])[:16], 16) % 3


def training_folds(train: pl.DataFrame) -> pl.DataFrame:
    keys = train.select("group_id", *SEARCH).unique("group_id")
    records = [{"group_id": row["group_id"], "fold": canonical_fold(row)}
               for row in keys.iter_rows(named=True)]
    return train.join(pl.DataFrame(records), on="group_id", validate="m:1")


def load_feature_items(data_dir: Path, cache: Path, ordered_ids: list[str],
                       sources: dict) -> pl.DataFrame:
    path, signature_path = cache / "features.parquet", cache / "features_signature.json"
    signature = digest({"sources": sources, "columns": ["item_id", "item_title_raw",
        "item_infm_params_text", "item_rating", "item_rating_reviews_count", "item_price",
        "item_location_id", "item_category_id"], "policy": "benchmark snapshot priority"})
    if not path.exists() or not signature_path.exists() or json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        columns = ["item_id", "item_title_raw", "item_infm_params_text", "item_rating",
                   "item_rating_reviews_count", "item_price", "item_location_id", "item_category_id"]
        benchmark = pl.read_parquet(data_dir / "benchmark_items.parquet", columns=columns)
        training = pl.read_parquet(data_dir / "train.parquet", columns=columns).unique(subset="item_id")
        extra = training.join(benchmark.select("item_id"), on="item_id", how="anti")
        items = pl.concat([benchmark, extra]).sort("item_id").with_columns(
            pl.col("item_price").cast(pl.Float64))
        cache.mkdir(parents=True, exist_ok=True)
        items.write_parquet(path)
        write_json(signature_path, signature)
    items = pl.read_parquet(path)
    if items["item_id"].to_list() != ordered_ids:
        raise ValueError("Feature rows do not align with the retrieval index")
    return items


def group_queries(train: pl.DataFrame, stage: str, limit: int = 0) -> list[dict]:
    column = "split"
    held = train.filter(pl.col(column) == (8 if stage == "dev" else 9))
    grouped = held.group_by("group_id", *SEARCH).agg(
        pl.col("item_id").unique().sort().alias("relevant"), pl.col("text").first()).sort("group_id")
    if limit:
        grouped = grouped.head(limit)
    return grouped.sort("text", "group_id").to_dicts()


def oof_queries(training: pl.DataFrame, per_fold: int) -> list[dict]:
    grouped = training.group_by("group_id", *SEARCH).agg(
        pl.col("item_id").unique().sort().alias("relevant"),
        pl.col("text").first(), pl.col("fold").first()).sort("group_id")
    queries = []
    for fold in range(3):
        queries.extend(grouped.filter(pl.col("fold") == fold).head(per_fold).to_dicts())
    return sorted(queries, key=lambda row: (row["fold"], row["text"], row["group_id"]))


def pool_ids(results: dict, include_history: bool = True) -> np.ndarray:
    sources = POOL_SOURCES if include_history else TEXT_SOURCES
    return np.unique(np.concatenate([results[name] for name in sources])).astype(np.int32)


def sample_training_rows(results: dict, truth: set[int], limit: int,
                         rng: np.random.Generator) -> tuple[np.ndarray, int]:
    pool = pool_ids(results)
    positives = sorted(truth & set(map(int, pool)))
    if not positives:
        return np.empty(0, dtype=np.int32), 0
    positive = set(positives)
    ranking = fuse(results, POOL_RRF, depth=len(pool))
    chosen = []
    for item in ranking:
        if int(item) not in positive:
            chosen.append(int(item))
        if len(chosen) >= min(70, limit):
            break
    memberships = Counter(int(item) for name in POOL_SOURCES for item in results[name])
    singles = [int(item) for item in ranking if memberships[int(item)] == 1 and
               int(item) not in positive and int(item) not in chosen]
    chosen.extend(singles[: min(30, max(0, limit - len(chosen)))])
    remain = [int(item) for item in ranking[::-1] if int(item) not in positive and int(item) not in chosen]
    count = min(limit - len(chosen), len(remain))
    if count:
        chosen.extend(int(x) for x in rng.choice(remain, size=count, replace=False))
    selected = np.array(sorted(positive | set(chosen)), dtype=np.int32)
    return selected, len(positives)


class FeatureRetriever(ScopedRetriever):
    def __init__(self, index, signals, benchmark_mask, items: pl.DataFrame):
        super().__init__(index, signals, benchmark_mask)
        if items["item_id"].to_list() != index.item_ids:
            raise ValueError("Features and index are misaligned")
        self.titles = items["item_title_raw"].to_list()
        self.params = items["item_infm_params_text"].to_list()
        self.rating = items["item_rating"].to_numpy()
        self.reviews = items["item_rating_reviews_count"].to_numpy()
        self.price = items["item_price"].to_numpy()
        self.last_text = None

    @functools.lru_cache(maxsize=30000)
    def document_tokens(self, item: int) -> tuple[str, frozenset, str, frozenset]:
        title = normalize(self.titles[item])
        params = normalize(self.params[item])
        return title, frozenset(title.split()), params, frozenset(params.split())

    def retrieve_with_scores(self, query: dict, relevant: np.ndarray,
                             augmented: bool) -> tuple[dict, dict, np.ndarray | None]:
        text = normalize(query["search_query"])
        if self.last_text != text:
            fields = [field.score(text) for field in self.index.fields]
            self.field_scores = fields
            self.words = sum(w * values for w, values in zip(INDEX_CONFIG["field_weights"], fields))
            self.chars = (self.index.char_vectorizer.transform([text]) @
                          self.index.char_postings).toarray().ravel()
            self.similarities = ((self.signals.history_vectorizer.transform([text]) @
                                  self.signals.history_postings).toarray().ravel()
                                 if self.signals.history_vectorizer is not None else
                                 np.empty(0, dtype=np.float32))
            self.last_text = text
        allowed = augmented_scope(self.benchmark_rows, relevant) if augmented else None
        geo = np.where(self.signals.locations == query["search_location_id"], 16., 1.).astype(np.float32)
        stats = geo * self.stats_weights.get(query["search_category"], self.default_stats)
        scored = {"bm25_geo16": self.words * geo, "char_geo16": self.chars * geo,
                  "bm25_stats": self.words * stats, "char_stats": self.chars * stats}
        raw_history = self.history_scores(self.similarities, relevant, augmented)
        scored["history"] = raw_history * stats
        results = {name: rank_in_scope(score, allowed, DEPTH, name == "history")
                   for name, score in scored.items()}
        raw = {"title": self.field_scores[0], "params": self.field_scores[1],
               "description": self.field_scores[2], "bm25": self.words,
               "char": self.chars, "history_unboosted": raw_history, **scored}
        return results, raw, allowed

    def features(self, query: dict, ids: np.ndarray, results: dict,
                 scores: dict, allowed: np.ndarray | None) -> np.ndarray:
        n = len(ids)
        x = np.zeros((n, len(FEATURES)), dtype=np.float32)
        columns = {name: x[:, i] for i, name in enumerate(FEATURES)}
        for field, name in [("title", "bm25_title"), ("params", "bm25_params"),
                            ("description", "bm25_description"), ("bm25", "bm25_raw"),
                            ("char", "char_raw")]:
            values = scores[field]
            leader = max(float(values.max() if allowed is None else values[allowed].max()), 1e-7)
            selected = values[ids]
            columns[name][:] = selected
            ratio = {"title": "title_ratio", "params": "params_ratio",
                     "description": "description_ratio", "bm25": "bm25_ratio",
                     "char": "char_ratio"}[field]
            gap = {"title": "title_gap", "params": "params_gap",
                   "description": "description_gap", "bm25": "bm25_gap",
                   "char": "char_gap"}[field]
            columns[ratio][:] = selected / leader
            columns[gap][:] = leader - selected
        positions = {source: {int(item): rank for rank, item in enumerate(results[source], 1)}
                     for source in POOL_SOURCES}
        for source, prefix in [("bm25_geo16", "geo_bm25"), ("char_geo16", "geo_char"),
                               ("bm25_stats", "stats_bm25"), ("char_stats", "stats_char"),
                               ("history", "history")]:
            ranks = np.array([positions[source].get(int(item), 0) for item in ids], dtype=np.float32)
            columns[prefix + "_rank"][:] = ranks
            columns[prefix + "_missing"][:] = ranks == 0
        geo_ranks = np.stack([columns["geo_bm25_rank"], columns["geo_char_rank"]])
        all_ranks = np.stack([columns[source + "_rank"] for source in
                              ["geo_bm25", "geo_char", "stats_bm25", "stats_char", "history"]])
        columns["geo_best_rank"][:] = np.where(geo_ranks > 0, geo_ranks, 501).min(axis=0)
        columns["all_best_rank"][:] = np.where(all_ranks > 0, all_ranks, 501).min(axis=0)
        columns["geo_source_count"][:] = (geo_ranks > 0).sum(axis=0)
        columns["all_source_count"][:] = (all_ranks > 0).sum(axis=0)
        for source, name in [("bm25_geo16", "geo_bm25_ratio"), ("char_geo16", "geo_char_ratio"),
                             ("bm25_stats", "stats_bm25_ratio"), ("char_stats", "stats_char_ratio"),
                             ("history", "history_ratio")]:
            values = scores[source]
            leader = max(float(values.max() if allowed is None else values[allowed].max()), 1e-7)
            columns[name][:] = scores[source][ids] / leader
        columns["history_raw"][:] = scores["history_unboosted"][ids]
        columns["history_nearest_similarity"][:] = float(self.similarities.max()) if len(self.similarities) else 0.
        columns["train_clicks_log"][:] = self.signals.popularity[ids]
        probs = self.signals.category_prob.get(query["search_category"], {})
        columns["category_prior"][:] = [probs.get(cat, 0.) for cat in self.signals.categories[ids]]
        columns["same_location"][:] = self.signals.locations[ids] == query["search_location_id"]
        columns["same_category"][:] = self.signals.categories[ids] == query["search_category"]
        columns["delivery_query"][:] = query["search_is_delivery_search"]
        text = normalize(query["search_query"])
        tokens = set(text.split())
        columns["query_token_count"][:] = len(tokens)
        filters = normalize(query["search_infm_params_text"])
        filter_tokens = set(filters.split())
        columns["has_filters"][:] = bool(filters)
        columns["filter_token_count"][:] = len(filter_tokens)
        for position, item in enumerate(ids):
            title, title_tokens, params, params_tokens = self.document_tokens(int(item))
            if tokens:
                columns["query_title_coverage"][position] = len(tokens & title_tokens) / len(tokens)
                columns["query_params_coverage"][position] = len(tokens & params_tokens) / len(tokens)
                columns["query_in_title"][position] = text in title
                columns["query_in_params"][position] = text in params
            if filter_tokens:
                columns["filter_params_coverage"][position] = len(filter_tokens & params_tokens) / len(filter_tokens)
                columns["filter_in_params"][position] = filters in params
        for values, name, missing in [(self.rating, "rating", "rating_missing"),
                                      (self.reviews, "reviews_log", "reviews_missing"),
                                      (self.price, "price_log", "price_missing")]:
            selected = values[ids].astype(np.float64)
            mask = ~np.isfinite(selected)
            columns[missing][:] = mask
            selected = np.where(mask, -1., selected)
            columns[name][:] = selected if name == "rating" else np.where(
                mask, -1., np.log1p(np.maximum(selected, 0.)))
        return x


def model_ranking(ids: np.ndarray, scores: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    if eligible is None:
        eligible = np.ones(len(ids), dtype=bool)
    positions = np.flatnonzero(eligible)
    order = np.lexsort((ids[positions], -scores[positions]))
    return ids[positions[order[:50]]]


def candidate_metrics(prediction: np.ndarray, pool: np.ndarray, truth: set[int],
                      signals: TrainSignals, query: dict, history_rank: np.ndarray,
                      history_only: set[int]) -> dict:
    selected = set(map(int, prediction))
    candidates = set(map(int, pool))
    found = len(truth & selected)
    candidate_found = len(truth & candidates)
    seen = {item for item in truth if signals.popularity[item] > 0}
    nearby = {item for item in truth if signals.locations[item] == query["search_location_id"]}
    history = truth & set(map(int, history_rank))
    metrics = {"positives": len(truth), "retrieved": found,
               "recall@50": found / len(truth), "candidate_found": candidate_found,
               "candidate_recall": candidate_found / len(truth),
               "lost_at_selection": candidate_found - found,
               "missed_by_sources": len(truth) - candidate_found,
               "avoidable_selection_loss": min(50, candidate_found) - found,
               "unavoidable_cap_loss": max(0, candidate_found - 50),
               "candidate_count": len(candidates)}
    metrics["history_only_positives"] = len(truth & history_only)
    metrics["history_only_found"] = len(truth & history_only & selected)
    for label, subset in [("seen_item", seen), ("new_item", truth - seen),
                          ("same_location", nearby), ("other_location", truth - nearby),
                          ("history_source", history), ("outside_history_source", truth - history)]:
        metrics[label + "_positives"] = len(subset)
        metrics[label + "_found"] = len(subset & selected)
        metrics[label + "_candidate_found"] = len(subset & candidates)
        metrics[label + "_recall@50"] = len(subset & selected) / len(subset) if subset else None
    return metrics


def prepare_oof(train: pl.DataFrame, index_path: Path, items: pl.DataFrame,
                feature_items: pl.DataFrame, cache: Path, per_fold: int) -> Path:
    path = cache / f"oof_{per_fold}.npz"
    audit_path = cache / f"oof_{per_fold}.json"
    if path.exists() and audit_path.exists():
        return path
    cache.mkdir(parents=True, exist_ok=True)
    training = training_folds(train.filter(pl.col("split") < 8))
    queries = oof_queries(training, per_fold)
    groups_by_fold = {fold: [q for q in queries if q["fold"] == fold] for fold in range(3)}
    index = joblib.load(index_path, mmap_mode="r")
    mapping = {name: i for i, name in enumerate(index.item_ids)}
    benchmark = pl.read_parquet(index_path.parent / "items.parquet")["in_benchmark"].to_numpy()
    batches, labels, group_ids = [], [], []
    audit = {"requested_queries": len(queries), "folds": [], "groups_with_no_found_positive": 0,
             "queries_used_for_training": 0, "sampled_rows": 0, "sampled_positives": 0,
             "known_positives": 0, "retrieved_positives": 0, "macro_pool_recall_sum": 0.,
             "sampling": "70 top RRF, 30 single-source, remaining random from lower ranks",
             "negative_limit_per_query": NEGATIVES_PER_QUERY}
    start = time.perf_counter()
    for fold in range(3):
        history_rows = training.filter(pl.col("fold") != fold).drop("fold")
        fold_queries = groups_by_fold[fold]
        held_groups = {q["group_id"] for q in fold_queries}
        if held_groups & set(history_rows["group_id"]):
            raise ValueError("OOF query group entered its own training history")
        signals = TrainSignals(history_rows, items)
        retriever = FeatureRetriever(index, signals, benchmark, feature_items)
        fold_info = {"fold": fold, "history_rows": history_rows.height,
                     "heldout_groups": len(fold_queries), "sampled_groups": 0,
                     "found_positive_groups": 0, "train_history_group_overlap": 0}
        for number, query in enumerate(fold_queries):
            truth = {mapping[item] for item in query["relevant"]}
            relevant = np.array(sorted(truth), dtype=np.int32)
            results, scores, allowed = retriever.retrieve_with_scores(query, relevant, augmented=False)
            pool = pool_ids(results)
            audit["known_positives"] += len(truth)
            found = len(truth & set(map(int, pool)))
            audit["retrieved_positives"] += found
            audit["macro_pool_recall_sum"] += found / len(truth)
            rng = np.random.default_rng(int(query["group_id"][:16], 16) ^ SEED)
            selected, positives_found = sample_training_rows(results, truth, NEGATIVES_PER_QUERY, rng)
            if not positives_found:
                audit["groups_with_no_found_positive"] += 1
                continue
            if len(selected) == positives_found:
                continue
            features = retriever.features(query, selected, results, scores, allowed)
            target = np.array([int(item in truth) for item in selected], dtype=np.uint8)
            batches.append(features)
            labels.append(target)
            group_ids.append(np.full(len(selected), audit["queries_used_for_training"], dtype=np.int32))
            audit["queries_used_for_training"] += 1
            audit["sampled_rows"] += len(selected)
            audit["sampled_positives"] += int(target.sum())
            fold_info["sampled_groups"] += 1
            fold_info["found_positive_groups"] += 1
            if (number + 1) % 200 == 0:
                print(f"OOF fold {fold}: {number + 1}/{len(fold_queries)}, {time.perf_counter()-start:.1f}s", flush=True)
        audit["folds"].append(fold_info)
        del signals, retriever, history_rows
        gc.collect()
    if not batches:
        raise ValueError("No positive candidate available for ranker training")
    matrix = np.concatenate(batches)
    target = np.concatenate(labels)
    groups = np.concatenate(group_ids)
    np.savez_compressed(path, features=matrix, labels=target, group_ids=groups)
    audit.update({"elapsed_seconds": time.perf_counter() - start,
                  "features": FEATURES, "feature_version": FEATURE_VERSION,
                  "macro_pool_recall": audit["macro_pool_recall_sum"] / len(queries),
                  "micro_positive_coverage": audit["retrieved_positives"] / audit["known_positives"]})
    write_json(audit_path, audit)
    print(f"OOF ready: {audit['queries_used_for_training']} groups, {audit['sampled_rows']} rows, "
          f"pool macro recall {audit['macro_pool_recall']:.4f}", flush=True)
    return path


def fit_models(oof_path: Path, cache: Path, short: bool = False) -> tuple[Path, Path]:
    text_path, history_path = cache / "text_model.cbm", cache / "history_model.cbm"
    if text_path.exists() and history_path.exists():
        return text_path, history_path
    with np.load(oof_path) as data:
        features = data["features"]
        labels = data["labels"]
        groups = data["group_ids"]
    if np.any(np.diff(groups) < 0):
        raise ValueError("CatBoost query groups must be contiguous")
    start = time.perf_counter()
    times = {}
    for name, subset, path in [("text", TEXT_FEATURES, text_path),
                               ("history", FEATURES, history_path)]:
        columns = [FEATURES.index(feature) for feature in subset]
        params = {**MODEL_PARAMS, "iterations": 30 if short else MODEL_PARAMS["iterations"]}
        model = CatBoostRanker(**params)
        data = Pool(features[:, columns], label=labels, group_id=groups, feature_names=subset)
        model.fit(data)
        model.save_model(str(path))
        times[name] = time.perf_counter() - start
        print(f"CatBoost {name}: {model.tree_count_} trees, {times[name]:.1f}s cumulative", flush=True)
    write_json(cache / "models.json", {"params": {**MODEL_PARAMS, "iterations": 30 if short else MODEL_PARAMS["iterations"]},
                                        "feature_sets": {"text": TEXT_FEATURES, "history": FEATURES},
                                        "training_rows": len(labels), "training_groups": int(groups.max()) + 1,
                                        "positive_rows": int(labels.sum()), "elapsed_seconds": time.perf_counter() - start,
                                        "model_sha256": {"text": sha256_file(text_path), "history": sha256_file(history_path)}})
    return text_path, history_path


_WORKER = None


def initialize_worker(index_path: str, signals_path: str, metadata_path: str,
                      feature_path: str, text_model_path: str, history_model_path: str) -> None:
    global _WORKER
    threadpool_limits(limits=1)
    index = joblib.load(index_path, mmap_mode="r")
    signals = joblib.load(signals_path, mmap_mode="r")
    metadata = pl.read_parquet(metadata_path)
    items = pl.read_parquet(feature_path)
    retriever = FeatureRetriever(index, signals, metadata["in_benchmark"].to_numpy(), items)
    text_model = CatBoostRanker()
    text_model.load_model(text_model_path)
    history_model = CatBoostRanker()
    history_model.load_model(history_model_path)
    _WORKER = retriever, text_model, history_model


def score_chunk(chunk_number: int, queries: list[dict], parts: str) -> tuple[int, int, int]:
    retriever, text_model, history_model = _WORKER
    rows = []
    text_columns = [FEATURES.index(feature) for feature in TEXT_FEATURES]
    for query in queries:
        relevant = np.array(query["relevant_rows"], dtype=np.int32)
        truth = set(map(int, relevant))
        for scheme, augmented in [("union", False), ("benchmark_plus_positives", True)]:
            results, scores, allowed = retriever.retrieve_with_scores(query, relevant, augmented)
            full = pool_ids(results)
            text = pool_ids(results, include_history=False)
            history_only = set(map(int, results["history"])) - set(map(int, text))
            x = retriever.features(query, full, results, scores, allowed)
            text_scores = text_model.predict(x[:, text_columns], thread_count=1)
            history_scores = history_model.predict(x, thread_count=1)
            text_members = np.isin(full, text, assume_unique=True)
            predicted = {
                "rrf_history_half": fuse(results, CONTROLS["rrf_history_half"], depth=50),
                "rrf_geo16": fuse(results, CONTROLS["rrf_geo16"], depth=50),
                "rrf_text_pool": fuse(results, TEXT_RRF, depth=50),
                "rrf_full_pool": fuse(results, POOL_RRF, depth=50),
                "catboost_text_pool": model_ranking(full, text_scores, text_members),
                "catboost_full_no_history_features": model_ranking(full, text_scores),
                "catboost_full_with_history_features": model_ranking(full, history_scores),
            }
            for variant, ranking in predicted.items():
                if variant == "rrf_history_half":
                    sources = ("bm25_stats", "char_stats", "history")
                elif variant == "rrf_geo16":
                    sources = ("bm25_geo16", "char_geo16")
                elif variant in ("rrf_text_pool", "catboost_text_pool"):
                    sources = TEXT_SOURCES
                else:
                    sources = POOL_SOURCES
                pool = np.unique(np.concatenate([results[name] for name in sources]))
                rows.append({"group_id": query["group_id"], "text": query["text"],
                             "stage": query["stage"], "scheme": scheme, "variant": variant,
                             **candidate_metrics(ranking, pool, truth, retriever.signals, query,
                                                 results["history"], history_only)})
    destination = Path(parts)
    frame = pl.DataFrame(rows, infer_schema_length=None,
        schema_overrides={f"{label}_recall@50": pl.Float64 for label in
                          ["seen_item", "new_item", "same_location", "other_location",
                           "history_source", "outside_history_source"]})
    frame.write_parquet(destination / f"metrics_{chunk_number:05d}.parquet")
    write_json(destination / f"done_{chunk_number:05d}.json", {"queries": len(queries)})
    memory = psutil.Process().memory_info()
    return chunk_number, len(queries), getattr(memory, "peak_wset", memory.rss)


def evaluate_stage(stage: str, queries: list[dict], output: Path, index_path: Path,
                   signals_path: Path, metadata_path: Path, features_path: Path,
                   text_model_path: Path, history_model_path: Path, workers: int,
                   run_signature: str) -> tuple[Path, dict]:
    path = output / stage
    parts = path / "parts" / run_signature[:16]
    parts.mkdir(parents=True, exist_ok=True)
    chunks = [queries[i:i + CHUNK_SIZE] for i in range(0, len(queries), CHUNK_SIZE)]
    pending, completed = [], 0
    for number, chunk in enumerate(chunks):
        if (parts / f"done_{number:05d}.json").exists() and (parts / f"metrics_{number:05d}.parquet").exists():
            completed += len(chunk)
        else:
            pending.append((number, chunk))
    start = time.perf_counter()
    peak = 0
    print(f"CatBoost {stage}: {len(queries)} queries, {completed} cached, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                             initargs=(str(index_path), str(signals_path), str(metadata_path),
                                       str(features_path), str(text_model_path), str(history_model_path))) as pool:
        futures = [pool.submit(score_chunk, number, chunk, str(parts)) for number, chunk in pending]
        for future in as_completed(futures):
            _, count, memory = future.result()
            completed += count
            peak = max(peak, memory)
            if completed % (CHUNK_SIZE * 12) < CHUNK_SIZE:
                print(f"CatBoost {stage}: {completed}/{len(queries)}, {time.perf_counter()-start:.1f}s", flush=True)
    frame = pl.read_parquet(str(parts / "metrics_*.parquet"))
    expected = len(queries) * 2 * 7
    if frame.height != expected or frame.unique(["group_id", "scheme", "variant"]).height != expected:
        raise ValueError(f"Incomplete model predictions: {frame.height} != {expected}")
    frame.sort("group_id", "scheme", "variant").write_parquet(path / "per_query.parquet")
    timing = {"queries": len(queries), "elapsed_seconds": time.perf_counter() - start,
              "worker_peak_rss_bytes": peak, "workers": workers,
              "cached_queries": len(queries) - sum(len(chunk) for _, chunk in pending)}
    write_json(path / "timing.json", timing)
    return path / "per_query.parquet", timing


def summary_table(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.group_by("stage", "scheme", "variant").agg(
        pl.len().alias("queries"), pl.col("recall@50").mean(),
        pl.col("candidate_recall").mean(), pl.col("positives").sum(),
        pl.col("retrieved").sum(), pl.col("candidate_found").sum(),
        pl.col("lost_at_selection").sum(), pl.col("missed_by_sources").sum(),
        pl.col("avoidable_selection_loss").sum(), pl.col("unavoidable_cap_loss").sum(),
        pl.col("candidate_count").mean(),
    ).sort("stage", "scheme", "variant")


def choose_dev(summary: pl.DataFrame, config_path: Path) -> dict:
    dev_union = summary.filter((pl.col("stage") == "dev") & (pl.col("scheme") == "union"))
    candidate_models = ["catboost_text_pool", "catboost_full_no_history_features",
                        "catboost_full_with_history_features"]
    candidates = dev_union.filter(pl.col("variant").is_in(candidate_models)).sort(
        ["recall@50", "variant"], descending=[True, False])
    if candidates.height != len(candidate_models):
        raise ValueError("All model ablations must be evaluated on union/dev")
    chosen = candidates.row(0, named=True)
    controls = dev_union.filter(pl.col("variant").is_in(list(CONTROLS))).sort("recall@50", descending=True)
    second = summary.filter((pl.col("stage") == "dev") &
                            (pl.col("scheme") == "benchmark_plus_positives"))
    selected_secondary = second.filter(pl.col("variant") == chosen["variant"])["recall@50"].item()
    best_secondary_control = second.filter(pl.col("variant").is_in(list(CONTROLS)))["recall@50"].max()
    config = {"selected_on": "union/dev macro Recall@50", "variant": chosen["variant"],
              "union_dev_recall50": chosen["recall@50"],
              "best_control": controls["variant"][0],
              "best_control_union_dev_recall50": controls["recall@50"][0],
              "augmented_dev_recall50": selected_secondary,
              "best_control_augmented_dev_recall50": best_secondary_control,
              "gain_over_best_control": chosen["recall@50"] - controls["recall@50"][0],
              "serious_secondary_drop": selected_secondary < best_secondary_control - .02,
              "model_params": MODEL_PARAMS, "feature_version": FEATURE_VERSION,
              "text_features": TEXT_FEATURES, "history_features": HISTORY_FEATURES,
              "negative_limit": NEGATIVES_PER_QUERY, "source_depth": DEPTH,
              "sources": POOL_SOURCES}
    write_json(config_path, config)
    return config


def run_ranker(data_dir: Path, output: Path, train: pl.DataFrame,
               train_queries: int = TRAIN_QUERIES, dev_limit: int = 0,
               workers: int = 4, include_test: bool = True) -> dict:
    if workers < 1 or train_queries < 3 or dev_limit < 0:
        raise ValueError("Invalid number of workers, training queries, or dev limit")
    sources = source_manifest(data_dir)
    expanded = output / "expanded"
    index_path, metadata_path, corpus = prepare_union(data_dir, expanded, sources)
    code = {name: sha256_file(Path(__file__).with_name(name))
            for name in ["ranker.py", "extended_validation.py", "retrieval.py", "data.py"]}
    signature = digest({"sources": sources, "code": code, "params": MODEL_PARAMS,
                        "features": FEATURES, "negative_limit": NEGATIVES_PER_QUERY,
                        "train_queries": train_queries, "dev_limit": dev_limit, "seed": SEED})
    run = output / "ranker" / ("smoke" if dev_limit else "full") / signature[:16]
    run.mkdir(parents=True, exist_ok=True)
    metadata = pl.read_parquet(metadata_path)
    index = joblib.load(index_path, mmap_mode="r")
    features = load_feature_items(data_dir, index_path.parent, index.item_ids, sources)
    oof = prepare_oof(train, index_path, metadata, features, run, max(1, train_queries // 3))
    text_model, history_model = fit_models(oof, run, short=bool(dev_limit))
    del features, metadata, index
    gc.collect()
    training = train.filter(pl.col("split") < 8)
    signals_path = run / "full_train_signals.joblib"
    if not signals_path.exists():
        signals = TrainSignals(training, pl.read_parquet(metadata_path))
        joblib.dump(signals, signals_path)
        del signals
        gc.collect()
    mapping = {item: number for number, item in enumerate(pl.read_parquet(metadata_path)["item_id"])}
    def enrich(queries: list[dict], stage: str) -> list[dict]:
        for query in queries:
            query["relevant_rows"] = [mapping[item] for item in query["relevant"]]
            query["stage"] = stage
        return queries
    config_path = run / "selection.json"
    dev = enrich(group_queries(train, "dev", dev_limit), "dev")
    dev_path, dev_timing = evaluate_stage("dev", dev, run, index_path, signals_path, metadata_path,
                                          index_path.parent / "features.parquet", text_model, history_model,
                                          workers, signature)
    dev_summary = summary_table(pl.read_parquet(dev_path))
    config = choose_dev(dev_summary, config_path)
    print(f"Dev selected {config['variant']}: {config['union_dev_recall50']:.4f}; "
          f"control {config['best_control']}: {config['best_control_union_dev_recall50']:.4f}", flush=True)
    test_path, test_timing = None, None
    if include_test:
        test = enrich(group_queries(train, "test", dev_limit), "test")
        test_path, test_timing = evaluate_stage("test", test, run, index_path, signals_path, metadata_path,
                                                index_path.parent / "features.parquet", text_model, history_model,
                                                workers, signature)
    tables = [pl.read_parquet(dev_path)] + ([pl.read_parquet(test_path)] if test_path else [])
    frame = pl.concat(tables)
    summary = summary_table(frame)
    summary.write_csv(run / "metrics.csv")
    result = {"signature": signature, "data": sources, "code": code, "corpus": corpus,
              "config": config, "train_queries_requested": train_queries,
              "dev_queries": len(dev), "test_queries": len(test) if include_test else 0,
              "timing": {"dev": dev_timing, "test": test_timing},
              "answer_sha256": sha256_file(Path("answer.csv")) if Path("answer.csv").exists() else None,
              "metrics": summary.to_dicts()}
    write_json(run / "summary.json", result)
    write_json(output / "ranker" / "latest.json", {"run": str(run), "signature": signature})
    from .ranker_report import write_ranker_report
    write_ranker_report(run, Path("reports/ranker.md"), frame)
    return result
