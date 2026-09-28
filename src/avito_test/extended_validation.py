"""Controlled corpus-scope experiments with the original query split and frozen parameters.

Both scopes use union-fitted, label-free text statistics. The augmented scope is
benchmark items plus ONLY the current query's missing positives. Scope masks are
applied before each source top-k and before history nearest-query selection.
"""
# Apply corpus membership before top-k so unavailable items cannot occupy retrieval slots.
from __future__ import annotations

import gc
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy import sparse
from threadpoolctl import threadpool_limits

from .data import SEARCH, SEED, digest, load_items, normalize, sha256_file, split_train, write_json
from .retrieval import DEPTH, EXPERIMENTS, INDEX_CONFIG, TextIndex, TrainSignals, fuse, topk

VARIANTS = {name: EXPERIMENTS[name] for name in
            ["bm25", "bm25_geo16", "rrf_global", "rrf_geo16", "rrf_stats", "rrf_history_half"]}
CURRENT = "rrf_history_half"
SCHEMES = ["union", "benchmark_plus_positives"]
SOURCE_NAMES = ["bm25", "char", "bm25_geo16", "char_geo16", "bm25_stats", "char_stats", "history"]


def union_items(data_dir: Path) -> tuple[pl.DataFrame, dict]:
    """Keep every item_id; benchmark snapshots take precedence on overlap."""
    benchmark = load_items(data_dir)
    raw = pl.read_parquet(data_dir / "train.parquet", columns=benchmark.columns)
    # The main audit checks exact snapshot equality. Check it again before selecting one.
    snapshots = raw.unique()
    if snapshots["item_id"].n_unique() != snapshots.height:
        raise ValueError("Conflicting item snapshots in train; choose a policy before evaluation")
    del raw
    shared = snapshots.join(benchmark, on="item_id", suffix="_benchmark")
    differences = shared.select(pl.any_horizontal([
        ~pl.col(c).eq_missing(pl.col(c + "_benchmark")) for c in benchmark.columns if c != "item_id"
    ]).sum()).item()
    extras = snapshots.join(benchmark.select("item_id"), on="item_id", how="anti")
    combined = pl.concat([benchmark, extras]).sort("item_id")
    if combined["item_id"].n_unique() != combined.height:
        raise ValueError("Union contains duplicate item_id")
    return combined, {"benchmark_items": benchmark.height, "train_items": snapshots.height,
                      "shared_items": shared.height, "extra_items": extras.height,
                      "union_items": combined.height, "conflicting_shared_snapshots": differences,
                      "overlap_policy": "benchmark snapshot takes precedence"}


def prepare_union(data_dir: Path, output: Path, sources: dict) -> tuple[Path, Path, dict]:
    signature = digest({"sources": sources, "index": INDEX_CONFIG,
                        "corpus_policy": "unique item_id; benchmark snapshot priority; v1",
                        "retrieval_code": sha256_file(Path(__file__).with_name("retrieval.py")),
                        "data_code": sha256_file(Path(__file__).with_name("data.py"))})
    cache = output / "cache" / signature[:16]
    index_path, metadata_path = cache / "union_index.joblib", cache / "items.parquet"
    manifest_path = cache / "corpus.json"
    if manifest_path.exists() and index_path.exists() and metadata_path.exists():
        return index_path, metadata_path, json.loads(manifest_path.read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    items, manifest = union_items(data_dir)
    print(f"Union corpus: {manifest}", flush=True)
    benchmark_ids = pl.read_parquet(data_dir / "benchmark_items.parquet", columns=["item_id"])["item_id"]
    items.select("item_id", "item_location_id", "item_category_id").with_columns(
        pl.col("item_id").is_in(benchmark_ids.implode()).alias("in_benchmark")).write_parquet(metadata_path)
    index = TextIndex(items)
    joblib.dump(index, index_path)
    manifest.update({"signature": signature, "build_seconds": time.perf_counter() - start,
                     "index_bytes": index_path.stat().st_size})
    write_json(manifest_path, manifest)
    return index_path, metadata_path, manifest


def rank_in_scope(scores: np.ndarray, allowed: np.ndarray | None, depth: int = DEPTH,
                  positive_only: bool = False) -> np.ndarray:
    """Allowed rows must be sorted, preserving the original item_id tie-break."""
    values = scores if allowed is None else scores[allowed]
    active = np.flatnonzero(values > 0)
    # Sparse retrieval usually matches only a fraction of the corpus. Partitioning
    # positive scores is exact when enough exist; keep zero-score tie filling otherwise.
    if positive_only or len(active) >= depth:
        local = active[topk(values[active], depth)]
    else:
        local = topk(values, depth)
    return (local if allowed is None else allowed[local]).astype(np.int32)


def augmented_scope(benchmark_rows: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    relevant = np.unique(relevant)
    positions = np.searchsorted(benchmark_rows, relevant)
    present = np.zeros(len(relevant), dtype=bool)
    in_bounds = positions < len(benchmark_rows)
    present[in_bounds] = benchmark_rows[positions[in_bounds]] == relevant[in_bounds]
    # Both arrays are sorted. Insert only the handful of missing rows; avoid
    # rebuilding a hash set over all 189k benchmark rows for each query.
    return np.insert(benchmark_rows, positions[~present], relevant[~present]).astype(np.int32)


class ScopedRetriever:
    def __init__(self, index: TextIndex, signals: TrainSignals, benchmark_mask: np.ndarray):
        self.index, self.signals = index, signals
        self.benchmark_mask = benchmark_mask
        self.benchmark_rows = np.flatnonzero(benchmark_mask).astype(np.int32)
        self.stats_weights = {}
        for category, probs in signals.category_prob.items():
            compatibility = np.zeros(len(signals.categories), dtype=np.float32)
            for item_category, probability in probs.items():
                compatibility[signals.categories == item_category] = probability
            self.stats_weights[category] = (.75 + .25 * compatibility) * (1 + .12 * signals.popularity)
        self.default_stats = .75 * (1 + .12 * signals.popularity)
        self.last_text = None
        if signals.history_vectorizer is not None:
            self.links_csc = signals.links.tocsc()
            self.benchmark_texts = np.asarray(signals.links[:, self.benchmark_rows].getnnz(axis=1)).ravel() > 0

    def history_scores(self, similarities: np.ndarray, relevant: np.ndarray, augmented: bool) -> np.ndarray:
        signals = self.signals
        if signals.history_vectorizer is None:
            return np.zeros(len(self.benchmark_mask), dtype=np.float32)
        available = None
        if augmented:
            active = self.benchmark_texts.copy()
            # Only training interactions of inserted items can activate history texts.
            # Their held-out query-item labels never enter links, popularity or vocabulary.
            for item in relevant[~self.benchmark_mask[relevant]]:
                start, end = self.links_csc.indptr[item:item + 2]
                active[self.links_csc.indices[start:end]] = True
            available = np.flatnonzero(active)
        nearest = rank_in_scope(similarities, available, 20, positive_only=True)
        nearest = nearest[similarities[nearest] >= .3]
        if not len(nearest):
            return np.zeros(len(self.benchmark_mask), dtype=np.float32)
        weights = sparse.csr_matrix((similarities[nearest] ** 3, ([0] * len(nearest), nearest)),
                                    shape=(1, len(similarities)))
        return (weights @ signals.links).toarray().ravel()

    def retrieve(self, query: dict, relevant: np.ndarray) -> dict:
        text = normalize(query["search_query"])
        if self.last_text != text:
            self.words, self.chars = self.index.score(query)
            if self.signals.history_vectorizer is not None:
                self.similarities = (self.signals.history_vectorizer.transform([text]) @
                                     self.signals.history_postings).toarray().ravel()
            else:
                self.similarities = np.empty(0, dtype=np.float32)
            self.last_text = text
        geography = np.where(self.signals.locations == query["search_location_id"], 16., 1.).astype(np.float32)
        stats = geography * self.stats_weights.get(query["search_category"], self.default_stats)
        scores = {"bm25": self.words, "char": self.chars,
                  "bm25_geo16": self.words * geography, "char_geo16": self.chars * geography,
                  "bm25_stats": self.words * stats, "char_stats": self.chars * stats}
        result = {}
        for scheme in SCHEMES:
            augmented = scheme == "benchmark_plus_positives"
            allowed = augmented_scope(self.benchmark_rows, relevant) if augmented else None
            ranked = {name: rank_in_scope(values, allowed) for name, values in scores.items()}
            history = self.history_scores(self.similarities, relevant, augmented) * stats
            ranked["history"] = rank_in_scope(history, allowed, positive_only=True)
            result[scheme] = ranked
        return result


def measure(prediction: np.ndarray, results: dict, weights: dict, relevant: np.ndarray,
            benchmark_mask: np.ndarray) -> dict:
    truth = set(map(int, relevant))
    candidate_union = {int(x) for source in weights for x in results[source]}
    predicted50 = set(map(int, prediction[:50]))
    found, candidate_found = len(truth & predicted50), len(truth & candidate_union)
    outside = {i for i in truth if not benchmark_mask[i]}
    inside = truth - outside
    metrics = {"positives": len(truth), "retrieved": found, "candidate_found": candidate_found,
               "lost_at_selection": candidate_found - found, "missed_by_sources": len(truth) - candidate_found,
               "unavoidable_candidate_cap_loss": max(0, candidate_found - 50),
               "avoidable_selection_loss": min(50, candidate_found) - found,
               "candidate_recall": candidate_found / len(truth), "hit@50": float(found > 0),
               "candidate_oracle@50": min(50, candidate_found) / len(truth),
               "ceiling@50": min(1., 50 / len(truth)), "candidate_count": len(candidate_union)}
    metrics.update({f"recall@{k}": len(set(map(int, prediction[:k])) & truth) / len(truth)
                    for k in [50, 100, 200, 500]})
    for name, subset in [("outside", outside), ("inside", inside)]:
        hits = len(subset & predicted50)
        candidates = len(subset & candidate_union)
        metrics.update({f"{name}_positives": len(subset), f"{name}_retrieved": hits,
                        f"{name}_candidate_found": candidates,
                        f"{name}_recall@50": hits / len(subset) if subset else None,
                        f"{name}_candidate_recall": candidates / len(subset) if subset else None})
    return metrics


_WORKER = None


def initialize_worker(index_path: str, signals_path: str, metadata_path: str) -> None:
    global _WORKER
    threadpool_limits(limits=1)
    index = joblib.load(index_path, mmap_mode="r")
    signals = joblib.load(signals_path, mmap_mode="r")
    metadata = pl.read_parquet(metadata_path)
    if metadata["item_id"].to_list() != index.item_ids:
        raise ValueError("Index metadata is misaligned")
    _WORKER = ScopedRetriever(index, signals, metadata["in_benchmark"].to_numpy())


def evaluate_chunk(number: int, queries: list[dict], destination: str) -> tuple[int, int]:
    retriever = _WORKER
    rows, positive_rows = [], []
    for query in queries:
        relevant = np.array(query["relevant_rows"], dtype=np.int32)
        schemes = retriever.retrieve(query, relevant)
        for scheme, results in schemes.items():
            predictions = {}
            for name, weights in VARIANTS.items():
                prediction = fuse(results, weights)
                predictions[name] = prediction
                rows.append({"group_id": query["group_id"], "text": query["text"],
                             "stage": query["stage"], "scheme": scheme, "experiment": name,
                             "all_in_benchmark": bool(retriever.benchmark_mask[relevant].all()),
                             "known_text": query["text"] in retriever.signals.seen_texts,
                             "all_items_seen": all(retriever.index.item_ids[i] in retriever.signals.seen_items for i in relevant),
                             "has_filters": bool(query["search_infm_params_text"]),
                             "corpus_size": (len(retriever.benchmark_rows) + int((~retriever.benchmark_mask[relevant]).sum())
                                             if scheme == "benchmark_plus_positives" else len(retriever.benchmark_mask)),
                             **measure(prediction, results, weights, relevant, retriever.benchmark_mask)})
            fused_ranks = {int(item): rank for rank, item in enumerate(predictions[CURRENT], 1)}
            positions = {name: {int(item): rank for rank, item in enumerate(values, 1)}
                         for name, values in results.items()}
            for positive in relevant:
                final_rank = fused_ranks.get(int(positive))
                in_candidates = any(int(positive) in positions[s] for s in VARIANTS[CURRENT])
                positive_rows.append({"group_id": query["group_id"], "stage": query["stage"], "scheme": scheme,
                                      "item_id": retriever.index.item_ids[positive],
                                      "in_benchmark": bool(retriever.benchmark_mask[positive]),
                                      "seen_item": retriever.index.item_ids[positive] in retriever.signals.seen_items,
                                      "final_rank": final_rank,
                                      "failure_stage": ("retrieved" if final_rank is not None and final_rank <= 50 else
                                                        "lost_at_selection" if in_candidates else "missed_by_sources"),
                                      **{s: positions[s].get(int(positive)) for s in SOURCE_NAMES}})
    path = Path(destination)
    nullable_metrics = {f"{part}_{metric}": pl.Float64 for part in ["inside", "outside"]
                        for metric in ["recall@50", "candidate_recall"]}
    pl.DataFrame(rows, infer_schema_length=None, schema_overrides=nullable_metrics).write_parquet(path / f"metrics_{number:05d}.parquet")
    pl.DataFrame(positive_rows, infer_schema_length=None,
                 schema_overrides={name: pl.Int64 for name in ["final_rank", *SOURCE_NAMES]}).write_parquet(path / f"positives_{number:05d}.parquet")
    # A completion marker is written only after both files have been closed.
    write_json(path / f"done_{number:05d}.json", {"queries": len(queries)})
    return number, len(queries)


def run_extended(data_dir: Path, output: Path, train: pl.DataFrame, sources: dict,
                 stages: list[str], workers: int = 4, limit: int = 0) -> None:
    if workers < 1 or limit < 0:
        raise ValueError("workers must be positive and limit must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    index_path, metadata_path, corpus = prepare_union(data_dir, output, sources)
    items = pl.read_parquet(metadata_path)
    ids = items["item_id"].to_list()
    mapping = {item: i for i, item in enumerate(ids)}
    training = split_train(train, "group")
    signals_key = digest({"corpus": corpus["signature"], "protocol": "group", "seed": SEED,
                          "training_rows": training.height,
                          "pairs": digest(training.select("group_id", "item_id").rows())})[:16]
    signals_path = index_path.parent / f"signals_{signals_key}.joblib"
    if not signals_path.exists():
        signals = TrainSignals(training, items)
        joblib.dump(signals, signals_path)
        del signals
        gc.collect()
    held = train.filter(pl.col("split").is_in([8 if s == "dev" else 9 for s in stages]))
    groups = held.group_by("group_id", *SEARCH).agg(
        pl.col("item_id").unique().sort().alias("relevant"), pl.col("text").first(),
        pl.col("split").first()).sort("group_id")
    all_count = groups.height
    if limit:
        groups = groups.head(limit)
    groups = groups.with_columns(pl.when(pl.col("split") == 8).then(pl.lit("dev")).otherwise(pl.lit("test")).alias("stage"))
    queries = []
    for query in groups.sort("text", "group_id").iter_rows(named=True):
        query["relevant_rows"] = [mapping[item] for item in query["relevant"]]
        queries.append(query)
    code = {name: sha256_file(Path(__file__).with_name(name))
            for name in ["extended_validation.py", "retrieval.py", "data.py"]}
    run_signature = digest({"sources": sources, "code": code, "stages": stages, "limit": limit,
                            "variants": VARIANTS, "index_config": INDEX_CONFIG, "seed": SEED})
    run_dir = output / ("_".join(stages) + (f"_smoke{limit}" if limit else ""))
    parts = run_dir / "parts" / run_signature[:16]
    parts.mkdir(parents=True, exist_ok=True)
    groups.write_parquet(run_dir / "queries.parquet")
    manifest = {"signature": run_signature, "corpus": corpus, "sources": sources, "code": code,
                "index_config": INDEX_CONFIG, "variants": VARIANTS, "seed": SEED,
                "stages": stages, "limit": limit, "available_groups": all_count, "evaluated_groups": groups.height,
                "evaluated_positives": groups["relevant"].list.len().sum(),
                "excluded_missing_positives": 0, "training_rows": training.height, "workers": workers,
                "text_statistics": "fixed union vocabulary, IDF and BM25 lengths; same in both scopes",
                "history_statistics": "TF-IDF fitted on train-only history texts linked to union items; fixed across scopes",
                "history_scope": "eligible history texts restricted before nearest-20, items before top-500",
                "augmented_scope": "benchmark union current query positives only; query-specific oracle diagnostic",
                "parts": str(parts), "complete": False}
    write_json(run_dir / "manifest.json", manifest)
    chunks = [queries[i:i + 128] for i in range(0, len(queries), 128)]
    completed = 0
    pending = []
    for number, chunk in enumerate(chunks):
        marker = parts / f"done_{number:05d}.json"
        if marker.exists() and (parts / f"metrics_{number:05d}.parquet").exists() and (parts / f"positives_{number:05d}.parquet").exists():
            completed += len(chunk)
        else:
            pending.append((number, chunk))
    start = time.perf_counter()
    print(f"Expanded validation: {len(queries)} groups, resumed {completed}, {workers} workers", flush=True)
    # Each worker mmaps immutable sparse arrays; it never loads raw descriptions.
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                             initargs=(str(index_path), str(signals_path), str(metadata_path))) as pool:
        futures = [pool.submit(evaluate_chunk, n, chunk, str(parts)) for n, chunk in pending]
        for future in as_completed(futures):
            _, count = future.result()
            completed += count
            print(f"Expanded: {completed}/{len(queries)} groups, {time.perf_counter()-start:.1f}s", flush=True)
    metrics = pl.read_parquet(str(parts / "metrics_*.parquet")).sort("stage", "group_id", "scheme", "experiment")
    expected = groups.height * len(SCHEMES) * len(VARIANTS)
    if metrics.height != expected or metrics.unique(["group_id", "scheme", "experiment"]).height != expected:
        raise ValueError("Missing or duplicate query metrics")
    metrics.write_parquet(run_dir / "per_query.parquet")
    positives = pl.read_parquet(str(parts / "positives_*.parquet")).sort("stage", "group_id", "scheme", "item_id")
    positives.write_parquet(run_dir / "positive_ranks.parquet")
    manifest.update({"complete": True, "evaluation_seconds_this_invocation": time.perf_counter() - start,
                     "resumed_groups": len(queries) - sum(len(chunk) for _, chunk in pending)})
    write_json(run_dir / "manifest.json", manifest)
    from .extended_report import write_extended_report
    write_extended_report(run_dir, Path("reports/extended_validation.md") if not limit else run_dir / "report.md")
