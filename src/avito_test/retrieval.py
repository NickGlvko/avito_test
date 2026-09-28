"""CPU sparse retrieval. Corpus text is public; click statistics require train-only rows."""
# Text may be indexed globally, while click statistics use only allowed training rows.
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .data import digest, normalize, sha256_file, write_json

INDEX_CONFIG = {"version": 1, "k1": 1.2, "b": .75, "min_df": 2,
                "word_max_features": 250000, "char_max_features": 250000,
                "char_ngrams": [3, 5], "field_weights": [3., .6, 1.],
                "description_limit": None, "char_fields": "title"}
DEPTH = 500


def topk(scores: np.ndarray, k: int, positive_only: bool = False) -> np.ndarray:
    """Stable tie break by corpus row (= lexicographically sorted original item_id)."""
    scores = np.asarray(scores).reshape(-1)
    candidates = np.flatnonzero(scores > 0) if positive_only else np.arange(len(scores))
    k = min(k, len(candidates))
    if k == 0:
        return np.empty(0, dtype=np.int32)
    if len(candidates) > k:
        values = scores[candidates]
        threshold = np.partition(values, len(values) - k)[len(values) - k]
        above = candidates[values > threshold]
        tied = candidates[values == threshold][:k - len(above)]
        candidates = np.concatenate([above, tied])
    return candidates[np.lexsort((candidates, -scores[candidates]))].astype(np.int32)


class BM25Field:
    def __init__(self, documents: list[str]):
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False,
                                          min_df=INDEX_CONFIG["min_df"],
                                          max_features=INDEX_CONFIG["word_max_features"], dtype=np.float32)
        counts = self.vectorizer.fit_transform(documents).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log1p((counts.shape[0] - df + .5) / (df + .5)).astype(np.float32)
        k1, b = INDEX_CONFIG["k1"], INDEX_CONFIG["b"]
        norm = k1 * (1 - b + b * lengths / max(float(lengths.mean()), 1e-8))
        counts.data = counts.data * (k1 + 1) / (counts.data + np.repeat(norm, np.diff(counts.indptr)))
        counts.data *= idf[counts.indices]
        self.postings = counts.T.tocsr()

    def score(self, query: str) -> np.ndarray:
        vector = self.vectorizer.transform([query])
        # Repeated query tokens do not give an accidental additional vote.
        vector.data[:] = 1
        return (vector @ self.postings).toarray().ravel()


class TextIndex:
    def __init__(self, items: pl.DataFrame):
        self.item_ids = items["item_id"].to_list()
        self.fields = []
        for field in ["item_title_raw", "item_infm_params_text", "item_description_raw"]:
            start = time.perf_counter()
            docs = [normalize(x) for x in items[field]]
            self.fields.append(BM25Field(docs))
            print(f"Indexed BM25 {field} in {time.perf_counter() - start:.1f}s", flush=True)
        del docs
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                               min_df=2, max_features=INDEX_CONFIG["char_max_features"],
                                               lowercase=False, sublinear_tf=True, dtype=np.float32)
        self.char_postings = self.char_vectorizer.fit_transform(
            [normalize(x) for x in items["item_title_raw"]]).T.tocsr()

    def score(self, query: dict) -> tuple[np.ndarray, np.ndarray]:
        text = normalize(query["search_query"])
        words = sum(w * field.score(text) for w, field in zip(INDEX_CONFIG["field_weights"], self.fields))
        chars = (self.char_vectorizer.transform([text]) @ self.char_postings).toarray().ravel()
        return words, chars


def get_index(items: pl.DataFrame, data_dir: Path, output: Path) -> TextIndex:
    signature = digest({"config": INDEX_CONFIG,
                        "corpus": sha256_file(data_dir / "benchmark_items.parquet"),
                        "code": sha256_file(Path(__file__)),
                        "normalization_code": sha256_file(Path(__file__).with_name("data.py"))})
    path = output / f"index_{signature[:16]}.joblib"
    if path.exists():
        print(f"Loading cached text index {path.name}", flush=True)
        index = joblib.load(path)
        if index.item_ids != items["item_id"].to_list():
            raise ValueError("Index item mapping does not match corpus")
        return index
    start = time.perf_counter()
    index = TextIndex(items)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, path)
    write_json(output / "index.json", {"signature": signature, "config": INDEX_CONFIG,
                                       "build_seconds": time.perf_counter() - start,
                                       "cache_bytes": path.stat().st_size})
    return index


class TrainSignals:
    """Only passed-in rows may contribute supervised signals (including query vocabulary)."""
    def __init__(self, train: pl.DataFrame, items: pl.DataFrame):
        self.locations = items["item_location_id"].to_numpy()
        self.categories = items["item_category_id"].to_numpy()
        ids = items["item_id"].to_list()
        mapping = {x: i for i, x in enumerate(ids)}
        self.seen_texts = set(train["text"])
        self.seen_items = set(train["item_id"])
        counts = Counter(train["item_id"])
        self.popularity = np.log1p(np.array([counts[x] for x in ids], dtype=np.float32))
        # Compatibility is learned from positives; never assume numeric category IDs are aligned.
        self.category_prob = {}
        for cat in train["search_category"].unique():
            cat_counts = Counter(train.filter(pl.col("search_category") == cat)["item_category_id"])
            total = sum(cat_counts.values())
            self.category_prob[cat] = {c: (n + 1) / (total + len(cat_counts)) for c, n in cat_counts.items()}
        available = train.filter(pl.col("item_id").is_in(items["item_id"].implode()))
        self.history_vectorizer = None
        self.history_texts = []
        if not available.height:
            return
        self.history_texts = sorted(set(available["text"]))
        text_index = {t: i for i, t in enumerate(self.history_texts)}
        pairs = available.group_by("text", "item_id").len()
        rows = [text_index[t] for t in pairs["text"]]
        cols = [mapping[x] for x in pairs["item_id"]]
        values = np.log1p(pairs["len"].to_numpy()).astype(np.float32)
        self.links = sparse.csr_matrix((values, (rows, cols)), shape=(len(text_index), len(ids)))
        self.history_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                                  min_df=1, max_features=150000, dtype=np.float32,
                                                  lowercase=False, sublinear_tf=True)
        try:
            self.history_postings = self.history_vectorizer.fit_transform(self.history_texts).T.tocsr()
        except ValueError as error:
            if "empty vocabulary" not in str(error):
                raise
            self.history_vectorizer = None

    def context(self, query: dict, location_factor: float, use_stats: bool = False) -> np.ndarray:
        weights = np.where(self.locations == query["search_location_id"], location_factor, 1.).astype(np.float32)
        if use_stats:
            probs = self.category_prob.get(query["search_category"], {})
            compatibility = np.array([probs.get(c, 0.) for c in self.categories], dtype=np.float32)
            weights *= (.75 + .25 * compatibility) * (1 + .12 * self.popularity)
        return weights

    def history(self, query: dict) -> np.ndarray:
        if self.history_vectorizer is None:
            return np.zeros(len(self.locations), dtype=np.float32)
        similarities = (self.history_vectorizer.transform([normalize(query["search_query"])]) @
                        self.history_postings).toarray().ravel()
        nearest = topk(similarities, 20, positive_only=True)
        nearest = nearest[similarities[nearest] >= .3]
        if not len(nearest):
            return np.zeros(len(self.locations), dtype=np.float32)
        weights = sparse.csr_matrix((similarities[nearest] ** 3, ([0] * len(nearest), nearest)),
                                    shape=(1, len(similarities)))
        return (weights @ self.links).toarray().ravel()


# All experiments are specified before the final local holdout is opened.
EXPERIMENTS = {
    "bm25": {"bm25": 1.}, "char": {"char": 1.},
    "rrf_global": {"bm25": 1., "char": 1.},
    "bm25_geo4": {"bm25_geo4": 1.}, "char_geo4": {"char_geo4": 1.},
    "rrf_geo4": {"bm25_geo4": 1., "char_geo4": 1.},
    "bm25_geo16": {"bm25_geo16": 1.}, "char_geo16": {"char_geo16": 1.},
    "rrf_geo16": {"bm25_geo16": 1., "char_geo16": 1.},
    "rrf_stats": {"bm25_stats": 1., "char_stats": 1.},
    "rrf_history_half": {"bm25_stats": 1., "char_stats": 1., "history": .5},
    "rrf_history": {"bm25_stats": 1., "char_stats": 1., "history": 1.},
}


def channels(index: TextIndex, signals: TrainSignals, query: dict, depth: int = DEPTH) -> dict:
    words, chars = index.score(query)
    results = {"bm25": topk(words, depth), "char": topk(chars, depth)}
    for factor in [4, 16]:
        weights = signals.context(query, factor)
        results[f"bm25_geo{factor}"] = topk(words * weights, depth)
        results[f"char_geo{factor}"] = topk(chars * weights, depth)
    weights = signals.context(query, 16, use_stats=True)
    results["bm25_stats"] = topk(words * weights, depth)
    results["char_stats"] = topk(chars * weights, depth)
    results["history"] = topk(signals.history(query) * weights, depth, positive_only=True)
    return results


def fuse(results: dict, weights: dict, depth: int = DEPTH, rrf_k: int = 60) -> np.ndarray:
    if len(weights) == 1:
        return results[next(iter(weights))][:depth]
    scores: dict[int, float] = {}
    for source, weight in weights.items():
        for rank, item in enumerate(results[source], start=1):
            key = int(item)
            scores[key] = scores.get(key, 0.) + weight / (rrf_k + rank)
    return np.array(sorted(scores, key=lambda i: (-scores[i], i))[:depth], dtype=np.int32)
