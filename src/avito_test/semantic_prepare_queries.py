"""Cache E5 vectors for every provided query text before parallel evaluation."""
# Deduplicate query text before encoding, then restore its original mapping.
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .semantic_embeddings import OUT, encode_queries


def prepare(threads: int = 2) -> dict:
    data = Path("src/avito_test")
    train = pl.read_parquet(data / "train.parquet", columns=["search_query"])
    benchmark = pl.read_parquet(data / "benchmark_queries.parquet", columns=["search_query"])
    queries = pl.concat((train, benchmark))["search_query"].unique().to_list()
    vectors = encode_queries(queries, out=OUT, threads=threads)
    report = {"raw_unique_queries": len(queries), "encoded_unique_queries": len(vectors),
              "output": str(OUT / "query_vectors.parquet")}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    prepare()
