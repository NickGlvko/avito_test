"""Deterministic dev examples of gains and losses from the E5 experiment."""
# Choose diagnostic examples deterministically from saved dev results.
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


ITEMS = Path("artifacts/expanded/cache/73915bb3bc082213/features.parquet")
OUT = Path("artifacts/semantic/error_examples.csv")


def examples(dev: Path, output: Path = OUT) -> pl.DataFrame:
    positives = pl.read_parquet(dev / "positive_ranks.parquet").filter(
        pl.col("scheme") == "union")
    items = (pl.read_parquet(ITEMS, columns=["item_id", "item_title_raw"])
             .with_row_index("item_row")
             .with_columns(pl.col("item_row").cast(pl.Int64)))
    frame = positives.join(items, on="item_row", validate="m:1")
    conditions = {
        "B rescues old candidate": ((pl.col("A_rank") == 0) | (pl.col("A_rank") > 50))
            & pl.col("B_rank").is_between(1, 50),
        "D finds semantic-only positive": (~pl.col("in_old_pool"))
            & pl.col("D_rank").is_between(1, 50),
        "D loses A positive": pl.col("A_rank").is_between(1, 50)
            & ((pl.col("D_rank") == 0) | (pl.col("D_rank") > 50)),
        "Semantic retrieval found but D missed": (~pl.col("in_old_pool"))
            & ((pl.col("semantic_global_rank") > 0) | (pl.col("semantic_local_rank") > 0))
            & (pl.col("D_rank") > 50),
    }
    rows = []
    for label, condition in conditions.items():
        part = frame.filter(condition).sort("group_id", "item_row").head(8)
        rows.append(part.with_columns(pl.lit(label).alias("case")))
    result = pl.concat(rows).select("case", "group_id", "text", "item_id",
        "item_title_raw", "same_location", "A_rank", "B_rank", "C_rank", "D_rank",
        "semantic_global_rank", "semantic_local_rank", "semantic_cosine")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(output)
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dev", type=Path)
    args = parser.parse_args()
    examples(args.dev)
