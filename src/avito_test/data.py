"""Shared data contract, deterministic splits and provenance."""
# Keep query_id and item_id as strings so leading zeros and letter case survive.
from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

import polars as pl

SEARCH = ["search_query", "search_location_id", "search_is_delivery_search",
          "search_infm_params_text", "search_category"]
FILES = ["train.parquet", "benchmark_queries.parquet", "benchmark_items.parquet"]
SEED = 20260926


def normalize(value: str | None) -> str:
    value = html.unescape(value or "").lower().replace("ё", "е")
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     default=str).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def source_manifest(data_dir: Path) -> dict:
    return {name: {"sha256": sha256_file(data_dir / name),
                   "bytes": (data_dir / name).stat().st_size} for name in FILES}


def bucket(value: object) -> int:
    return int(digest([SEED, value])[:16], 16) % 10


def enrich_train(train: pl.DataFrame) -> pl.DataFrame:
    """Group exact search features; normalized variants stay in the same split."""
    keys = []
    for row in train.select(SEARCH).unique(maintain_order=True).iter_rows(named=True):
        raw = [row[c] for c in SEARCH]
        normalized = raw.copy()
        normalized[0] = normalize(raw[0])
        normalized[3] = normalize(raw[3])
        keys.append({**row, "group_id": digest(raw), "text": normalized[0],
                     "split": bucket(normalized), "text_split": bucket(normalized[0])})
    return train.join(pl.DataFrame(keys), on=SEARCH, how="left", validate="m:1")


def split_train(train: pl.DataFrame, protocol: str) -> pl.DataFrame:
    column = "text_split" if protocol == "unseen" else "split"
    return train.filter(pl.col(column) < 8)


def evaluation_groups(train: pl.DataFrame, item_ids: list[str], protocol: str,
                      stage: str, limit: int = 0) -> tuple[pl.DataFrame, dict]:
    column = "text_split" if protocol == "unseen" else "split"
    held = train.filter(pl.col(column) == (8 if stage == "dev" else 9))
    groups = held.group_by("group_id", *SEARCH).agg(
        pl.col("item_id").unique().sort().alias("relevant"), pl.col("text").first())
    eligible = groups.filter(pl.col("relevant").list.eval(
        pl.element().is_in(pl.Series(item_ids).implode())).list.all()).sort("group_id")
    report = {"protocol": protocol, "stage": stage, "total_groups": groups.height,
              "eligible_groups": eligible.height,
              "excluded_missing_positives": groups.height - eligible.height,
              "total_positives": groups["relevant"].list.len().sum(),
              "eligible_positives": eligible["relevant"].list.len().sum()}
    # Sorting SHA256 keys gives a reproducible, content-independent sample.
    if limit:
        eligible = eligible.head(limit)
    report["evaluated_groups"] = eligible.height
    report["evaluated_positives"] = eligible["relevant"].list.len().sum()
    if not eligible.height:
        raise ValueError("No query groups have all positives in the chosen corpus")
    return eligible, report


def load_items(data_dir: Path) -> pl.DataFrame:
    items = pl.read_parquet(data_dir / "benchmark_items.parquet").sort("item_id")
    if items["item_id"].null_count() or items["item_id"].n_unique() != items.height:
        raise ValueError("Corpus item_id must be non-null and unique")
    if not items["item_id"].str.contains(r"^[0-9a-f]{16}$").all():
        raise ValueError("Corpus item_id has an invalid format")
    return items
