"""Dataset audit; no model or hyperparameter selection."""
# Inspect raw fields before choosing features; audit output must never modify labels.
from __future__ import annotations

from pathlib import Path
import polars as pl

from .data import SEARCH, FILES, enrich_train, normalize, sha256_file, source_manifest, write_json


def preparation_signature() -> dict:
    return {name: sha256_file(Path(__file__).with_name(name)) for name in ["audit.py", "data.py"]}


def describe(frame: pl.DataFrame) -> dict:
    return {"rows": frame.height, "schema": {k: str(v) for k, v in frame.schema.items()},
            "nulls": frame.null_count().row(0, named=True),
            "empty_strings": {c: frame[c].eq("").sum() for c, t in frame.schema.items()
                              if t == pl.String},
            "text_lengths": {c: {"median": frame[c].str.len_chars().median(),
                                 "p95": frame[c].str.len_chars().quantile(.95),
                                 "max": frame[c].str.len_chars().max()}
                             for c in frame.columns if c.endswith(("_raw", "_text"))}}


def audit(data_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frames = {name: pl.read_parquet(data_dir / name) for name in FILES}
    train, queries, items = (frames[name] for name in FILES)
    item_columns = items.columns
    report = {"sources": source_manifest(data_dir),
              "files": {name: describe(frame) for name, frame in frames.items()}}
    light = enrich_train(train.select(*SEARCH, "item_id", "item_category_id"))
    groups = light.group_by("group_id").agg(pl.col("item_id").unique().alias("relevant"))
    eligible = groups.filter(pl.col("relevant").list.eval(
        pl.element().is_in(items["item_id"].implode())).list.all())
    lengths = groups["relevant"].list.len()
    report["groups"] = {"count": groups.height, "raw_texts": train["search_query"].n_unique(),
                         "normalized_texts": light["text"].n_unique(),
                         "duplicate_pairs": train.height - light.unique(["group_id", "item_id"]).height,
                         "positives_median": lengths.median(), "positives_p95": lengths.quantile(.95),
                         "positives_max": lengths.max(), "groups_over_50": (lengths > 50).sum(),
                         "recall50_ceiling": lengths.cast(pl.Float64).map_elements(
                             lambda n: min(1., 50 / n), return_dtype=pl.Float64).mean()}
    train_ids = train.select("item_id").unique()
    report["overlap"] = {"train_unique_items": train_ids.height,
                         "shared_items": train_ids.join(items.select("item_id"), on="item_id").height,
                         "train_rows_in_corpus": train.join(items.select("item_id"), on="item_id").height,
                         "eligible_groups": eligible.height}
    for cols, name in [(["search_query"], "raw_text"),
                       (["search_query", "search_location_id"], "raw_text_location"),
                       (SEARCH, "full_context")]:
        report["overlap"]["benchmark_" + name] = queries.join(
            train.select(cols).unique(), on=cols).height
    normalized_train = set(light["text"])
    report["overlap"]["benchmark_normalized_text"] = sum(
        normalize(q) in normalized_train for q in queries["search_query"])
    report["positive_context"] = {
        "equal_location_rate": train.select((pl.col("search_location_id") == pl.col("item_location_id")).mean()).item(),
        "equal_category_rate": train.select((pl.col("search_category") == pl.col("item_category_id")).mean()).item(),
        "category_pairs": train.group_by("search_category", "item_category_id").len().sort("len", descending=True).to_dicts(),
        "location_by_delivery": train.group_by("search_is_delivery_search").agg(
            (pl.col("search_location_id") == pl.col("item_location_id")).mean().alias("match_rate"),
            pl.len()).to_dicts()}
    # Exact row deduplication detects changing snapshots without selecting a train snapshot for retrieval.
    unique_snapshots = train.select(item_columns).unique()
    changing = unique_snapshots.group_by("item_id").len().filter(pl.col("len") > 1)
    shared = train.select(item_columns).join(items, on="item_id", suffix="_benchmark")
    differences = {c: shared.select((~pl.col(c).eq_missing(pl.col(c + "_benchmark"))).sum()).item()
                   for c in item_columns if c != "item_id"}
    report["identity"] = {"duplicate_corpus_ids": items.height - items["item_id"].n_unique(),
                          "duplicate_query_ids": queries.height - queries["query_id"].n_unique(),
                          "changing_train_items": changing.height,
                          "shared_row_feature_differences": differences}
    report["validation_composition"] = {}
    for name, subset in [("all", light), ("eligible", light.join(eligible.select("group_id"), on="group_id"))]:
        distinct = subset.unique("group_id")
        report["validation_composition"][name] = {
            "groups": distinct.height,
            "with_filters_rate": distinct["search_infm_params_text"].ne("").mean(),
            "categories": distinct.group_by("search_category").len().sort("len", descending=True).to_dicts(),
            "top_locations": distinct.group_by("search_location_id").len().sort("len", descending=True).head(15).to_dicts()}
    light.write_parquet(output / "train_pairs.parquet")
    write_json(output / "audit.json", report)
    write_json(output / "sources.json", report["sources"])
    write_json(output / "preparation.json", preparation_signature())
    print(f"Audit: {groups.height} groups, {eligible.height} eligible; saved to {output}", flush=True)
