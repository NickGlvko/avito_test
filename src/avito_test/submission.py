"""Strict CSV contract. IDs are never converted to numbers or normalized."""
# Validate the serialized CSV as well as the in-memory predictions.
from __future__ import annotations

import csv
import re
from pathlib import Path


def validate_answer(path: Path, query_ids: list[str], item_ids: list[str]) -> dict:
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Benchmark has duplicate query_id")
    valid_items, expected, seen = set(item_ids), set(query_ids), set()
    counts = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["query_id", "answer"]:
            raise ValueError("CSV header must be exactly query_id,answer")
        for row in reader:
            if set(row) != {"query_id", "answer"} or row["answer"] is None:
                raise ValueError("Malformed CSV row")
            qid, answer = row["query_id"], row["answer"]
            if len(qid) != 16 or qid not in expected or qid in seen:
                raise ValueError(f"Invalid, unexpected or duplicate query_id: {qid}")
            seen.add(qid)
            ids = answer.split(" ") if answer else []
            if len(ids) > 50 or len(ids) != len(set(ids)):
                raise ValueError(f"Too many or duplicate items: {qid}")
            if any(not re.fullmatch(r"[0-9a-f]{16}", x) or x not in valid_items for x in ids):
                raise ValueError(f"Invalid or unknown item_id / whitespace: {qid}")
            counts.append(len(ids))
    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} queries")
    return {"queries": len(seen), "min_candidates": min(counts, default=0),
            "max_candidates": max(counts, default=0)}


def write_answer(path: Path, query_ids: list[str], predictions: list[list[str]],
                 item_ids: list[str]) -> dict:
    if len(query_ids) != len(predictions):
        raise ValueError("Different numbers of queries and predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["query_id", "answer"])
        writer.writerows((qid, " ".join(ids)) for qid, ids in zip(query_ids, predictions, strict=True))
    return validate_answer(path, query_ids, item_ids)
