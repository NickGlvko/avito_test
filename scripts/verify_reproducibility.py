"""Recompute the frozen answer with network connections blocked and compare bytes."""
# Block network access during the repeat run to test local reproducibility.
import argparse
import json
import socket
from pathlib import Path

from avito_test.cli import predict, prepare
from avito_test.data import load_items, sha256_file, write_json
from avito_test.retrieval import get_index


def blocked(*args, **kwargs):
    raise RuntimeError("Network access is forbidden during offline reproduction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("src/avito_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--answer", type=Path, default=Path("answer.csv"))
    args = parser.parse_args()
    original_sha = sha256_file(args.answer)
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    config = json.loads(args.config.read_text(encoding="utf-8"))
    items = load_items(args.data_dir)
    train = prepare(args.data_dir, args.output_dir)
    index = get_index(items, args.data_dir, args.output_dir)
    repeated = args.output_dir / "answer_repeat.csv"
    manifest = predict(index, items, train, args.data_dir, args.output_dir, config, repeated)
    report = {"original_sha256": original_sha, "repeated_sha256": manifest["answer_sha256"],
              "identical": original_sha == manifest["answer_sha256"],
              "python_socket_connections": "blocked"}
    write_json(args.output_dir / "reproducibility.json", report)
    if not report["identical"]:
        raise RuntimeError("Reproduced CSV differs from original")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
