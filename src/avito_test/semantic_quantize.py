"""Create a local int8 copy of the pinned E5 ONNX model for CPU comparison."""
# Store quantized weights separately from the original E5 reference.
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from .data import sha256_file


SOURCE = Path("models/multilingual-e5-small-614241f6-onnx")
TARGET = Path("models/multilingual-e5-small-614241f6-onnx-int8")


def main(source: Path = SOURCE, target: Path = TARGET) -> dict:
    if target.exists() and (target / "onnx/model.onnx").exists():
        return json.loads((target / "quantization.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name in ("onnx", ".cache"):
            continue
        destination = target / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)
    (target / "onnx").mkdir(exist_ok=True)
    source_weights = source / "onnx/model.onnx"
    target_weights = target / "onnx/model.onnx"
    quantize_dynamic(str(source_weights), str(target_weights),
                     weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    report = {"source": str(source), "source_sha256": sha256_file(source_weights),
              "target": str(target), "target_sha256": sha256_file(target_weights),
              "method": "ONNX Runtime dynamic QInt8 for MatMul and Gemm"}
    (target / "quantization.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()
    main(args.source, args.target)
