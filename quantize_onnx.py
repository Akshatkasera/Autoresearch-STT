import argparse
import json
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from common import ensure_dir, write_json


def decoder_ffn_nodes(base_layers: int = 10, total_layers: int = 12) -> list[str]:
    nodes = []
    for i in range(base_layers):
        nodes.append(f"/layers.{i}/fc1/MatMul")
        nodes.append(f"/layers.{i}/fc2/MatMul")
    for i in range(base_layers, total_layers):
        nodes.append(f"/layers.{i}/fc1/MatMul")
    return nodes


def quantize_directory(input_dir: Path, output_dir: Path):
    ensure_dir(output_dir)
    quantized = []

    for src in sorted(input_dir.glob("*.onnx")):
        dst = output_dir / src.name
        if "encoder" in src.stem.lower():
            quantize_dynamic(
                model_input=str(src),
                model_output=str(dst),
                weight_type=QuantType.QInt8,
                per_channel=True,
                op_types_to_quantize=["MatMul", "Gemm"],
            )
        else:
            quantize_dynamic(
                model_input=str(src),
                model_output=str(dst),
                weight_type=QuantType.QInt8,
                per_channel=True,
                op_types_to_quantize=["MatMul", "Gemm"],
                nodes_to_quantize=decoder_ffn_nodes(),
            )
        quantized.append(str(dst))

    for src in sorted(input_dir.glob("*.txt")):
        shutil.copy2(src, output_dir / src.name)

    result = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "quantized_files": quantized,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Quantize Whisper ONNX artifacts to int8 with ONNX Runtime.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    result = quantize_directory(Path(args.input_dir).resolve(), Path(args.output_dir).resolve())
    if args.output_json:
        write_json(Path(args.output_json).resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
