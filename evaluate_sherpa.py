import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf
from jiwer import wer
from scipy.signal import resample_poly

from common import load_config, normalize_text, read_jsonl, resolve_path, write_json


TARGET_SAMPLE_RATE = 16000


def load_audio(path: Path, target_sr=TARGET_SAMPLE_RATE):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio


def find_file(directory: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file found in {directory} for patterns: {patterns}")


def evaluate_model(config_path: Path, onnx_dir: Path, output_path: Path | None = None):
    config, config_dir = load_config(config_path)
    manifest_path = resolve_path(config_dir, config["paths"]["eval_manifest"])
    num_threads = int(config.get("num_threads", 4))
    language = config.get("language", "en")
    task = config.get("task", "transcribe")

    encoder = find_file(onnx_dir, ["*encoder*.onnx", "encoder.onnx", "*.onnx"])
    decoder = find_file(onnx_dir, ["*decoder*.onnx", "decoder.onnx"])
    tokens = find_file(onnx_dir, ["*tokens*.txt", "tokens.txt"])

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(encoder),
        decoder=str(decoder),
        tokens=str(tokens),
        language=language,
        task=task,
        num_threads=num_threads,
    )

    refs = []
    preds = []
    samples = []
    manifest = read_jsonl(manifest_path)

    for item in manifest:
        stream = recognizer.create_stream()
        audio = load_audio(Path(item["audio_filepath"]))
        stream.accept_waveform(TARGET_SAMPLE_RATE, audio)
        recognizer.decode_stream(stream)
        pred = normalize_text(stream.result.text)
        ref = normalize_text(item["text"])
        refs.append(ref)
        preds.append(pred)
        samples.append({"id": item["id"], "ref": ref, "pred": pred})

    error = 100.0 * wer(refs, preds)
    result = {
        "onnx_dir": str(onnx_dir),
        "encoder": str(encoder),
        "decoder": str(decoder),
        "tokens": str(tokens),
        "num_items": len(manifest),
        "wer": error,
        "war": 100.0 - error,
        "samples": samples[:10],
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a sherpa-onnx Whisper export on the eval manifest.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_model(
        Path(args.config),
        Path(args.onnx_dir).resolve(),
        Path(args.output).resolve() if args.output else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
