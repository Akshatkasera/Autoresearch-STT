import argparse
import io
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset, load_from_disk
from scipy.signal import resample_poly

from common import ensure_dir, load_config, normalize_text, resolve_path


TARGET_SAMPLE_RATE = 16000
DATASET_NAME = "ai4bharat/Svarah"
TEST_SIZE = 0.1
EVAL_SIZE = 0.111
SPLIT_SEED = 42


def load_audio_from_bytes(audio_bytes, target_sr=TARGET_SAMPLE_RATE):
    audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio_array = resample_poly(audio_array, target_sr // g, sr // g).astype(np.float32)
    return audio_array, target_sr


def ensure_eval_raw_dataset(eval_raw_dir: Path):
    if eval_raw_dir.exists():
        return eval_raw_dir

    dataset = load_dataset(DATASET_NAME, split="test")
    train_val = dataset.train_test_split(test_size=TEST_SIZE, seed=SPLIT_SEED)["train"]
    eval_raw = train_val.train_test_split(test_size=EVAL_SIZE, seed=SPLIT_SEED)["test"]
    eval_raw.save_to_disk(str(eval_raw_dir))
    return eval_raw_dir


def build_eval_manifest(config_path: Path):
    config, config_dir = load_config(config_path)
    paths = config["paths"]
    eval_raw_dir = resolve_path(config_dir, paths["eval_raw_dir"])
    eval_audio_dir = ensure_dir(resolve_path(config_dir, paths["eval_audio_dir"]))
    eval_manifest = resolve_path(config_dir, paths["eval_manifest"])
    max_eval_items = int(config.get("max_eval_items", -1))

    ensure_eval_raw_dataset(eval_raw_dir)
    dataset = load_from_disk(str(eval_raw_dir))
    audio_col = "audio_filepath" if "audio_filepath" in dataset.column_names else "audio"
    dataset = dataset.cast_column(audio_col, Audio(decode=False))

    ensure_dir(eval_manifest.parent)
    count = 0
    with eval_manifest.open("w", encoding="utf-8") as manifest_f:
        for idx in range(len(dataset)):
            if max_eval_items > 0 and count >= max_eval_items:
                break
            row = dataset[idx]
            text = normalize_text((row.get("text") or "").strip())
            audio_bytes = row[audio_col]["bytes"]
            samples, sample_rate = load_audio_from_bytes(audio_bytes)
            wav_path = eval_audio_dir / f"{idx:06d}.wav"
            sf.write(wav_path, samples, sample_rate)
            record = {
                "id": idx,
                "audio_filepath": str(wav_path),
                "text": text,
            }
            manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return {"manifest_path": str(eval_manifest), "num_items": count}


def main():
    parser = argparse.ArgumentParser(description="Export eval audio and manifest from the raw HF dataset.")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    result = build_eval_manifest(Path(args.config))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
