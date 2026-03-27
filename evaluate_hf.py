import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from jiwer import wer
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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


def evaluate_model(config_path: Path, model_path: Path, output_path: Path | None = None):
    config, config_dir = load_config(config_path)
    manifest_path = resolve_path(config_dir, config["paths"]["eval_manifest"])
    device_name = config.get("device", "cuda")
    language = config.get("language", "en")
    task = config.get("task", "transcribe")

    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    processor = WhisperProcessor.from_pretrained(str(model_path))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model_dtype = next(model.parameters()).dtype

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    manifest = read_jsonl(manifest_path)

    refs = []
    preds = []
    samples = []
    for item in manifest:
        audio = load_audio(Path(item["audio_filepath"]))
        features = processor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt").input_features.to(
            device=device,
            dtype=model_dtype,
        )
        with torch.no_grad():
            generated = model.generate(
                input_features=features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )
        pred = normalize_text(processor.batch_decode(generated, skip_special_tokens=True)[0])
        ref = normalize_text(item["text"])
        refs.append(ref)
        preds.append(pred)
        samples.append({"id": item["id"], "ref": ref, "pred": pred})

    error = 100.0 * wer(refs, preds)
    result = {
        "model_path": str(model_path),
        "num_items": len(manifest),
        "wer": error,
        "war": 100.0 - error,
        "samples": samples[:10],
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a merged Whisper HF model on the eval manifest.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_model(
        Path(args.config),
        Path(args.model_path).resolve(),
        Path(args.output).resolve() if args.output else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
