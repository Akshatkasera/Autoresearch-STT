import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import ensure_dir, load_config, resolve_path


def merge_lora(config_path: Path):
    config, config_dir = load_config(config_path)
    paths = config["paths"]
    base_model_name = config["base_model_name"]
    adapter_dir = resolve_path(config_dir, paths["lora_adapter_dir"])
    merged_model_dir = ensure_dir(resolve_path(config_dir, paths["merged_model_dir"]))
    device = config.get("device", "cuda")

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    processor = WhisperProcessor.from_pretrained(base_model_name)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if use_cuda:
        base_model = base_model.to("cuda")

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_model_dir), safe_serialization=True)
    processor.save_pretrained(str(merged_model_dir))

    return {
        "merged_model_dir": str(merged_model_dir),
        "adapter_dir": str(adapter_dir),
        "base_model_name": base_model_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Merge a Whisper LoRA adapter into the base model.")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    result = merge_lora(Path(args.config))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
