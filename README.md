# autoresearch-stt

This directory turns your existing Whisper fine-tuning project into an `autoresearch`-style workspace for Android + Sherpa deployment.

It is built for your actual target:

- train or update a Whisper model
- merge the LoRA adapter
- export to Sherpa-compatible ONNX
- quantize to int8
- evaluate the deployed model
- keep or discard changes based on deployed accuracy

The original GPT demo in the repo does not solve this problem. This workspace does.

## What is included

- `config.example.json`: paths and runtime settings
- `prepare_eval_audio.py`: exports eval audio to wav + manifest from your raw HF dataset
- `merge_lora.py`: merges the LoRA adapter into the base Whisper model
- `evaluate_hf.py`: evaluates the merged HF model on the eval manifest
- `evaluate_sherpa.py`: evaluates exported Sherpa Whisper ONNX artifacts
- `quantize_onnx.py`: dynamic int8 quantization for ONNX artifacts
- `run_pipeline.py`: one-command pipeline runner and result logger
- `program.md`: instructions for an autonomous coding agent
- `bootstrap_wsl.sh`: WSL bootstrap helper
- `results.tsv`: experiment log

## Directory assumptions

This workspace assumes it lives next to your current Whisper project:

- `../finetune model/svarah_train_raw`
- `../finetune model/svarah_eval_raw`
- `../finetune model/svarah_train_preprocessed`
- `../finetune model/svarah_eval_preprocessed`
- `../finetune model/whisper-small-svarah-lora-final`

If your remote GPU machine uses different paths, update `config.json`.

## Important limitation

Your current pipeline ends at the LoRA adapter. Sherpa needs exported ONNX artifacts:

- encoder ONNX
- decoder ONNX
- tokens file

This workspace isolates that step behind `export.hook_command` in `config.json`.

Reason: Sherpa's official Whisper exporter is primarily documented for standard Whisper checkpoints and may need adaptation for a locally merged fine-tuned model. The workspace is designed so the agent can work on that export step directly without changing the rest of the evaluation pipeline.

## Baseline workflow

1. Copy this folder and your `finetune model` folder to the WSL-accessible GPU machine.
2. In WSL, run `bash bootstrap_wsl.sh`
3. Edit `config.json`
4. Set `export.hook_command`
5. Run:

```bash
source .venv/bin/activate
python run_pipeline.py --config config.json --description "baseline"
```

## What gets measured

The pipeline records:

- merged HF WER/WAR
- Sherpa fp32 WER/WAR if export works
- Sherpa int8 WER/WAR if quantization works

The metric you actually want to optimize is:

- primary: `sherpa_int8_wer`
- secondary: `sherpa_fp32_wer`
- tertiary: model size and Android latency

## Suggested export hook shape

Set `export.hook_command` to a shell command. Available placeholders:

- `{base_model_name}`
- `{merged_model_dir}`
- `{onnx_fp32_dir}`
- `{repo_root}`
- `{workspace_root}`

Example shape:

```bash
bash -lc 'cd {repo_root}/autoresearch-stt/external/sherpa-onnx && python3 scripts/whisper/export-onnx.py ...'
```

The command must write the fp32 ONNX artifacts into `{onnx_fp32_dir}`.

## Manual commands

Prepare eval audio:

```bash
python prepare_eval_audio.py --config config.json
```

Merge LoRA:

```bash
python merge_lora.py --config config.json
```

Evaluate merged HF model:

```bash
python evaluate_hf.py --config config.json --model-path artifacts/merged-model
```

Quantize exported ONNX:

```bash
python quantize_onnx.py --input-dir artifacts/onnx-fp32 --output-dir artifacts/onnx-int8
```

Evaluate Sherpa export:

```bash
python evaluate_sherpa.py --config config.json --onnx-dir artifacts/onnx-int8
```
