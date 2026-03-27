# autoresearch-stt

This workspace is for autonomous improvement of a Whisper-based STT model that must run on Android with Sherpa at high accuracy.

## Objective

Lower deployed error, not just training loss.

The priority order is:

1. `sherpa_int8_wer` on the fixed eval manifest
2. `sherpa_fp32_wer`
3. `hf_merged_wer`
4. Android size, memory, and latency

## In-scope files

You may edit:

- `run_pipeline.py`
- `prepare_eval_audio.py`
- `merge_lora.py`
- `evaluate_hf.py`
- `evaluate_sherpa.py`
- `quantize_onnx.py`
- `config.json`
- `README.md`
- `program.md`
- `../finetune model/03_train.py`
- new helper files in this directory

You should not edit:

- raw dataset contents
- eval references
- the original GPT demo files in the repo root unless absolutely necessary

## Ground truth

The deployed target is Sherpa on Android, after quantization.

Do not keep a change only because HF WER improved. If Sherpa int8 got worse, the change is a failure.

## First priority

Make the export path correct.

Before running broad research on hyperparameters, establish parity:

1. merged HF eval works
2. Sherpa fp32 export works
3. Sherpa fp32 roughly matches HF
4. quantify the drop from fp32 to int8

Only after that should you spend time on training experiments.

## Experiment loop

1. Check git status and current commit.
2. Pick exactly one idea.
3. Make the change.
4. Run:

```bash
python run_pipeline.py --config config.json --description "short description"
```

5. Read the JSON summary in `artifacts/run-summaries/`.
6. Append or verify the row in `results.tsv`.
7. Keep the change only if the primary objective improved.
8. If the run failed or got worse, revert and try a different idea.

## High-value ideas

- export correctness
- decoder prompt parity with HF
- text normalization parity
- encoder-only quantization vs encoder+decoder quantization
- LoRA target modules
- LoRA rank and dropout
- learning rate, warmup, eval cadence
- batch size and gradient accumulation
- base checkpoint choice such as `whisper-small` vs `whisper-small.en`
- merging and save format correctness

## Logging

Every run must have:

- a short description
- git commit
- HF merged WER
- Sherpa fp32 WER if available
- Sherpa int8 WER if available
- pass/fail status

## Operating mode

Keep changes small and testable.

Do not start long training runs until the export/eval path is proven. The fastest path is:

1. get the measurement system right
2. get one working fp32 export
3. measure int8 loss
4. optimize the actual bottleneck
