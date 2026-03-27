# Autoresearch-STT: Whisper to Android Deployment Pipeline

Autoresearch-STT is an agentic research system designed to automate the process of fine-tuning, merging, exporting, and deploying Whisper models to Android using the Sherpa-ONNX framework.

## The Problem: Edge Deployment Accuracy Loss

When deploying fine-tuned Whisper models to edge devices (Android) via Sherpa-ONNX, there is often a significant drop in transcription accuracy compared to the original Hugging Face model. This degradation occurs during the conversion to ONNX format and subsequent int8 quantization required for mobile performance.

To solve this, the Autoresearch system was implemented to systematically test multiple deployment configurations, quantization strategies, and preprocessing techniques to find the optimal path that preserves accuracy while maintaining mobile efficiency.

---

## Fine-tuned Model

The pre-trained/fine-tuned model used in this pipeline is available on Hugging Face:
[Akshatkasera007/Svarah-Whisper-v1](https://huggingface.co/Akshatkasera007/Svarah-Whisper-v1)

---

## Experimental Results and Trials

The following table summarizes the research trials conducted using this system to minimize the accuracy gap.

| Trial Description | HF WER (%) | Sherpa FP32 WER (%) | Sherpa INT8 WER (%) | Status |
|---|---|---|---|---|
| baseline-final | 14.29 | 14.91 | 15.44 | OK |
| quantize-encoder-only | 14.29 | 14.91 | 15.25 | OK |
| decoder-ffn-int8 | 14.29 | 14.91 | 15.24 | OK |
| decoder-top2-fc1-only | 14.29 | 14.91 | 15.18 | OK |
| **normalize-strip-punctuation** | **9.47** | **10.31** | **10.57** | **Best** |
| tail-paddings-300 | 9.47 | 12.63 | 13.02 | OK |

**Current Deployment State**: We are currently deploying the **normalize-strip-punctuation** configuration, as it achieved the lowest Word Error Rate (WER) across all deployment formats.

---

## Key Features

- Automated LoRA Merging: Seamlessly merges your fine-tuned LoRA adapters with the base Whisper model.
- Optimized ONNX Export: Converts merged models into Sherpa-compatible ONNX artifacts.
- Int8 Quantization: Dynamic quantization to reduce model size and improve Android inference speed.
- End-to-End Evaluation: Measures WER and WAR through each stage (HF Merged -> FP32 ONNX -> INT8 ONNX).
- Agentic Program Loop: Includes program.md specifically designed for autonomous coding agents to iterate on the pipeline.

---

## Project Structure

- run_pipeline.py: The main entry point. Orchestrates the entire research/deployment loop.
- merge_lora.py: Merges LoRA weights into the base model.
- export_hf_to_sherpa_onnx.py: Handles the conversion to Sherpa-ONNX format.
- quantize_onnx.py: Applies int8 quantization to the exported models.
- evaluate_sherpa.py: Benchmarks the performance of the exported ONNX model.
- prepare_eval_audio.py: Prepares your dataset for the evaluation pipeline.

---

## Getting Started

### 1. Prerequisites
- WSL2 (Ubuntu 22.04 recommended)
- Python 3.10+
- Access to a GPU for faster export/merging processes.

### 2. Setup
Clone the repository and run the bootstrap script:
```bash
./bootstrap_wsl.sh
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
Copy config.example.json to config.json and update your paths:
```bash
cp config.example.json config.json
```

### 4. Run the Pipeline
Execute the full research pipeline with a single command:
```bash
python run_pipeline.py --config config.json --description "Deployment optimization"
```

---

## Project Metrics
The pipeline logs results to results.tsv, tracking:
- merged_hf_wer: Performance of the merged model.
- sherpa_fp32_wer: Performance after ONNX export.
- sherpa_int8_wer: Performance after quantization.
