# 🚀 Autoresearch-STT: Whisper to Android Deployment Pipeline

**Autoresearch-STT** is an agentic research system designed to automate the process of fine-tuning, merging, exporting, and deploying Whisper models to Android using the [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) framework.

This workspace bridges the gap between a raw Hugging Face LoRA adapter and a production-ready, quantized ONNX model running on a mobile device.

---

## 📦 Fine-tuned Model

The pre-trained/fine-tuned model used in this pipeline is available on Hugging Face:
🔗 **[Akshatkasera007/Svarah-Whisper-v1](https://huggingface.co/Akshatkasera007/Svarah-Whisper-v1)**

---

## ✨ Key Features

- **Automated LoRA Merging**: Seamlessly merges your fine-tuned LoRA adapters with the base Whisper model.
- **Optimized ONNX Export**: Converts merged models into Sherpa-compatible ONNX artifacts.
- **Int8 Quantization**: Dynamic quantization to reduce model size and improve Android inference speed.
- **End-to-End Evaluation**: Measures WER (Word Error Rate) and WAR (Word Accuracy Rate) through each stage (HF Merged -> FP32 ONNX -> INT8 ONNX).
- **Agentic Program Loop**: Includes `program.md` specifically designed for autonomous coding agents to iterate on the pipeline.

---

## 🛠️ Project Structure

- `run_pipeline.py`: The main entry point. Orchestrates the entire research/deployment loop.
- `merge_lora.py`: Merges LoRA weights into the base model.
- `export_hf_to_sherpa_onnx.py`: Handles the complex conversion to Sherpa-ONNX format.
- `quantize_onnx.py`: Applies int8 quantization to the exported models.
- `evaluate_sherpa.py`: Benchmarks the performance of the exported ONNX model.
- `prepare_eval_audio.py`: Prepares your dataset for the evaluation pipeline.

---

## 🚀 Getting Started

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
Copy `config.example.json` to `config.json` and update your paths:
```bash
cp config.example.json config.json
```
Ensure your fine-tuned weights and base model paths are correctly pointed in the configuration.

### 4. Run the Pipeline
Execute the full research pipeline with a single command:
```bash
python run_pipeline.py --config config.json --description "First deployment test"
```

---

## 📊 Deployment Metrics
The pipeline logs results to `results.tsv`, tracking:
- **`merged_hf_wer`**: Performance of the merged model.
- **`sherpa_fp32_wer`**: Performance after ONNX export.
- **`sherpa_int8_wer`**: Performance after quantization (this is your target for Android).

---

## 📄 License
Check the `LICENSE` file for details (if applicable). Developed for efficient Whisper-to-Android research.
