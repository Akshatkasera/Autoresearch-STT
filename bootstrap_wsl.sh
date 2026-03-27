#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

mkdir -p artifacts logs external

if [ ! -d external/sherpa-onnx ]; then
  git clone --depth 1 https://github.com/k2-fsa/sherpa-onnx external/sherpa-onnx
fi

if [ ! -f config.json ]; then
  cp config.example.json config.json
fi

cat <<'EOF'
Bootstrap complete.

Next:
1. Edit config.json.
2. Set export.hook_command to your custom Whisper->Sherpa export command.
3. Run: source .venv/bin/activate
4. Run: python run_pipeline.py --config config.json --description "baseline"
EOF
