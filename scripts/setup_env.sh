#!/bin/bash
# Recreate the project environment on the GPU server.
# Usage: bash scripts/setup_env.sh
set -e
cd "$(dirname "$0")/.."

export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Non-interactive conda needs the default channel terms of service accepted first.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

if ! conda env list | grep -q "^nlp "; then
  conda create -n nlp python=3.10 -y
fi
conda activate nlp

python -m pip install --upgrade pip -q
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q
python -m pip install -r requirements.txt -q

python - <<'PY'
import torch, transformers, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
PY
