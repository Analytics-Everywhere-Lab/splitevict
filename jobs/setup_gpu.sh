#!/usr/bin/env bash
# One-time setup for the GPU tests: Python environment, the three models and the four datasets at pinned revisions.
# Run on a machine with internet access; the test jobs themselves run offline.
#   VENV=$HOME/venvs/splitphase bash jobs/setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements-gpu.txt
python -c "import torch, transformers, sklearn, numba; print('versions', torch.__version__, torch.version.cuda, transformers.__version__, sklearn.__version__, numba.__version__)"

hf download allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5
hf download Qwen/Qwen3-30B-A3B --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --include "*.safetensors" "*.json" "*.txt"
hf download mistralai/Mixtral-8x7B-v0.1 --revision fc7ac94680e38d7348cfa806e51218e6273104b0 \
  --include "*.safetensors" "*.json" "*.model" "*.txt"

python - <<'PY'
from datasets import load_dataset
pinned = {
    ("Salesforce/wikitext", "wikitext-103-raw-v1", "test"): "b08601e04326c79dfdd32d625aee71d232d685c3",
    ("google-research-datasets/mbpp", "full", "test"): "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    ("openai/gsm8k", "main", "test"): "740312add88f781978c0658806c59bc2815b9866",
    ("allenai/ai2_arc", "ARC-Challenge", "test"): "210d026faf9955653af8916fad021475a3f00453",
}
for (path, name, split), revision in pinned.items():
    ds = load_dataset(path, name, split=split, revision=revision)
    print("cached", path, name, split, revision[:8], len(ds), "rows")
PY
echo "setup done"
