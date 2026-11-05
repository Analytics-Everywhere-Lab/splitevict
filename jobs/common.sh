# Shared settings for the GPU job scripts (sourced from the repository root).
VENV=${VENV:-.venv}
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=1
mkdir -p logs
step() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
fail() { echo "FAILED: $*" | tee -a "$LOG"; exit 1; }
skip() { echo "FAILED (continuing with the next step): $*" | tee -a "$LOG"; }
host_info() {
  python -c "import torch, transformers, sklearn, numba; print('versions', torch.__version__, torch.version.cuda, transformers.__version__, sklearn.__version__, numba.__version__, torch.cuda.get_device_name(0))" | tee -a "$LOG" || fail versions
  nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv | tee -a "$LOG"
  lscpu | grep -E "Model name|Socket|Core|Thread|NUMA node" | tee -a "$LOG"
}
# first N calibration sequence ids of a replay spec, comma-separated
ids() { python -c "import json; s=json.load(open('$1')); print(','.join(map(str, s['calibration_sequence_ids'][${3:-0}:$2])))"; }
# HOBBIT's weight fitted on calibration data: hobbit_w MODEL FIT CAPACITY
hobbit_w() { python -c "import json; print(json.load(open('data/baselines/$1_$2.json'))['hobbit']['$3']['w'])"; }
# metadata of every earlier trace: new prompts must be disjoint from all of them
META_T1_T4=(measurement/stage3_residency/traces/full/routing_trace.metadata.json
            data/traces/olmoe_t1/routing_trace.metadata.json data/traces/olmoe_t4/routing_trace.metadata.json
            data/traces/qwen3_30b_a3b/routing_trace.metadata.json data/traces/qwen3_30b_a3b_t3/routing_trace.metadata.json
            data/traces/mixtral_8x7b/routing_trace.metadata.json data/traces/mixtral_8x7b_t3/routing_trace.metadata.json)
META_T1_T5=("${META_T1_T4[@]}" data/traces/olmoe_t5/routing_trace.metadata.json
            data/traces/qwen3_t5/routing_trace.metadata.json data/traces/mixtral_t5/routing_trace.metadata.json)
META_T1_T7=("${META_T1_T5[@]}" data/traces/mixtral_t6/routing_trace.metadata.json
            data/traces/olmoe_t7/routing_trace.metadata.json data/traces/qwen3_t7/routing_trace.metadata.json)
