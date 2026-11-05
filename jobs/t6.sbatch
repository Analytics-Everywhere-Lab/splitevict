#!/bin/bash
# T6 on one H100 node: Mixtral (B=4) at higher copy costs. A new trace (examples 280-284 of each domain), its replay
# spec, then the four policies interleaved, one pass, with every expert copy issued K = 1, 2, 7 and 4 times.
# Steps skip finished output. Submit from the repository root:  sbatch jobs/t6.sbatch
#SBATCH --job-name=t6
#SBATCH --gpus=h100:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --output=t6-%j.out
set -uo pipefail
LOG=logs/t6.log
source jobs/common.sh
export CUDA_VISIBLE_DEVICES=0
OFFSET=280
PER=5
CAP=4
R=data/rankers
OUT=data/runs/t6/mixtral
TRACE=data/traces/mixtral_t6
SPEC=data/specs/mixtral_t6_replay_spec.json
COMMON=(--placement physical --monitor --lean --copy-api driver --warmup 1 --gpu-index 0 --experts-on-host)
ARMS=(lfu_decay hobbit splitphase global_token_gbt)
MX=(--p2-bundle $R/mixtral_8x7b_frozen_lag_req_trees200.pkl --p1-bundle $R/mixtral_8x7b_frozen_lag_tok_trees200.pkl
    --sync-bundle $R/mixtral_8x7b_dev_trees800.pkl)
host_info
step tests
python -m unittest tests.test_hardware tests.test_baselines tests.test_prefetch_architectures -q >> "$LOG" 2>&1 || fail tests
step "copy issue cost"
python -u src/runtime/bench_copy_issue.py 2>&1 | tee -a "$LOG" || skip "copy benchmark"
if [ ! -f "$TRACE/routing_trace.npz" ]; then
  step "generate trace (examples $OFFSET-$((OFFSET + PER - 1)) of each domain)"
  python -u src/runtime/fresh_trace_offloaded.py generate --opt-root measurement --base-config configs/trace_mixtral_8x7b.json \
    --offset $OFFSET --prompts-per-domain $PER --capacity 5 --output-dir "$TRACE" >> "$LOG" 2>&1 || fail generate
fi
if [ ! -f "$SPEC" ]; then
  step "replay spec"
  python -u src/runtime/fresh_trace_offloaded.py spec --opt-root measurement --trace-root "$TRACE" --output "$SPEC" \
    --original-metadata "${META_T1_T5[@]}" >> "$LOG" 2>&1 || fail spec
fi
W=$(hobbit_w mixtral frozen $CAP) || fail "HOBBIT weight"
mkdir -p "$OUT"
for K in 1 2 7 4; do
  [ -f "$OUT/B${CAP}_K$K.json" ] && continue
  step "arms B${CAP}_K$K, 1 pass"
  python -u src/runtime/interleaved_h100.py --spec "$SPEC" --capacity $CAP --sequences calibration --passes 1 \
    --copy-repeat "$K" "${COMMON[@]}" --hobbit-w "$W" "${MX[@]}" --arms "${ARMS[@]}" \
    --output "$OUT/B${CAP}_K$K.json" --label "t6/mixtral/B${CAP}_K$K" >> "$LOG" 2>&1 || skip "arms K=$K"
done
grep -q "^FAILED" "$LOG" && step "finished with failures (see FAILED lines)" || step done
