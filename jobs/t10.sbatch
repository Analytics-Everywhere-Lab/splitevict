#!/bin/bash
# T10 on one H100 node: T9's graphed decode step with split-phase's scores computed by two worker processes, each owning
# alternate layers, at the H100's own copy cost (K = 1), B = 32 (jobs/t10_b24.sbatch runs B = 24). T9's graph check
# covers this decode step.
#   1. Checks: versions, CPUs, GPUs; unit tests.
#   2. Calibration: three arms, one pass, on the first 40 calibration sequences of T5's OLMoE trace; the predictions are
#      written and their sha256 logged ("PREDICTIONS FROZEN") before any new trace exists.
#   3. A new OLMoE trace (examples 335-344 of each domain) and its replay spec, disjoint from every earlier trace.
#   4. The three arms on it, one pass.
# Steps skip finished output and predictions are never recomputed. Submit from the repository root:  sbatch jobs/t10.sbatch
#SBATCH --job-name=t10
#SBATCH --gpus=h100:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=t10-%j.out
set -uo pipefail
LOG=logs/t10.log
source jobs/common.sh
export CUDA_VISIBLE_DEVICES=0
OFFSET=335
PER=10
CALN=40
CAP=32
K=1
WORKERS=2
R=data/rankers
OUT=data/runs/t10/olmoe
CALSPEC=data/specs/olmoe_t5_replay_spec.json
TRACE=data/traces/olmoe_t10
SPEC=data/specs/olmoe_t10_replay_spec.json
ARMS=(lfu_decay hobbit splitphase)
META=("${META_T1_T7[@]}")
for t in olmoe_t8 olmoe_t9 olmoe_t11; do
  [ -f data/traces/$t/routing_trace.metadata.json ] && META+=(data/traces/$t/routing_trace.metadata.json)
done
host_info
CPUS=$(python -c "import os; print(len(os.sched_getaffinity(0)))")
echo "CPUs this step may use: $CPUS" | tee -a "$LOG"
[ "$CPUS" -ge 16 ] || fail "only $CPUS CPUs; the placement needs whole cores for the decode process and the workers"
mkdir -p "$OUT"
step tests
python -m unittest tests.test_hardware tests.test_baselines tests.test_freerun tests.test_graphs \
  tests.test_graphs_workers -q >> "$LOG" 2>&1 || fail tests

COMMON=(--splitphase-workers "$WORKERS" --placement physical --monitor --lean --copy-api driver --warmup 1 --gpu-index 0
        --passes 1 --copy-repeat "$K" --hobbit-w "$(hobbit_w olmoe frozen $CAP)"
        --p2-bundle $R/olmoe_frozen_lag_req_trees200.pkl --p1-bundle $R/olmoe_frozen_lag_tok_trees200.pkl --arms "${ARMS[@]}")

frozen() { [ -f "$OUT/predictions_e24.json" ]; }
# no new trace and no test output may exist before the predictions are frozen
if ! frozen && { [ -e "$TRACE" ] || [ -e "$OUT/B${CAP}_K$K.json" ]; }; then
  fail "a new trace or a test output exists before the predictions are frozen"
fi
if [ ! -f "$OUT/cal_B${CAP}_K$K.json" ]; then
  step "calibration: three arms, graphed decode, $WORKERS split-phase workers ($CALN sequences, 1 pass)"
  python -u src/runtime/interleaved_graphs_workers.py --spec "$CALSPEC" --capacity "$CAP" \
    --sequences "$(ids "$CALSPEC" $CALN)" "${COMMON[@]}" --output "$OUT/cal_B${CAP}_K$K.json" \
    --label "t10/olmoe/cal_B${CAP}_K$K" >> "$LOG" 2>&1 || fail calibration
fi
if ! frozen; then
  python analysis/predictions_calibration.py "$OUT/cal_B${CAP}_K$K.json" --output "$OUT/predictions_e24.json" 2>&1 \
    | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail predictions
  step "PREDICTIONS FROZEN"
  sha256sum "$OUT/predictions_e24.json" "$OUT/cal_B${CAP}_K$K.json" | tee -a "$LOG"
fi
if [ ! -f "$TRACE/routing_trace.npz" ]; then
  step "new trace (examples $OFFSET-$((OFFSET + PER - 1)) of each domain)"
  python -u src/runtime/fresh_trace_offloaded.py generate --opt-root measurement \
    --base-config measurement/stage3_residency/configs/full_preregistered.json --offset $OFFSET --prompts-per-domain $PER \
    --capacity $CAP --output-dir "$TRACE" >> "$LOG" 2>&1 || fail generate
fi
if [ ! -f "$SPEC" ]; then
  step "replay spec"
  python -u src/runtime/fresh_trace_offloaded.py spec --opt-root measurement --trace-root "$TRACE" --output "$SPEC" \
    --original-metadata "${META[@]}" >> "$LOG" 2>&1 || fail spec
fi
if [ ! -f "$OUT/B${CAP}_K$K.json" ]; then
  step "three arms, graphed decode, $WORKERS split-phase workers (new trace, 1 pass)"
  python -u src/runtime/interleaved_graphs_workers.py --spec "$SPEC" --capacity "$CAP" --sequences calibration \
    "${COMMON[@]}" --output "$OUT/B${CAP}_K$K.json" --label "t10/olmoe/B${CAP}_K$K" >> "$LOG" 2>&1 || fail "test run"
fi
cp "$LOG" "$OUT/t10.log"
step done
