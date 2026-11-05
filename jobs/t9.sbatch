#!/bin/bash
# T9 on one H100 node: the whole decode step replayed from CUDA graphs, one split-phase worker, copies from pinned host
# memory issued K = 1, 2 and 4 times.
#   1. Checks: versions, CPUs, GPUs; unit tests; the graph check (graph replay must give exactly the tokens of the same
#      pieces run eagerly, or the job stops).
#   2. Calibration at K = 1, 2, 4: three arms, one pass, on the first 40 calibration sequences of T5's OLMoE trace; the
#      predictions are written and their sha256 logged ("PREDICTIONS FROZEN") before any new trace exists.
#   3. A new OLMoE trace (examples 315-324 of each domain) and its replay spec, disjoint from every earlier trace.
#   4. The three arms on it at K = 1, 2, 4, one pass each.
# Steps skip finished output and predictions are never recomputed. Submit from the repository root:  sbatch jobs/t9.sbatch
#SBATCH --job-name=t9
#SBATCH --gpus=h100:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=t9-%j.out
set -uo pipefail
LOG=logs/t9.log
source jobs/common.sh
export CUDA_VISIBLE_DEVICES=0
OFFSET=315
PER=10
CALN=40
CAP=32
KS=(1 2 4)
R=data/rankers
OUT=data/runs/t9/olmoe
CALSPEC=data/specs/olmoe_t5_replay_spec.json
TRACE=data/traces/olmoe_t9
SPEC=data/specs/olmoe_t9_replay_spec.json
ARMS=(lfu_decay hobbit splitphase)
META=("${META_T1_T7[@]}")
[ -f data/traces/olmoe_t8/routing_trace.metadata.json ] && META+=(data/traces/olmoe_t8/routing_trace.metadata.json)
host_info
CPUS=$(python -c "import os; print(len(os.sched_getaffinity(0)))")
echo "CPUs this step may use: $CPUS" | tee -a "$LOG"
[ "$CPUS" -ge 16 ] || fail "only $CPUS CPUs; the placement needs whole cores for the decode process and the worker"
mkdir -p "$OUT"
step tests
python -m unittest tests.test_hardware tests.test_baselines tests.test_freerun tests.test_graphs -q >> "$LOG" 2>&1 \
  || fail tests
if [ ! -f "$OUT/graph_check.json" ]; then
  step "graph check (T5 prompts, 2 sequences, decayed LFU)"
  python -u src/runtime/check_graphs.py --spec "$CALSPEC" --capacity "$CAP" --output "$OUT/graph_check.json" 2>&1 \
    | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "graph check"
fi

COMMON=(--placement physical --monitor --lean --copy-api driver --warmup 1 --gpu-index 0 --passes 1
        --hobbit-w "$(hobbit_w olmoe frozen $CAP)" --p2-bundle $R/olmoe_frozen_lag_req_trees200.pkl
        --p1-bundle $R/olmoe_frozen_lag_tok_trees200.pkl --arms "${ARMS[@]}")

frozen() { [ -f "$OUT/predictions_e19.json" ]; }
# no new trace and no test output may exist before the predictions are frozen
if ! frozen && { [ -e "$TRACE" ] || ls "$OUT"/B${CAP}_K*.json >/dev/null 2>&1; }; then
  fail "a new trace or a test output exists before the predictions are frozen"
fi
for K in "${KS[@]}"; do
  [ -f "$OUT/cal_B${CAP}_K$K.json" ] && continue
  step "calibration K=$K: three arms, graphed decode ($CALN sequences, 1 pass)"
  python -u src/runtime/interleaved_graphs.py --spec "$CALSPEC" --capacity "$CAP" --sequences "$(ids "$CALSPEC" $CALN)" \
    --copy-repeat "$K" "${COMMON[@]}" --output "$OUT/cal_B${CAP}_K$K.json" --label "t9/olmoe/cal_B${CAP}_K$K" \
    >> "$LOG" 2>&1 || fail "calibration K=$K"
done
if ! frozen; then
  CALS=()
  for K in "${KS[@]}"; do CALS+=("$OUT/cal_B${CAP}_K$K.json"); done
  python analysis/predictions_calibration.py "${CALS[@]}" --output "$OUT/predictions_e19.json" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail predictions
  step "PREDICTIONS FROZEN"
  sha256sum "$OUT/predictions_e19.json" "${CALS[@]}" | tee -a "$LOG"
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
for K in "${KS[@]}"; do
  [ -f "$OUT/B${CAP}_K$K.json" ] && continue
  step "K=$K: three arms, graphed decode (new trace, 1 pass)"
  python -u src/runtime/interleaved_graphs.py --spec "$SPEC" --capacity "$CAP" --sequences calibration --copy-repeat "$K" \
    "${COMMON[@]}" --output "$OUT/B${CAP}_K$K.json" --label "t9/olmoe/B${CAP}_K$K" >> "$LOG" 2>&1 || fail "test run K=$K"
done
cp "$LOG" "$OUT/t9.log"
step done
