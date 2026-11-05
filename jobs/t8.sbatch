#!/bin/bash
# T8 on one H100 node: every demand copy reads its expert from the node's NVMe drive with direct I/O.
#   1. Checks: versions, CPUs, GPUs and the drive behind the experts' directory; unit tests; copy issue cost.
#   2. Calibration: the four arms, one pass, on the first 40 calibration sequences of T5's OLMoE trace; the development
#      costs and calibration gains are written and their sha256 logged ("PREDICTIONS FROZEN") before any new trace
#      exists.
#   3. A new OLMoE trace (examples 305-314 of each domain) and its replay spec, disjoint from every earlier trace.
#   4. The four arms on it, one pass.
# Steps skip finished output and predictions are never recomputed. NVME_DIR sets the experts' directory (default: the
# job's local disk). Submit from the repository root:  sbatch jobs/t8.sbatch
#SBATCH --job-name=t8
#SBATCH --gpus=h100:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=t8-%j.out
set -uo pipefail
LOG=logs/t8.log
source jobs/common.sh
export CUDA_VISIBLE_DEVICES=0
OFFSET=305
PER=10
CALN=40
CAP=32
NVME=${NVME_DIR:-${SLURM_TMPDIR:-/tmp}/t8_experts}
R=data/rankers
OUT=data/runs/t8/olmoe
CALSPEC=data/specs/olmoe_t5_replay_spec.json
TRACE=data/traces/olmoe_t8
SPEC=data/specs/olmoe_t8_replay_spec.json
ARMS=(lfu_decay hobbit splitphase global_token_gbt)
host_info
CPUS=$(python -c "import os; print(len(os.sched_getaffinity(0)))")
echo "CPUs this step may use: $CPUS" | tee -a "$LOG"
[ "$CPUS" -ge 16 ] || fail "only $CPUS CPUs; the placement needs whole cores for the decode process and the worker"
mkdir -p "$NVME" "$OUT"
df -hT "$NVME" | tee -a "$LOG"
findmnt -T "$NVME" -o SOURCE,FSTYPE,TARGET | tee -a "$LOG"
lsblk -d -o NAME,ROTA,TRAN,SIZE,MODEL | tee -a "$LOG"
free -g | tee -a "$LOG"
step tests
python -m unittest tests.test_hardware tests.test_baselines tests.test_prefetch_architectures tests.test_freerun \
  tests.test_nvme -q >> "$LOG" 2>&1 || fail tests
step "copy issue cost"
python -u src/runtime/bench_copy_issue.py 2>&1 | tee -a "$LOG"

COMMON=(--nvme-dir "$NVME" --placement physical --monitor --lean --copy-api driver --warmup 1 --gpu-index 0 --passes 1
        --hobbit-w "$(hobbit_w olmoe frozen $CAP)" --p2-bundle $R/olmoe_frozen_lag_req_trees200.pkl
        --p1-bundle $R/olmoe_frozen_lag_tok_trees200.pkl --sync-bundle $R/olmoe_setdev_policy.pkl --arms "${ARMS[@]}")

frozen() { [ -f "$OUT/dev_costs_e18.json" ] && [ -f "$OUT/predictions_e18.json" ]; }
# no new trace and no test output may exist before the predictions are frozen
if ! frozen && { [ -e "$TRACE" ] || [ -f "$OUT/B${CAP}_nvme.json" ]; }; then
  fail "a new trace or a test output exists before the predictions are frozen"
fi
if [ ! -f "$OUT/cal_B${CAP}_nvme.json" ]; then
  step "calibration: four arms, experts read from the drive ($CALN sequences, 1 pass)"
  python -u src/runtime/interleaved_nvme.py --spec "$CALSPEC" --capacity "$CAP" --sequences "$(ids "$CALSPEC" $CALN)" \
    "${COMMON[@]}" --output "$OUT/cal_B${CAP}_nvme.json" --label "t8/olmoe/cal_B${CAP}_nvme" >> "$LOG" 2>&1 \
    || fail calibration
fi
if ! frozen; then
  python analysis/dev_costs.py "$OUT/cal_B${CAP}_nvme.json" --output "$OUT/dev_costs_e18.json" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "development costs"
  python analysis/predictions_calibration.py "$OUT/cal_B${CAP}_nvme.json" --output "$OUT/predictions_e18.json" 2>&1 \
    | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail predictions
  step "PREDICTIONS FROZEN"
  sha256sum "$OUT/dev_costs_e18.json" "$OUT/predictions_e18.json" "$OUT/cal_B${CAP}_nvme.json" | tee -a "$LOG"
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
    --original-metadata "${META_T1_T7[@]}" >> "$LOG" 2>&1 || fail spec
fi
if [ ! -f "$OUT/B${CAP}_nvme.json" ]; then
  step "four arms, experts read from the drive (new trace, 1 pass)"
  python -u src/runtime/interleaved_nvme.py --spec "$SPEC" --capacity "$CAP" --sequences calibration "${COMMON[@]}" \
    --output "$OUT/B${CAP}_nvme.json" --label "t8/olmoe/B${CAP}_nvme" >> "$LOG" 2>&1 || fail "test run"
fi
cp "$LOG" "$OUT/t8.log"
step done
