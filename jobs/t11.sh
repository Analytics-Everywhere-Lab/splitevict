#!/bin/bash
# T11 on a desktop with one RTX 4090 (PCIe gen4 x16) and an Intel i9-13900K, OLMoE at B = 32, in two settings:
# native (every demand copy from pinned host memory) and nvme (every demand copy read from the local NVMe drive with
# direct I/O, T8's reader). The decode process and the worker use P-cores only (src/runtime/interleaved_desktop.py).
#   1. Checks: versions, CPUs (P-cores, E-cores, clocks), GPU, the drive behind the experts' directory; unit tests; copy
#      issue cost.
#   2. Calibration, native then nvme: the four arms, two passes, on the first 40 calibration sequences of T5's OLMoE
#      trace; the development costs and predictions of both settings are written and their sha256 logged ("PREDICTIONS
#      FROZEN") before any new trace exists.
#   3. A new OLMoE trace (examples 325-334 of each domain), generated on the 4090, and its replay spec, disjoint from
#      every earlier trace.
#   4. The four arms on it, two passes, native then nvme.
# On a shared machine: before every GPU step the script waits until no process is on the GPU (each wait is logged), and
# around every run it records the drive's /proc/diskstats line, the GPU's processes every 30 s and at the end, and
# (nvme) the expert files' bytes in the page cache. Steps skip finished output and predictions are never recomputed.
# NVME_DIR sets the experts' directory on the drive. Run from the repository root:  bash jobs/t11.sh
set -uo pipefail
LOG=logs/t11.log
source jobs/common.sh
export CUDA_VISIBLE_DEVICES=0
OFFSET=325
PER=10
CALN=40
CAP=32
PASSES=2
NVME=${NVME_DIR:-data/t11_experts}
R=data/rankers
OUT=data/runs/t11/olmoe
CALSPEC=data/specs/olmoe_t5_replay_spec.json
TRACE=data/traces/olmoe_t11
SPEC=data/specs/olmoe_t11_replay_spec.json
ARMS=(lfu_decay hobbit splitphase global_token_gbt)
SETTINGS=(native nvme)
META=("${META_T1_T7[@]}")
LATER=(data/traces/olmoe_t8/routing_trace.metadata.json data/traces/olmoe_t9/routing_trace.metadata.json)

step "host"
python -c "import torch, transformers, sklearn, numba, numpy; print('versions', torch.__version__, torch.version.cuda, transformers.__version__, sklearn.__version__, numba.__version__, numpy.__version__, torch.cuda.get_device_name(0))" | tee -a "$LOG" || fail versions
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current --format=csv | tee -a "$LOG"
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA node" | tee -a "$LOG"
echo "P-cores (cpu_core) $(cat /sys/devices/cpu_core/cpus), E-cores (cpu_atom) $(cat /sys/devices/cpu_atom/cpus); governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" | tee -a "$LOG"
for c in $(seq 0 15); do echo -n "cpu$c $(cat /sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq) kHz, "; done | tee -a "$LOG"; echo | tee -a "$LOG"
mkdir -p "$NVME" "$OUT"
df -hT "$NVME" | tee -a "$LOG"
findmnt -T "$NVME" -o SOURCE,FSTYPE,TARGET,OPTIONS | tee -a "$LOG"
DEV=$(basename "$(findmnt -n -o SOURCE -T "$NVME")")
lsblk -d -o NAME,ROTA,TRAN,SIZE,MODEL "/dev/$DEV" | tee -a "$LOG"
echo "max_sectors_kb $(cat /sys/block/$DEV/queue/max_sectors_kb)" | tee -a "$LOG"
free -g | tee -a "$LOG"

gpu_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }
wait_gpu() {  # until no process is on the GPU; every wait is logged
  local apps; apps=$(gpu_apps)
  while [ -n "$apps" ]; do
    echo "$(date -u +%FT%TZ) waiting, process(es) on the GPU: $(echo $apps)" | tee -a "$LOG"
    sleep 60; apps=$(gpu_apps)
  done
}
wait_gpu
step tests
python -m unittest tests.test_hardware tests.test_baselines tests.test_prefetch_architectures tests.test_freerun \
  tests.test_nvme -q >> "$LOG" 2>&1 || fail tests
step "copy issue cost"
python -u src/runtime/bench_copy_issue.py 2>&1 | tee -a "$LOG"

COMMON=(--placement physical --monitor --lean --copy-api driver --warmup 1 --gpu-index 0 --passes $PASSES
        --hobbit-w "$(hobbit_w olmoe frozen $CAP)" --p2-bundle $R/olmoe_frozen_lag_req_trees200.pkl
        --p1-bundle $R/olmoe_frozen_lag_tok_trees200.pkl --sync-bundle $R/olmoe_setdev_policy.pkl --arms "${ARMS[@]}")

run_arms() {  # run_arms SETTING SPEC SEQUENCES NAME: the four arms in one process, two passes -> $OUT/NAME.json
  local setting=$1 spec=$2 seqs=$3 name=$4 extra=()
  [ "$setting" = nvme ] && extra=(--nvme-dir "$NVME")
  wait_gpu
  grep " $DEV " /proc/diskstats > "$OUT/$name.diskstats"
  python -u src/runtime/interleaved_desktop.py --spec "$spec" --capacity "$CAP" --sequences "$seqs" "${extra[@]}" \
    "${COMMON[@]}" --output "$OUT/$name.json" --label "t11/olmoe/$name" >> "$LOG" 2>&1 &
  local pid=$!
  echo "runner $pid" > "$OUT/$name.gpu_apps"
  taskset -c 16-31 nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv,noheader -l 30 \
    >> "$OUT/$name.gpu_apps" 2>&1 &
  local mon=$!
  wait $pid
  local rc=$?
  kill $mon
  grep " $DEV " /proc/diskstats >> "$OUT/$name.diskstats"
  gpu_apps > "$OUT/$name.gpu_end"
  echo "$name: runner pid $pid exit $rc; processes on the GPU at the end: $(cat "$OUT/$name.gpu_end" | tr '\n' ' ')" | tee -a "$LOG"
  if [ "$setting" = nvme ]; then
    find "$NVME" -name 'layer*.bin' -exec fincore -b -n -o RES {} + | awk '{s += $1} END {print s + 0}' > "$OUT/$name.fincore"
    echo "$name: expert files' bytes in the page cache after the run: $(cat "$OUT/$name.fincore")" | tee -a "$LOG"
  fi
  [ $rc -eq 0 ] || fail "$name"
}

frozen() { [ -f "$OUT/dev_costs_e23.json" ] && [ -f "$OUT/predictions_e23.json" ]; }
# no new trace and no test output may exist before the predictions are frozen
if ! frozen && { [ -e "$TRACE" ] || [ -f "$OUT/B${CAP}_native.json" ] || [ -f "$OUT/B${CAP}_nvme.json" ]; }; then
  fail "a new trace or a test output exists before the predictions are frozen"
fi
for s in "${SETTINGS[@]}"; do
  if [ ! -f "$OUT/cal_B${CAP}_$s.json" ]; then
    step "calibration ($s): four arms, $PASSES passes ($CALN sequences)"
    run_arms "$s" "$CALSPEC" "$(ids "$CALSPEC" $CALN)" "cal_B${CAP}_$s"
  fi
done
if ! frozen; then
  python analysis/dev_costs.py "$OUT/cal_B${CAP}_native.json" "$OUT/cal_B${CAP}_nvme.json" \
    --output "$OUT/dev_costs_e23.json" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "development costs"
  python analysis/predictions_t11.py --native "$OUT/cal_B${CAP}_native.json" --nvme "$OUT/cal_B${CAP}_nvme.json" \
    --output "$OUT/predictions_e23.json" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail predictions
  step "PREDICTIONS FROZEN"
  sha256sum "$OUT/dev_costs_e23.json" "$OUT/predictions_e23.json" "$OUT/cal_B${CAP}_native.json" \
    "$OUT/cal_B${CAP}_nvme.json" | tee -a "$LOG"
fi
if [ ! -f "$TRACE/routing_trace.npz" ]; then
  wait_gpu
  step "new trace (examples $OFFSET-$((OFFSET + PER - 1)) of each domain)"
  python -u src/runtime/fresh_trace_offloaded.py generate --opt-root measurement \
    --base-config measurement/stage3_residency/configs/full_preregistered.json --offset $OFFSET --prompts-per-domain $PER \
    --capacity $CAP --output-dir "$TRACE" >> "$LOG" 2>&1 || fail generate
fi
if [ ! -f "$SPEC" ]; then
  for m in "${LATER[@]}"; do [ -f "$m" ] && META+=("$m"); done
  step "replay spec, disjoint from ${#META[@]} earlier traces"
  python -u src/runtime/fresh_trace_offloaded.py spec --opt-root measurement --trace-root "$TRACE" --output "$SPEC" \
    --original-metadata "${META[@]}" >> "$LOG" 2>&1 || fail spec
fi
for s in "${SETTINGS[@]}"; do
  if [ ! -f "$OUT/B${CAP}_$s.json" ]; then
    step "$s: four arms, $PASSES passes (new trace)"
    run_arms "$s" "$SPEC" calibration "B${CAP}_$s"
  fi
done
cp "$LOG" "$OUT/t11.log"
step done
