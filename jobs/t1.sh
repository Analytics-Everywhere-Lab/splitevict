#!/usr/bin/env bash
# T1: OLMoE on an A40, unseen prompts, each policy in its own pass, then a final decayed-LFU pass (drift check).
# Steps skip finished output, so the script resumes.
#   bash jobs/t1.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/t1.log
source jobs/common.sh
TRACE=data/traces/olmoe_t1
FROZEN=data/prepared/olmoe_t1
SPEC=data/specs/olmoe_t1_replay_spec.json
OUT=data/runs/t1
host_info
step tests
python -m unittest tests.test_hardware -q >> "$LOG" 2>&1 || fail tests
if [ ! -f "$TRACE/routing_trace.npz" ]; then
  step "generate trace"
  python -u src/runtime/fresh_trace.py generate --opt-root measurement --output-dir "$TRACE" >> "$LOG" 2>&1 || fail generate
fi
if [ ! -f "$FROZEN/frozen_workloads.json" ]; then
  step "workload paths"
  python -u src/runtime/fresh_trace.py prepare --opt-root measurement --trace-root "$TRACE" --out "$FROZEN" \
    --original-metadata measurement/stage3_residency/traces/full/routing_trace.metadata.json >> "$LOG" 2>&1 || fail prepare
fi
if [ ! -f "$SPEC" ]; then
  step "replay spec"
  python -u src/runtime/build_replay_spec.py --repository-root measurement --fresh-trace-root "$TRACE" \
    --fresh-frozen-dir "$FROZEN" --output "$SPEC" >> "$LOG" 2>&1 || fail spec
fi
SP=(--p2-bundle data/rankers/olmoe_frozen_lag_req_trees200.pkl --p1-bundle data/rankers/olmoe_frozen_lag_tok_trees200.pkl --workers 1)
SYNC=(--policy-bundle data/rankers/olmoe_setdev_policy.pkl)
run() {
  local out="$1"; shift
  step "run $*"
  python -u src/runtime/offload_runtime.py --spec "$SPEC" --workloads mixed_interleaved --output-dir "$out" \
    --allow-foreign-bundle --skip-existing "$@" >> "$LOG" 2>&1 || fail "run $*"
}
run "$OUT" --policy lfu_decay --capacity 32
run "$OUT" --policy splitphase --capacity 32 "${SP[@]}"
run "$OUT" --policy global_token_gbt --capacity 32 "${SYNC[@]}"
run "$OUT" --policy resident --capacity 64
run "$OUT" --policy splitphase --capacity 24 "${SP[@]}"
run "$OUT" --policy lfu_decay --capacity 24
run "$OUT" --policy global_token_gbt --capacity 24 "${SYNC[@]}"
run "${OUT}_drift" --policy lfu_decay --capacity 32
step done
