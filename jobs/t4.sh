#!/usr/bin/env bash
# T4: four arms on OLMoE (an A40), with and without the same next-layer prefetcher: decayed LFU, split-phase eviction,
# and each with prefetching (m = 8 guessed experts, up to q = 2 copies, variant g1n). New trace from examples
# 200-239 of each domain; two passes per capacity. Steps skip finished output.
#   bash jobs/t4.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/t4.log
source jobs/common.sh
TRACE=data/traces/olmoe_t4
SPEC=data/specs/olmoe_t4_replay_spec.json
OUT=data/runs/t4
mkdir -p "$OUT"
host_info
step tests
python -m unittest tests.test_hardware -q >> "$LOG" 2>&1 || fail tests
if [ ! -f "$TRACE/routing_trace.npz" ]; then
  step "generate trace (examples 200-239 of each domain)"
  python -u src/runtime/fresh_trace_offloaded.py generate --resident --opt-root measurement \
    --base-config measurement/stage3_residency/configs/full_preregistered.json --offset 200 \
    --prompts-per-domain 40 --output-dir "$TRACE" >> "$LOG" 2>&1 || fail generate
fi
if [ ! -f "$SPEC" ]; then
  step "replay spec"
  python -u src/runtime/fresh_trace_offloaded.py spec --opt-root measurement --trace-root "$TRACE" --output "$SPEC" \
    --original-metadata measurement/stage3_residency/traces/full/routing_trace.metadata.json \
    data/traces/olmoe_t1/routing_trace.metadata.json >> "$LOG" 2>&1 || fail spec
fi
for B in 32 24; do
  [ -f "$OUT/B$B.json" ] && continue
  step "arms B=$B, 2 passes"
  python -u src/runtime/interleaved_prefetch.py --spec "$SPEC" --capacity "$B" --sequences calibration \
    --p2-bundle data/rankers/olmoe_frozen_lag_req_trees200.pkl --p1-bundle data/rankers/olmoe_frozen_lag_tok_trees200.pkl \
    --placement physical --monitor --lean --copy-api driver --warmup 1 --passes 2 \
    --prefetch-m 8 --prefetch-q 2 --prefetch-variant g1n --lfu-wrong-rule last \
    --arms lfu_decay splitphase lfu_decay+pf splitphase+pf --output "$OUT/B$B.json" --label "t4/B$B" \
    >> "$LOG" 2>&1 || fail "arms B=$B"
done
step done
