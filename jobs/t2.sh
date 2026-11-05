#!/usr/bin/env bash
# T2: decayed LFU and split-phase eviction interleaved sequence by sequence in one process, on the 80 sequences of
# T1's trace that T1 did not time (an A40).
#   bash jobs/t2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/t2.log
source jobs/common.sh
host_info
step tests
python -m unittest tests.test_hardware -q >> "$LOG" 2>&1 || fail tests
mkdir -p data/runs/t2
for B in 32 24; do
  OUT=data/runs/t2/B$B.json
  [ -f "$OUT" ] && continue
  step "interleaved B=$B"
  python -u src/runtime/interleaved_runtime.py --spec data/specs/olmoe_t1_replay_spec.json --capacity $B \
    --sequences calibration --p2-bundle data/rankers/olmoe_frozen_lag_req_trees200.pkl \
    --p1-bundle data/rankers/olmoe_frozen_lag_tok_trees200.pkl --output "$OUT" --label "t2/B$B" >> "$LOG" 2>&1 \
    || fail "interleaved B=$B"
done
step done
