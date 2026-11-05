#!/usr/bin/env bash
# Table 1: wall time, decision time and copies per token for each policy on OLMoE (B=32, ten evaluation paths).
# Every finished (policy, path) file is skipped, so the script resumes. About 6-10 hours on an A40.
#   bash jobs/a40_bottleneck.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/a40_bottleneck.log
source jobs/common.sh
SPEC=data/specs/olmoe_replay_spec.json
OUT=data/runs/a40
host_info
for POLICY in resident lru lfu_decay global_token_gbt caer_gbt; do
  CAP=32
  [ "$POLICY" = resident ] && CAP=64
  step "$POLICY B=$CAP"
  python -u src/runtime/offload_runtime.py --spec "$SPEC" --policy-bundle data/rankers/olmoe_setdev_policy.pkl \
    --policy "$POLICY" --capacity "$CAP" --output-dir "$OUT" --skip-existing >> "$LOG" 2>&1 || fail "$POLICY"
done
python src/runtime/analyze_hardware.py --runs "$OUT" | tee -a "$LOG"
step done
