#!/usr/bin/env bash
# T3: Qwen3-30B-A3B on an A40 and Mixtral-8x7B on an A100 (PCIe gen3 host), three policies interleaved.
# Steps skip finished output, so the script resumes.
#   bash jobs/t3.sh qwen3      # or: bash jobs/t3.sh mixtral
set -uo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  qwen3)   M=qwen3_30b_a3b; GEN_B=64; CAPS="64 48" ;;
  mixtral) M=mixtral_8x7b; GEN_B=5; CAPS="4 3" ;;
  *) echo "usage: $0 qwen3|mixtral"; exit 2 ;;
esac
LOG=logs/t3_$M.log
source jobs/common.sh
TRACE=data/traces/${M}_t3
SPEC=data/specs/${M}_t3_replay_spec.json
OUT=data/runs/t3_$M
R=data/rankers
mkdir -p "$OUT"
host_info
step tests
python -m unittest tests.test_hardware -q >> "$LOG" 2>&1 || fail tests

# Worker count (Qwen3 only), decided before the new trace exists: one worker passes if its waits are at most 0.5% of
# decayed LFU's time per token on 8 calibration sequences of the original trace; otherwise two workers are tried.
WORKERS=1
if [ "$M" = qwen3_30b_a3b ]; then
  pilot() {
    [ -f "$OUT/pilot_w$1.json" ] || python -u src/runtime/interleaved_multi.py \
      --spec data/specs/qwen3_30b_a3b_original_calibration_spec.json --capacity 64 --experts-on-host --limit 8 \
      --policies lfu_decay splitphase --workers $1 --p2-bundle $R/${M}_frozen_lag_req_trees200.pkl \
      --p1-bundle $R/${M}_frozen_lag_tok_trees200.pkl --output "$OUT/pilot_w$1.json" --label "pilot/w$1" >> "$LOG" 2>&1 \
      || fail "pilot w$1"
    python - "$OUT/pilot_w$1.json" <<'PY' | tee -a "$LOG"
import json, sys
r = json.load(open(sys.argv[1]))
wait, lfu = r["splitphase"]["wait_ms_per_token"], r["policies"]["lfu_decay"]["totals"]["ms_per_token_wall"]
print(f"pilot {sys.argv[1]}: waits {wait:.3f} ms/token, limit {0.005 * lfu:.3f}")
sys.exit(0 if wait <= 0.005 * lfu else 1)
PY
  }
  if ! pilot 1; then pilot 2 && WORKERS=2; fi
fi
echo "{\"workers\": $WORKERS}" > "$OUT/workers.json"
step "workers = $WORKERS"

if [ ! -f "$TRACE/routing_trace.npz" ]; then
  step "generate trace (20 prompts per domain, offset 100)"
  python -u src/runtime/fresh_trace_offloaded.py generate --opt-root measurement --base-config configs/trace_${M}.json \
    --prompts-per-domain 20 --capacity $GEN_B --output-dir "$TRACE" >> "$LOG" 2>&1 || fail generate
fi
if [ ! -f "$SPEC" ]; then
  step "replay spec"
  python -u src/runtime/fresh_trace_offloaded.py spec --opt-root measurement --trace-root "$TRACE" --output "$SPEC" \
    --original-metadata data/traces/${M}/routing_trace.metadata.json >> "$LOG" 2>&1 || fail spec
fi
for B in $CAPS; do
  [ -f "$OUT/B$B.json" ] && continue
  step "interleaved B=$B"
  python -u src/runtime/interleaved_multi.py --spec "$SPEC" --capacity $B --experts-on-host \
    --policies lfu_decay splitphase global_token_gbt --workers $WORKERS --sync-bundle $R/${M}_dev_trees800.pkl \
    --p2-bundle $R/${M}_frozen_lag_req_trees200.pkl --p1-bundle $R/${M}_frozen_lag_tok_trees200.pkl \
    --output "$OUT/B$B.json" --label "t3/$M/B$B" >> "$LOG" 2>&1 || fail "interleaved B=$B"
done
step done
