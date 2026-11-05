#!/bin/bash
# T7 on one H100 node: choosing the policy on the host, with free-running decoding.
#   1. Calibration for both models first: decayed LFU and split-phase eviction, one pass, on the first 40 calibration
#      sequences of the model's T5 trace, at K = 1, 2, 4; after each run the selector's decision
#      (analysis/selector.py decide), then the decisions extrapolated from the K = 1 run. Every decision file's
#      sha256 goes into the log ("DECISIONS FROZEN") before any new trace exists.
#   2. Per model: a new trace (examples 285-304 of each domain), its replay spec, then both policies at K = 1, 2, 4.
# Steps skip finished output and decisions are never recomputed. Submit from the repository root:  sbatch jobs/t7.sbatch
#SBATCH --job-name=t7
#SBATCH --gpus=h100:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=t7-%j.out
set -uo pipefail
LOG=logs/t7.log
source jobs/common.sh
export CUDA_VISIBLE_DEVICES=0
OFFSET=285
PER=20
CALN=40
KS=(1 2 4)
R=data/rankers
OUT=data/runs/t7
COMMON=(--placement physical --monitor --lean --copy-api driver --warmup 1 --gpu-index 0 --passes 1 --arms lfu_decay splitphase)
OL=(--p2-bundle $R/olmoe_frozen_lag_req_trees200.pkl --p1-bundle $R/olmoe_frozen_lag_tok_trees200.pkl)
QW=(--experts-on-host --p2-bundle $R/qwen3_30b_a3b_frozen_lag_req_trees200.pkl --p1-bundle $R/qwen3_30b_a3b_frozen_lag_tok_trees200.pkl)
host_info
step tests
python -m unittest tests.test_hardware tests.test_baselines tests.test_prefetch_architectures tests.test_freerun -q >> "$LOG" 2>&1 || fail tests
step "copy issue cost"
python -u src/runtime/bench_copy_issue.py 2>&1 | tee -a "$LOG" || skip "copy benchmark"

# calibrate NAME CAP FLAGS...: calibration runs on T5's prompts and the selector's decisions
calibrate() {
  local NAME=$1 CAP=$2; shift 2
  local CALSPEC=data/specs/${NAME}_t5_replay_spec.json DIR=$OUT/$NAME SEQS
  mkdir -p "$DIR"
  SEQS=$(ids "$CALSPEC" $CALN) || { skip "$NAME calibration ids"; return 1; }
  for K in "${KS[@]}"; do
    if [ ! -f "$DIR/cal_B${CAP}_K$K.json" ]; then
      step "$NAME: calibration K=$K ($CALN sequences, 1 pass)"
      python -u src/runtime/interleaved_freerun.py --spec "$CALSPEC" --capacity "$CAP" --sequences "$SEQS" --copy-repeat "$K" \
        "${COMMON[@]}" "$@" --output "$DIR/cal_B${CAP}_K$K.json" --label "t7/$NAME/cal_B${CAP}_K$K" >> "$LOG" 2>&1 \
        || { skip "$NAME calibration K=$K"; return 1; }
    fi
    if [ ! -f "$DIR/decisions_B${CAP}_K$K.json" ]; then
      python analysis/selector.py decide "$DIR/cal_B${CAP}_K$K.json" --ks 1 --label "$NAME B$CAP K$K per host" \
        --output "$DIR/decisions_B${CAP}_K$K.json" 2>&1 | tee -a "$LOG"
      [ "${PIPESTATUS[0]}" -eq 0 ] || { skip "$NAME decision K=$K"; return 1; }
    fi
  done
  if [ ! -f "$DIR/decisions_B${CAP}_extrapolated.json" ]; then
    python analysis/selector.py decide "$DIR/cal_B${CAP}_K1.json" --ks "${KS[@]}" --label "$NAME B$CAP extrapolated" \
      --output "$DIR/decisions_B${CAP}_extrapolated.json" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { skip "$NAME extrapolated decisions"; return 1; }
  fi
}

# evaluate NAME KEY CAP GEN_B FLAGS...: new trace, replay spec, both policies at every K (only after the decisions)
evaluate() {
  local NAME=$1 KEY=$2 CAP=$3 GEN=$4; shift 4
  local TRACE=data/traces/${NAME}_t7 SPEC=data/specs/${NAME}_t7_replay_spec.json DIR=$OUT/$NAME CONFIG
  for K in "${KS[@]}"; do
    [ -f "$DIR/decisions_B${CAP}_K$K.json" ] || { skip "$NAME: no decision for K=$K, no evaluation"; return 1; }
  done
  [ -f "$DIR/decisions_B${CAP}_extrapolated.json" ] || { skip "$NAME: no extrapolated decisions, no evaluation"; return 1; }
  case $KEY in olmoe) CONFIG=measurement/stage3_residency/configs/full_preregistered.json ;; *) CONFIG=configs/trace_${KEY}.json ;; esac
  if [ ! -f "$TRACE/routing_trace.npz" ]; then
    step "$NAME: generate fresh trace (examples $OFFSET-$((OFFSET + PER - 1)) of each domain)"
    python -u src/runtime/fresh_trace_offloaded.py generate --opt-root measurement --base-config "$CONFIG" --offset $OFFSET \
      --prompts-per-domain $PER --capacity $GEN --output-dir "$TRACE" >> "$LOG" 2>&1 || { skip "$NAME generate"; return 1; }
  fi
  if [ ! -f "$SPEC" ]; then
    step "$NAME: replay spec"
    python -u src/runtime/fresh_trace_offloaded.py spec --opt-root measurement --trace-root "$TRACE" --output "$SPEC" \
      --original-metadata "${META_T1_T5[@]}" >> "$LOG" 2>&1 || { skip "$NAME spec"; return 1; }
  fi
  for K in "${KS[@]}"; do
    [ -f "$DIR/B${CAP}_K$K.json" ] && continue
    step "$NAME: arms B${CAP}_K$K, 1 pass, free-running"
    python -u src/runtime/interleaved_freerun.py --spec "$SPEC" --capacity "$CAP" --sequences calibration --copy-repeat "$K" \
      "${COMMON[@]}" "$@" --output "$DIR/B${CAP}_K$K.json" --label "t7/$NAME/B${CAP}_K$K" >> "$LOG" 2>&1 \
      || skip "$NAME arms K=$K"
  done
}

DECISIONS=($OUT/olmoe/decisions_B32_K{1,2,4}.json $OUT/olmoe/decisions_B32_extrapolated.json
           $OUT/qwen3/decisions_B64_K{1,2,4}.json $OUT/qwen3/decisions_B64_extrapolated.json)
frozen() { local f; for f in "${DECISIONS[@]}"; do [ -f "$f" ] || return 1; done; }
# no new trace and no evaluation output may exist before every decision is made
if ! frozen && { ls -d data/traces/*_t7 >/dev/null 2>&1 || ls "$OUT"/*/B*_K*.json >/dev/null 2>&1; }; then
  fail "a new trace or an evaluation output exists before every decision"
fi
calibrate olmoe 32 "${OL[@]}"
calibrate qwen3 64 "${QW[@]}"
frozen || fail "not every decision exists; resubmit to finish the calibration"
if ! grep -q "DECISIONS FROZEN" "$LOG"; then
  step "DECISIONS FROZEN"
  sha256sum "${DECISIONS[@]}" "$OUT"/*/cal_*.json | tee -a "$LOG"
fi
evaluate olmoe olmoe 32 32 "${OL[@]}"
evaluate qwen3 qwen3_30b_a3b 64 64 "${QW[@]}"
cp "$LOG" "$OUT/t7.log"
grep -q "^FAILED" "$LOG" && step "finished with failures (see FAILED lines)" || step done
