#!/bin/bash
# T5 development on one H100 node, on the original traces' calibration sequences only. Measures the costs that the
# break-even model needs before the test (copy time, issue time, decision time, remainder), then
# `python analysis/analyze_t5_dev.py` writes results/t5/dev_costs.json and results/t5/dev_summary.json.
# Submit from the repository root:  sbatch jobs/t5_dev.sbatch
#SBATCH --job-name=t5dev
#SBATCH --gpus=h100:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=t5dev-%j.out
set -uo pipefail
LOG=logs/t5_dev.log
source jobs/common.sh
export CUDA_VISIBLE_DEVICES=0
OUT=data/runs/t5_dev
R=data/rankers
FIT=data/baselines
mkdir -p "$OUT"
host_info
step tests
python -m unittest tests.test_hardware tests.test_baselines -q >> "$LOG" 2>&1 || fail tests
step "copy issue cost"
python -u src/runtime/bench_copy_issue.py 2>&1 | tee -a "$LOG" || fail "copy benchmark"
COMMON=(--placement physical --monitor --lean --copy-api driver --warmup 1 --passes 1 --gpu-index 0
        --prefetch-m 8 --prefetch-q 2 --prefetch-variant g1n)
ALL=(lfu_decay splitphase global_token_gbt hobbit moe_infinity)

OSPEC=data/specs/olmoe_replay_spec.json
for CAP in 32 24; do
  [ -f "$OUT/olmoe_B$CAP.json" ] && continue
  step "OLMoE B=$CAP"
  python -u src/runtime/interleaved_h100.py --spec $OSPEC --sequences "$(ids $OSPEC 80 40)" --capacity $CAP "${COMMON[@]}" \
    --p2-bundle $R/olmoe_dev_lag_req_trees200.pkl --p1-bundle $R/olmoe_dev_lag_tok_trees200.pkl \
    --sync-bundle $R/olmoe_setdev_policy.pkl --hobbit-w "$(hobbit_w olmoe dev $CAP)" \
    --moe-infinity-store $FIT/olmoe_dev_eams.npy --arms "${ALL[@]}" lfu_decay+pf splitphase+pf hobbit+pf \
    --output "$OUT/olmoe_B$CAP.json" --label "t5dev/olmoe/B$CAP" >> "$LOG" 2>&1 || skip "OLMoE B=$CAP"
done
QSPEC=data/specs/qwen3_30b_a3b_original_calibration_spec.json
for CAP in 64 48; do
  [ -f "$OUT/qwen3_B$CAP.json" ] && continue
  step "Qwen3 B=$CAP"
  python -u src/runtime/interleaved_h100.py --spec $QSPEC --sequences "$(ids $QSPEC 60 40)" --capacity $CAP "${COMMON[@]}" \
    --experts-on-host --p2-bundle $R/qwen3_30b_a3b_frozen_lag_req_trees200.pkl \
    --p1-bundle $R/qwen3_30b_a3b_frozen_lag_tok_trees200.pkl --sync-bundle $R/qwen3_30b_a3b_dev_trees800.pkl \
    --hobbit-w "$(hobbit_w qwen3 dev $CAP)" --moe-infinity-store $FIT/qwen3_dev_eams.npy \
    --arms "${ALL[@]}" lfu_decay+pf splitphase+pf hobbit+pf \
    --output "$OUT/qwen3_B$CAP.json" --label "t5dev/qwen3/B$CAP" >> "$LOG" 2>&1 || skip "Qwen3 B=$CAP"
done
# Mixtral without prefetching. For Qwen3 and Mixtral these sequences are inside the rankers' fit, so these runs give
# timings and costs only, never copy counts.
MSPEC=data/specs/mixtral_8x7b_original_calibration_spec.json
for CAP in 4 3; do
  [ -f "$OUT/mixtral_B$CAP.json" ] && continue
  step "Mixtral B=$CAP"
  python -u src/runtime/interleaved_h100.py --spec $MSPEC --sequences "$(ids $MSPEC 50 40)" --capacity $CAP "${COMMON[@]}" \
    --experts-on-host --p2-bundle $R/mixtral_8x7b_frozen_lag_req_trees200.pkl \
    --p1-bundle $R/mixtral_8x7b_frozen_lag_tok_trees200.pkl --sync-bundle $R/mixtral_8x7b_dev_trees800.pkl \
    --hobbit-w "$(hobbit_w mixtral dev $CAP)" --moe-infinity-store $FIT/mixtral_dev_eams.npy --arms "${ALL[@]}" \
    --output "$OUT/mixtral_B$CAP.json" --label "t5dev/mixtral/B$CAP" >> "$LOG" 2>&1 || skip "Mixtral B=$CAP"
done
# copy-cost sweep: every expert copy issued K times into its slot
for K in 4 16; do
  [ -f "$OUT/olmoe_B32_K$K.json" ] && continue
  step "copy-cost sweep K=$K"
  python -u src/runtime/interleaved_h100.py --spec $OSPEC --sequences "$(ids $OSPEC 60 40)" --capacity 32 "${COMMON[@]}" \
    --copy-repeat $K --p2-bundle $R/olmoe_dev_lag_req_trees200.pkl --p1-bundle $R/olmoe_dev_lag_tok_trees200.pkl \
    --sync-bundle $R/olmoe_setdev_policy.pkl --hobbit-w "$(hobbit_w olmoe dev 32)" \
    --arms lfu_decay splitphase global_token_gbt hobbit \
    --output "$OUT/olmoe_B32_K$K.json" --label "t5dev/olmoe/B32/K$K" >> "$LOG" 2>&1 || skip "copy-cost sweep K=$K"
done
step done
