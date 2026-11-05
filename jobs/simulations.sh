#!/usr/bin/env bash
# Trace simulations (CPU only): the stale-ranking results of Section 4 and Appendices A-B, the decode emulation of
# Section 5, and the simulated copy counts of T1, T2 and T3. Each writes into results/, so `git diff --stat results/`
# shows what changed. One OLMoE arm takes about 3 minutes; JOBS sets how many run at once.
#   bash jobs/simulations.sh staleness|stale_trained|refresh|regret|crossmodel|emulation|worker_slack|t1|t2|t3|all
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
JOBS=${JOBS:-6}
R=data/rankers
SIM=results/simulation
parallel() { tr '\n' '\0' | xargs -0 -n 1 -P "$JOBS" bash -c; }  # one command per line

staleness() {  # Figure 3 (fresh, stale and HALP-style arms) and the windows of Section 4
  mkdir -p $SIM/staleness
  for B in 32 24; do
    for ARM in "fresh none" "fresh oracle" "lfu none" "lfu fresh" "lfu oracle" "lag_req none" "lag_tok none" \
               "lag_req fresh" "lag_tok fresh" "lag_req oracle"; do
      set -- $ARM
      echo "python analysis/splitphase.py --capacity $B --source $1 --window $2 --output $SIM/staleness/${1}_${2}_B$B.json"
    done
  done | parallel
}

stale_trained() {  # Figure 3: rankers trained on stale features, the layer-0 rule, and the final 200-round configuration
  mkdir -p $SIM/refresh
  for B in 32 24; do
    echo "python analysis/splitphase.py --capacity $B --source lag_req --lag-bundle $R/olmoe_dev_lag_req_trees800.pkl --output $SIM/refresh/lag_req_native_B$B.json"
    echo "python analysis/splitphase.py --capacity $B --source lag_tok --lag-bundle $R/olmoe_dev_lag_tok_trees800.pkl --output $SIM/refresh/lag_tok_native_B$B.json"
    echo "python analysis/splitphase.py --capacity $B --source lag_req --mix-layer0 --lag-bundle $R/olmoe_dev_lag_req_trees800.pkl --lag-tok-bundle $R/olmoe_dev_lag_tok_trees800.pkl --no-diagnose --output $SIM/refresh/frozen_candidate_B$B.json"
    echo "python analysis/splitphase.py --capacity $B --source fresh --bundle $R/olmoe_dev_fresh_trees200.pkl --no-diagnose --output $SIM/refresh/fresh_trees200_B$B.json"
    echo "python analysis/splitphase.py --capacity $B --source lag_req --mix-layer0 --lag-bundle $R/olmoe_dev_lag_req_trees200.pkl --lag-tok-bundle $R/olmoe_dev_lag_tok_trees200.pkl --no-diagnose --output $SIM/refresh/frozen_candidate_trees200_B$B.json"
    echo "python analysis/splitphase.py --capacity $B --source fresh --bundle $R/olmoe_dev_fresh_trees800.pkl --no-diagnose --output $SIM/refresh/refit_fresh_B$B.json"
  done | parallel
}

refresh() {  # Figure A.1: refreshing the ranking every d tokens, with and without rescoring the touched experts
  mkdir -p $SIM/refresh
  for B in 32 24; do
    for D in 2 4 8 16; do
      echo "python analysis/splitphase.py --capacity $B --source lag_req --refresh $D --touch --no-diagnose --output $SIM/refresh/lag_req_d${D}_touch_B$B.json"
      echo "python analysis/splitphase.py --capacity $B --source lag_req --refresh $D --no-diagnose --output $SIM/refresh/lag_req_d${D}_none_B$B.json"
    done
    echo "python analysis/splitphase.py --capacity $B --source lag_req --refresh 1 --mix-layer0 --no-diagnose --output $SIM/refresh/lag_req_d1_mix0_B$B.json"
  done | parallel
}

regret() {  # Figure 4: where stale and fresh rankings disagree, which choice was better
  mkdir -p $SIM/regret
  {
    for B in 32 24; do
      echo "python analysis/splitphase.py --capacity $B --source lag_req --regret --output $SIM/regret/lag_req_regret_B$B.json"
    done
    echo "python analysis/splitphase.py --capacity 32 --source lag_tok --regret --output $SIM/regret/lag_tok_regret_B32.json"
  } | parallel
}

crossmodel() {  # Figure B.1: staleness on Qwen3 and Mixtral (development fold of each model's original trace)
  mkdir -p $SIM/crossmodel
  for SPEC in "qwen3_30b_a3b 48 64" "mixtral_8x7b 3 4"; do
    set -- $SPEC
    M=$1; shift
    for B in "$@"; do
      for SRC in lfu fresh lag_req lag_tok; do
        echo "python analysis/splitphase.py --prepared-root data/prepared/$M --bundle $R/${M}_dev_trees800.pkl --capacity $B --source $SRC --no-diagnose --output $SIM/crossmodel/${M}_${SRC}_B$B.json"
      done
    done
  done | parallel
}

emulation() {  # Section 5: critical-path time per token in a decode emulation (run alone: timing)
  mkdir -p $SIM/emulation
  P=(--p2 $R/olmoe_dev_lag_req_trees200.pkl --p1 $R/olmoe_dev_lag_tok_trees200.pkl)
  python analysis/emulate_decode.py --policy splitphase --capacity 32 --layer-ms 2.6 --workers 1 "${P[@]}" --output $SIM/emulation/splitphase_B32_l2.6.json
  python analysis/emulate_decode.py --policy lfu_decay --capacity 32 --layer-ms 2.6 --output $SIM/emulation/lfu_B32_l2.6.json
  python analysis/emulate_decode.py --policy splitphase --capacity 32 --layer-ms 1.7 --workers 1 "${P[@]}" --output $SIM/emulation/splitphase_B32_l1.7.json
  python analysis/emulate_decode.py --policy splitphase --capacity 24 --layer-ms 2.6 --workers 1 "${P[@]}" --output $SIM/emulation/splitphase_B24_l2.6.json
  python analysis/emulate_decode.py --policy sync --capacity 32 --layer-ms 2.6 --output $SIM/emulation/sync_B32_l2.6.json
}

worker_slack() {  # Section 5 and Appendix O: waits as the worker is slowed down (run alone: timing)
  mkdir -p $SIM/worker_slack
  for F in 1 2 4 6 8 12; do
    SPLITPHASE_SLOWDOWN=$F python analysis/emulate_decode.py --policy splitphase --slow-service --capacity 32 --layer-ms 2.6 \
      --workers 1 --limit 10 --p2 $R/olmoe_dev_lag_req_trees200.pkl --p1 $R/olmoe_dev_lag_tok_trees200.pkl \
      --output $SIM/worker_slack/slow_x$F.json
  done
}

t1() {  # T1: simulated copies on T1's trace (all ten workload paths)
  mkdir -p results/t1/sim
  for ARM in "lfu 32" "sync 32" "splitphase 32" "lfu 24" "sync 24" "splitphase 24"; do
    set -- $ARM
    echo "python analysis/unseen_sim.py --trace-root data/traces/olmoe_t1 --frozen-dir data/prepared/olmoe_t1 --arm $1 --capacity $2 --output results/t1/sim/sim_$1_B$2.json"
  done | parallel
}

t2() {  # T2: simulated copies on the 80 calibration sequences of T1's trace, which T2 timed
  mkdir -p results/t2/sim
  for ARM in "lfu 32" "splitphase 32" "lfu 24" "splitphase 24"; do
    set -- $ARM
    echo "python analysis/unseen_sim.py --trace-root data/traces/olmoe_t1 --frozen-dir data/prepared/olmoe_t1 --arm $1 --capacity $2 --workloads calibration --output results/t2/sim/sim_$1_B$2.json"
  done | parallel
}

t3() {  # T3: simulated copies on the new Qwen3 and Mixtral traces, in the GPU runs' sequence order
  mkdir -p results/t3/sim
  for SPEC in "qwen3_30b_a3b 64" "qwen3_30b_a3b 48" "mixtral_8x7b 4" "mixtral_8x7b 3"; do
    set -- $SPEC
    for ARM in lfu sync splitphase; do
      echo "python analysis/unseen_sim.py --trace-root data/traces/${1}_t3 --spec data/specs/${1}_t3_replay_spec.json --arm $ARM --capacity $2 --sync-bundle $R/${1}_dev_trees800.pkl --p2-bundle $R/${1}_frozen_lag_req_trees200.pkl --p1-bundle $R/${1}_frozen_lag_tok_trees200.pkl --output results/t3/sim/sim_${1}_${ARM}_B$2.json"
    done
  done | parallel
}

case "${1:-}" in
  staleness|stale_trained|refresh|regret|crossmodel|emulation|worker_slack|t1|t2|t3) "$1" ;;
  all) staleness; stale_trained; refresh; regret; crossmodel; t1; t2; t3; emulation; worker_slack ;;
  *) echo "usage: $0 staleness|stale_trained|refresh|regret|crossmodel|emulation|worker_slack|t1|t2|t3|all"; exit 2 ;;
esac
