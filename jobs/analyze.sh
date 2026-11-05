#!/usr/bin/env bash
# Recompute every test summary, the break-even tables and the paper's data figures from the released runs and traces
# (CPU only, about 45 minutes). Outputs go to results/ and figures/; `git diff --stat` shows what changed.
#   bash jobs/analyze.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg
run() { echo "== $*"; "$@"; }
run python src/runtime/analyze_hardware.py --runs data/runs/a40 --output results/a40/summary.json
for t in t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t10_b24 t11; do run python analysis/analyze_$t.py; done
run python analysis/report_t5.py
run python analysis/remainder_split.py
run python analysis/selector.py t5
run python analysis/selector.py t6-evaluate
run python analysis/breakeven_crossmodel.py
run python analysis/breakeven_h100.py
run python analysis/report_rules.py
run python src/runtime/summarize_overhead.py
run python analysis/figures.py
run python analysis/figures_appendix.py
