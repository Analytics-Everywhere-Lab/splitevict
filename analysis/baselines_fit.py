"""Fit the published-system baselines on calibration data, as their papers do (T5).

Per model and capacity, HOBBIT's recency weight is the grid value (0, 0.05, ..., 1) with the fewest
transfers on the fitting sequences (one cold cache, in order; ties to the smaller weight), and
MoE-Infinity's store is the fitting sequences' request EAMs.

* ``dev``: fitted on calA (calibration sequences 0-39), for development runs on calB;
* ``frozen``: fitted on all 80 calibration sequences, for the sealed test on fresh traces (as the frozen
  rankers were).

Writes ``data/baselines/{model}_{dev,frozen}.json`` and ``..._eams.npy``.

    python analysis/baselines_fit.py --model olmoe --capacities 32 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src" / "runtime"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import splitphase  # noqa: E402
from baselines_dev import PREPARED, W_GRID, replay  # noqa: E402

OUT = ROOT / "data/baselines"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("olmoe", "qwen3", "mixtral"), required=True)
    parser.add_argument("--capacities", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.model != "olmoe":
        splitphase.PREPARED_ROOT = PREPARED[args.model].resolve()
    inputs, cal_a, _cal_b = splitphase.load_dev()
    trace = inputs.trace
    from policy_baselines import HobbitPolicy, request_eams

    L, E, k = int(trace.num_layers), int(trace.num_experts), int(trace.top_k)
    l2o = {layer: i for i, layer in enumerate(map(int, trace.metadata["layer_indices"]))}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fit in (("dev", cal_a), ("frozen", inputs.calibration)):
        ids = [int(s.source_sequence_id) for s, _v in fit.iter_slices(trace)]
        eams = request_eams(trace, fit, l2o)
        npy = OUT / f"{args.model}_{name}_eams.npy"
        np.save(npy, eams)
        entry = {"model": args.model, "fit": name, "fit_sequence_ids": ids, "trace_hash": trace.trace_hash,
                 "eams": {"path": str(npy.relative_to(ROOT)), "shape": list(eams.shape),
                          "sha256": hashlib.sha256(npy.read_bytes()).hexdigest()}, "hobbit": {}}
        for B in args.capacities:
            grid = {w: replay(trace, fit, HobbitPolicy(L, E, B, k, w=w)) for w in W_GRID}
            best = min(grid, key=lambda w: (grid[w], w))
            entry["hobbit"][str(B)] = {"w": best, "grid_transfers": grid}
            print(f"{args.model} {name} B={B}: hobbit w={best} ({grid[best]:,} transfers on {len(ids)} sequences)",
                  flush=True)
        (OUT / f"{args.model}_{name}.json").write_text(json.dumps(entry, indent=1) + "\n")


if __name__ == "__main__":
    main()
