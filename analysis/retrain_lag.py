"""Train the learned ranker on the stale (lagged) features it sees under split-phase eviction (P2, P1).

The collector follows ``run4.simulate_multi`` (the OLMoE row collector ``run_caer.fit_global_models``
uses): dynamics under the round-1 seed scorer on fresh features, stride 3, warm-up 40, groups
with at least four candidates, target = capped next-use distance. At every sampled decision it
stores three versions of the same candidate rows:

  fresh   : the block the synchronous ranker sees (its row count must equal the development fit's)
  lag_req : state through p-1, request R_{p-1}, current token          (P2)
  lag_tok : lag_req with the five token rows of event p-1's block        (P1)

and fits one ranker per (version, capacity) with the development hyperparameters. Bundles are written
with the global models replaced and everything else (tables, stage-2 models) copied from the base bundle.

``--prepared-root`` runs the same recipe on Qwen3-30B-A3B or Mixtral-8x7B through the
``crossmodel`` bindings (their feature state, seed scorer and calibration split). A group is any
decision with at least ``min(4, spare + 1)`` candidates, the ``crossmodel`` rule; on OLMoE
(spare >= 16) that is ``>= 4``.

    python analysis/retrain_lag.py --fit calibration --versions lag_req lag_tok --max-iter 200
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OPT = Path(__file__).resolve().parents[1] / "measurement"
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "src" / "runtime"))

TOKEN_ROWS = slice(45, 50)
CAPACITIES = (24, 32)


def collect(stride: int = 3, warmup: int = 40, fit: str = "calA", prepared_root=None,
            capacities=CAPACITIES, versions=("fresh", "lag_req", "lag_tok")):
    import run_caer

    run_caer.configure_repository(OPT)
    from collect2 import sub
    from race_stage1.models import same_layer_indices
    from race_stage2.diagnostics import build_future_occurrences
    from residency_headroom.workloads import calibration_frequency_scores
    from run8 import fit_token_tables, layer_streams

    load, _refs, seed_weights, _sim, state_base = run_caer.harness_bindings(prepared_root)
    inputs, transition_models = load()
    trace = inputs.trace
    calA = sub(inputs.calibration, 0, 40, "calA") if fit == "calA" else inputs.calibration
    print(f"fit workload {calA.name}: {len(calA.sequences)} sequences", flush=True)
    static = calibration_frequency_scores(trace, calA).astype(np.float64)
    token_index, token_current, token_next = fit_token_tables(
        trace, calA, trace.num_layers, trace.num_experts, static
    )
    state_type = run_caer.causal_token_state_class(state_base(trace) if state_base is not None else None)
    factory = run_caer.make_state_factory(
        state_type, transition_models, trace, static, token_index, token_current, token_next, layer_streams
    )
    state = factory(calA)
    w0 = seed_weights()

    L, E = trace.num_layers, trace.num_experts
    l2o = {layer: i for i, layer in enumerate(map(int, trace.metadata["layer_indices"]))}
    CAPS = tuple(capacities)
    req_all = trace.requested_expert_ids.astype(np.int64)
    srt = np.sort(req_all, axis=1)
    gate_all = trace.router_weights.astype(np.float64)
    res = np.zeros((len(CAPS), L, E), bool)
    lu = np.full((L, E), -1, np.int64)
    pos = np.zeros(L, np.int64)
    mask = np.zeros(E, bool)
    spares = [c - trace.top_k for c in CAPS]
    min_group = [min(4, sp + 1) for sp in spares]
    fut = build_future_occurrences(trace, calA, same_layer_indices(trace, calA))
    buckets = {v: [[[], [], [], 0] for _ in CAPS] for v in versions}
    prev_req = [None] * L
    prev_gates = [None] * L
    prev_tok = [None] * L
    state.reset()
    n = 0
    start = time.perf_counter()
    for _s, view in calA.iter_slices(trace):
        state.begin_sequence()
        for idx in range(view.start, view.stop):
            o = l2o[int(trace.layer_index[idx])]
            r = req_all[idx]
            p = int(pos[o])
            td = fut.advance(o, p, r)
            f = state.features(o, r, gate_all[idx], srt[idx], p)
            seed = w0 @ f[:45]
            sample = (n % stride == 0) and p >= warmup
            lag_req = lag_tok = None
            if sample and prev_req[o] is not None:
                lag_req = state.features(o, prev_req[o], prev_gates[o], np.sort(prev_req[o]), p)
                lag_tok = lag_req.copy()
                lag_tok[TOKEN_ROWS] = prev_tok[o]
            state.absorb(o, r, gate_all[idx], p)
            mask[r] = True
            lu[o, r] = p
            for ci, sp in enumerate(spares):
                row = res[ci, o]
                cand = np.flatnonzero(row & ~mask)
                if cand.size > sp:
                    if sample and cand.size >= min_group[ci]:
                        if lag_req is None:
                            raise RuntimeError("sampled decision without a previous request")
                        target = np.minimum(td[cand], 33).astype(np.float32)
                        for version, block in (("fresh", f), ("lag_req", lag_req), ("lag_tok", lag_tok)):
                            if version not in buckets:
                                continue
                            b = buckets[version][ci]
                            b[0].append(block[:, cand].T.astype(np.float32))
                            b[1].append(target)
                            b[2].append(np.full(cand.size, b[3], np.int32))
                            b[3] += 1
                    s = seed[cand]
                    order = np.lexsort((cand, -lu[o, cand], -s))
                    row.fill(False)
                    row[r] = True
                    row[cand[order[:sp]]] = True
                else:
                    row[r] = True
            mask[r] = False
            prev_req[o] = r
            prev_gates[o] = gate_all[idx]
            prev_tok[o] = np.array(f[TOKEN_ROWS])
            pos[o] += 1
            n += 1
    fut.finish()
    print(f"collection {time.perf_counter() - start:.0f}s", flush=True)
    return buckets


def fit_one(X, Y, G, max_iter: int = 800):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from run_caer import rank_target

    model = HistGradientBoostingRegressor(
        max_leaf_nodes=127, max_iter=max_iter, learning_rate=0.06, l2_regularization=1.0,
        early_stopping=False, min_samples_leaf=40, random_state=0,
    )
    model.fit(X, rank_target(Y, G))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bundle", default=str(ROOT / "data/rankers/olmoe_setdev_dev_trees800.pkl"))
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "data/rankers"))
    parser.add_argument("--versions", nargs="+", default=["fresh", "lag_req", "lag_tok"])
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--fit", default="calA", choices=("calA", "calibration"),
                        help="calA for development; calibration (all 80 sequences) for the frozen rankers")
    parser.add_argument("--prepared-root", help="Qwen3 / Mixtral: the crossmodel prepared root")
    parser.add_argument("--capacities", type=int, nargs="+", default=list(CAPACITIES))
    parser.add_argument("--stem", default="olmoe", help="bundle name prefix")
    args = parser.parse_args()
    buckets = collect(fit=args.fit, prepared_root=Path(args.prepared_root).resolve() if args.prepared_root else None,
                      capacities=args.capacities, versions=tuple(args.versions))
    from policy import PolicyBundle

    base = PolicyBundle.load(args.base_bundle)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for version in args.versions:
        payload = copy.copy(base.payload)
        payload["global_models"] = dict(base.payload["global_models"])
        for ci, cap in enumerate(args.capacities):
            b = buckets[version][ci]
            X = np.concatenate(b[0]).astype(np.float64)
            Y = np.concatenate(b[1]).astype(np.float64)
            G = np.concatenate(b[2])
            t0 = time.perf_counter()
            model = fit_one(X, Y, G, max_iter=args.max_iter)
            payload["global_models"][cap] = {"inner": model, "sign": -1.0, "target": "rank"}
            report[f"{version}_B{cap}"] = {"rows": int(X.shape[0]), "groups": int(b[3]),
                                           "fit_seconds": time.perf_counter() - t0}
            print(f"fitted {version} B={cap}: {X.shape[0]:,} rows [{time.perf_counter() - t0:.0f}s]", flush=True)
        payload["lag_training"] = {"version": version, "rows": {str(c): report[f"{version}_B{c}"]["rows"] for c in args.capacities},
                                   "fit": args.fit, "max_iter": args.max_iter}
        stem = f"{args.stem}_dev" if args.fit == "calA" else f"{args.stem}_frozen"
        path = out_dir / f"{stem}_{version}_trees{args.max_iter}.pkl"
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"wrote {path}", flush=True)
    name = f"retrain_report_{args.fit}_trees{args.max_iter}.json" if args.stem == "olmoe" else f"retrain_report_{args.stem}_{args.fit}_trees{args.max_iter}.json"
    Path(out_dir / name).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
