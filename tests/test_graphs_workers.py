"""T10 checks: split-phase with two worker processes, and the runner's worker placement.

* The running service with two workers (worker j owns the layers l with l % 2 == j) makes the simulator's decisions
  exactly, as the one-worker service does (``test_hardware``).
* Every split-phase arm gets its own group of worker cores, none shared with the decode process; the runner hands the
  arm the first core of its group.
* The runner needs ``--splitphase-workers``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/runtime", "src", "analysis"):
    value = str(ROOT / relative)
    if value not in sys.path:
        sys.path.append(value)

try:
    import numba  # noqa: F401
    import sklearn  # noqa: F401

    HAVE_SCORERS = True
except ImportError:  # pragma: no cover
    HAVE_SCORERS = False
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

OPT = Path(__file__).resolve().parents[1] / "measurement"
BUNDLES = ROOT / "data" / "rankers"


@unittest.skipUnless(HAVE_SCORERS and (OPT / "stage3_residency").exists()
                     and (BUNDLES / "olmoe_dev_lag_req_trees200.pkl").exists(),
                     "needs the frozen trace repository and the development split-phase bundles")
class TwoWorkerServiceTests(unittest.TestCase):
    def test_two_workers_make_the_simulators_decisions(self) -> None:
        sys.path.insert(0, str(ROOT / "analysis"))
        from splitphase import SplitPhase, load_bundle, load_dev
        from splitphase_service import SplitPhasePolicy, SplitPhaseService

        inputs, _fit, deploy = load_dev()
        trace = inputs.trace
        p2 = str(BUNDLES / "olmoe_dev_lag_req_trees200.pkl")
        p1 = str(BUNDLES / "olmoe_dev_lag_tok_trees200.pkl")
        L, E, k = trace.num_layers, trace.num_experts, trace.top_k
        sim = SplitPhase(load_bundle(p2), 32, "lag_req", "none", diagnose=False, mix_layer0=True,
                         lag_bundle=load_bundle(p2), lag_tok_bundle=load_bundle(p1))
        service = SplitPhaseService(p2, p1, 32, workers=2, num_layers=L, num_experts=E, top_k=k)
        try:
            live = SplitPhasePolicy(service, L, E, 32, k)
            layers = tuple(map(int, trace.metadata["layer_indices"]))
            l2o = {layer: i for i, layer in enumerate(layers)}
            requests = trace.requested_expert_ids.astype(np.int64, copy=False)
            gates = trace.router_weights.astype(np.float64, copy=False)
            for count, (_sequence, view) in enumerate(deploy.iter_slices(trace)):
                if count >= 2:
                    break
                sim.begin_sequence()
                live.begin_sequence()
                for base in range(view.start, view.stop, L):
                    token = int(trace.token_id[base])
                    live.begin_token(token)
                    for index in range(base, base + L):
                        layer = l2o[int(trace.layer_index[index])]
                        sim.step(layer, token, requests[index], gates[index])
                        live.step(layer, token, requests[index], gates[index])
                        self.assertTrue(np.array_equal(sim.resident[layer], live.resident[layer]),
                                        f"resident sets diverge at event {index}")
            self.assertEqual(sim.transfers, live.transfers)
            self.assertGreater(live.eviction_decisions, 0)
        finally:
            busy = service.close()
        self.assertEqual(len(busy), 2)
        self.assertTrue(all(b > 0 for b in busy))


@unittest.skipUnless(HAVE_TORCH, "torch/transformers missing")
class RunnerTests(unittest.TestCase):
    def test_every_arm_gets_its_own_group_of_worker_cores(self) -> None:
        import interleaved_graphs_workers
        import interleaved_h100

        def plan(workers, gpu_index=0):
            cores = [[c] for c in range(0, 2 * (workers + 3), 2)]
            return {"cores": cores, "workers": [c[0] for c in cores[:workers]],
                    "main": [c[0] for c in cores[workers:]]}

        saved = interleaved_h100.plan_placement
        interleaved_h100.plan_placement = plan
        try:
            placement = interleaved_graphs_workers.placement_for(2)(2)
        finally:
            interleaved_h100.plan_placement = saved
        self.assertEqual(placement["worker_groups"], [[0, 2], [4, 6]])
        self.assertEqual(placement["workers"], [0, 4])
        self.assertEqual(interleaved_graphs_workers.GROUPS[4], [4, 6])
        self.assertFalse(set(placement["main"]) & {0, 2, 4, 6})

    def test_runner_needs_the_worker_count(self) -> None:
        import interleaved_graphs_workers

        with self.assertRaises(SystemExit):
            interleaved_graphs_workers.main(["--output", "x.json"])


if __name__ == "__main__":
    unittest.main()
