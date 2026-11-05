from __future__ import annotations
import numpy as np
from pathlib import Path


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[1] / 'measurement'
    if (root / 'stage3_residency').exists():
        return root
    raise SystemExit('Could not locate the measurement codebase')


from race_stage1.models import TransitionModels
from race_stage1.frozen import _validate_split, workload_from_record
from residency_headroom.common import read_json, sha256_file
from residency_headroom.trace import RoutingTrace
from residency_headroom.simulator import simulate_oracle, simulate_policy
from residency_headroom.workloads import calibration_frequency_scores

ROOT = _repository_root()
PREREG = ROOT / 'stage3_residency/stage2_race/configs/stage2_preregistered.json'


class Inputs:
    def __init__(self, trace, calibration, workloads):
        self.trace, self.calibration, self.workloads = trace, calibration, workloads


def load():
    ref = read_json(PREREG)["stage0_reference"]
    frozen_path = ROOT / ref["frozen_evaluation_config_path"]
    if sha256_file(frozen_path) != ref["frozen_evaluation_config_file_sha256"]:
        raise ValueError("frozen evaluation split changed")
    trace = RoutingTrace.load(ROOT / ref["trace_path"], validate=True)
    if trace.trace_hash != ref["trace_logical_hash"]:
        raise ValueError("routing trace changed")
    frozen = read_json(frozen_path)
    calibration = workload_from_record(frozen["calibration_workload"])
    workloads = tuple(workload_from_record(item) for item in frozen["workloads"])
    if {w.name: w.hash for w in workloads} != frozen["workload_hashes"]:
        raise ValueError("workload hashes changed")
    _validate_split(frozen, calibration, workloads)
    models = TransitionModels.load(
        ROOT / 'stage3_residency/stage2_race/results/calibration/transition_models.npz')
    if (models.trace_hash, models.calibration_workload_hash) != (trace.trace_hash, calibration.hash):
        raise ValueError("transition models were not fitted on this calibration split")
    return Inputs(trace, calibration, workloads), models


def references(inputs, workload, capacities):
    out = {}
    out['oracle'] = {c: int(simulate_oracle(inputs.trace, workload, c).misses) for c in capacities}
    static = calibration_frequency_scores(inputs.trace, inputs.calibration).astype(np.float64)
    simple = {}
    for name, kw in (('lru', {}), ('lfu', {}), ('lfu_decay', {'alpha': 0.95}),
                     ('static_hotset', {'static_scores': static})):
        simple[name] = {c: int(simulate_policy(inputs.trace, workload, c, name, **kw).misses)
                        for c in capacities}
    out['simple_all'] = simple
    out['simple'] = {c: min(simple[n][c] for n in simple) for c in capacities}
    out['simple_which'] = {c: min(simple, key=lambda n: simple[n][c]) for c in capacities}
    return out


