from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .common import sha256_json
from .oracle import FarthestFutureOracle
from .policies import AtomicCachePolicy, make_policy
from .trace import RoutingTrace
from .workloads import Workload


@dataclass(frozen=True)
class SequenceMetrics:
    sequence_position: int
    source_sequence_id: int
    domain: str
    segment_index: int
    events: int
    requests: int
    hits: int
    misses: int
    admissions: int
    evictions: int
    bytes_requested: int
    bytes_transferred: int
    admission_bytes: int


@dataclass(frozen=True)
class SimulationResult:
    policy: str
    capacity: int
    workload: str
    regime: str
    seed: int | None
    alpha: float | None
    events: int
    requests: int
    hits: int
    misses: int
    admissions: int
    evictions: int
    bytes_requested: int
    bytes_transferred: int
    admission_bytes: int
    maximum_occupancy: int
    per_sequence: tuple[SequenceMetrics, ...]

    @property
    def miss_rate(self) -> float:
        return self.misses / self.requests if self.requests else float("nan")

    @property
    def cache_churn(self) -> int:
        return self.admissions + self.evictions

    @property
    def condition_id(self) -> str:
        return sha256_json(
            {
                "policy": self.policy,
                "capacity": self.capacity,
                "workload": self.workload,
                "seed": self.seed,
                "alpha": self.alpha,
            }
        )[:20]

    def base_record(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("per_sequence")
        value["miss_rate"] = self.miss_rate
        value["cache_churn"] = self.cache_churn
        value["condition_id"] = self.condition_id
        value["expert_transfers"] = self.misses
        return value


def expert_byte_matrix(trace: RoutingTrace) -> np.ndarray:
    raw = np.asarray(trace.metadata["expert_bytes_by_layer"], dtype=np.int64)
    if raw.shape == (trace.num_layers,):
        raw = np.broadcast_to(raw[:, None], (trace.num_layers, trace.num_experts)).copy()
    if raw.shape != (trace.num_layers, trace.num_experts):
        raise ValueError(
            f"expert_bytes_by_layer has shape {raw.shape}; expected "
            f"{(trace.num_layers,)} or {(trace.num_layers, trace.num_experts)}"
        )
    if np.any(raw <= 0):
        raise ValueError("Expert byte sizes must be positive")
    return raw


def simulate_policy(
    trace: RoutingTrace,
    workload: Workload,
    capacity: int,
    policy: str,
    *,
    alpha: float | None = None,
    seed: int | None = None,
    static_scores: np.ndarray | None = None,
) -> SimulationResult:
    layers = list(map(int, trace.metadata["layer_indices"]))
    layer_to_ordinal = {layer: ordinal for ordinal, layer in enumerate(layers)}
    if static_scores is not None:
        static_scores = np.asarray(static_scores)
        if static_scores.shape != (trace.num_layers, trace.num_experts):
            raise ValueError("Static score matrix does not match trace architecture")
    policies: dict[int, AtomicCachePolicy] = {}
    for layer in layers:
        ordinal = layer_to_ordinal[layer]
        policies[layer] = make_policy(
            policy,
            capacity,
            trace.num_experts,
            alpha=alpha,
            seed=seed,
            layer=layer,
            static_scores=static_scores[ordinal] if static_scores is not None else None,
        )
    return _simulate(
        trace,
        workload,
        capacity,
        policy,
        policies,
        layer_to_ordinal,
        alpha=alpha,
        seed=seed,
    )


def simulate_oracle(
    trace: RoutingTrace, workload: Workload, capacity: int
) -> SimulationResult:
    layers = list(map(int, trace.metadata["layer_indices"]))
    layer_to_ordinal = {layer: ordinal for ordinal, layer in enumerate(layers)}
    requests: dict[int, list[frozenset[int]]] = {layer: [] for layer in layers}
    for _sequence, view in workload.iter_slices(trace):
        for index in range(view.start, view.stop):
            layer = int(trace.layer_index[index])
            requests[layer].append(frozenset(map(int, trace.requested_expert_ids[index])))
    oracles = {
        layer: FarthestFutureOracle(requests[layer], capacity, trace.num_experts)
        for layer in layers
    }
    result = _simulate(
        trace,
        workload,
        capacity,
        "oracle",
        oracles,
        layer_to_ordinal,
        alpha=None,
        seed=None,
    )
    for oracle in oracles.values():
        oracle.finish()
    return result


def _simulate(
    trace: RoutingTrace,
    workload: Workload,
    capacity: int,
    policy_name: str,
    policies: Mapping[int, Any],
    layer_to_ordinal: Mapping[int, int],
    *,
    alpha: float | None,
    seed: int | None,
) -> SimulationResult:
    byte_matrix = expert_byte_matrix(trace)
    sequence_results: list[SequenceMetrics] = []
    maximum_occupancy = 0
    totals = np.zeros(9, dtype=np.int64)
    # events, requests, hits, misses, admissions, evictions,
    # requested bytes, transfer bytes, admission bytes
    for sequence, view in workload.iter_slices(trace):
        values = np.zeros(9, dtype=np.int64)
        for index in range(view.start, view.stop):
            layer = int(trace.layer_index[index])
            ordinal = layer_to_ordinal[layer]
            request_array = trace.requested_expert_ids[index].astype(np.int64, copy=False)
            request = frozenset(map(int, request_array))
            if len(request) != trace.top_k:
                raise RuntimeError("Atomic request lost a duplicate during simulation")
            transition = policies[layer].process(request)
            missing = np.fromiter(sorted(transition.misses), dtype=np.int64)
            admitted = np.fromiter(sorted(transition.admissions), dtype=np.int64)
            requested_bytes = int(byte_matrix[ordinal, request_array].sum(dtype=np.int64))
            transfer_bytes = (
                int(byte_matrix[ordinal, missing].sum(dtype=np.int64))
                if missing.size
                else 0
            )
            admission_bytes = (
                int(byte_matrix[ordinal, admitted].sum(dtype=np.int64))
                if admitted.size
                else 0
            )
            event = np.asarray(
                [
                    1,
                    len(request),
                    len(transition.hits),
                    len(transition.misses),
                    len(transition.admissions),
                    len(transition.evictions),
                    requested_bytes,
                    transfer_bytes,
                    admission_bytes,
                ],
                dtype=np.int64,
            )
            values += event
            maximum_occupancy = max(maximum_occupancy, len(transition.after))
            if len(transition.hits) + len(transition.misses) != len(request):
                raise RuntimeError("Atomic event hit/miss accounting failed")
            if capacity > 0 and transition.admissions != transition.misses:
                raise RuntimeError("Mandatory admission no longer equals misses")
        totals += values
        sequence_results.append(
            SequenceMetrics(
                sequence_position=sequence.position,
                source_sequence_id=sequence.source_sequence_id,
                domain=sequence.domain,
                segment_index=sequence.segment_index,
                events=int(values[0]),
                requests=int(values[1]),
                hits=int(values[2]),
                misses=int(values[3]),
                admissions=int(values[4]),
                evictions=int(values[5]),
                bytes_requested=int(values[6]),
                bytes_transferred=int(values[7]),
                admission_bytes=int(values[8]),
            )
        )
    if int(totals[2] + totals[3]) != int(totals[1]):
        raise RuntimeError("Total hits + misses does not equal requested experts")
    if capacity > 0 and int(totals[3]) != int(totals[4]):
        raise RuntimeError("Total misses and admissions differ")
    return SimulationResult(
        policy=policy_name,
        capacity=capacity,
        workload=workload.name,
        regime=workload.regime,
        seed=seed,
        alpha=alpha,
        events=int(totals[0]),
        requests=int(totals[1]),
        hits=int(totals[2]),
        misses=int(totals[3]),
        admissions=int(totals[4]),
        evictions=int(totals[5]),
        bytes_requested=int(totals[6]),
        bytes_transferred=int(totals[7]),
        admission_bytes=int(totals[8]),
        maximum_occupancy=maximum_occupancy,
        per_sequence=tuple(sequence_results),
    )


def result_rows(
    result: SimulationResult,
    *,
    lambda_values: Sequence[float],
    cost_models: Sequence[str],
    trace_hash: str,
    config_hash: str,
    domain_label: str,
    selected_decay_alpha: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = result.base_record()
    for cost_model in cost_models:
        if cost_model == "unit_miss":
            miss_cost = float(result.misses)
            switch_cost = float(result.admissions)
            denominator = float(result.requests)
        elif cost_model == "expert_bytes":
            miss_cost = float(result.bytes_transferred)
            switch_cost = float(result.admission_bytes)
            denominator = float(result.bytes_requested)
        else:
            raise ValueError(f"Unknown cost model {cost_model!r}")
        for switch_lambda in lambda_values:
            switch_lambda = float(switch_lambda)
            total_cost = miss_cost + switch_lambda * switch_cost
            row = dict(base)
            row.update(
                {
                    "schema_version": "race_stage0_result_row_v1",
                    "cache_capacity": result.capacity,
                    "domain": domain_label,
                    "cost_model": cost_model,
                    "lambda": switch_lambda,
                    "miss_cost": miss_cost,
                    "switch_cost": switch_cost,
                    "total_cost": total_cost,
                    "normalized_total_cost": total_cost / denominator,
                    "trace_hash": trace_hash,
                    "config_hash": config_hash,
                    "selected_decay_alpha": (
                        result.policy == "lfu_decay"
                        and result.alpha is not None
                        and selected_decay_alpha is not None
                        and abs(result.alpha - selected_decay_alpha) < 1e-12
                    ),
                }
            )
            rows.append(row)
    return rows


def per_sequence_rows(
    result: SimulationResult, *, trace_hash: str, config_hash: str
) -> list[dict[str, Any]]:
    rows = []
    for item in result.per_sequence:
        row = asdict(item)
        row.update(
            {
                "schema_version": "race_stage0_per_sequence_v1",
                "condition_id": result.condition_id,
                "policy": result.policy,
                "capacity": result.capacity,
                "cache_capacity": result.capacity,
                "workload": result.workload,
                "regime": result.regime,
                "seed": result.seed,
                "alpha": result.alpha,
                "trace_hash": trace_hash,
                "config_hash": config_hash,
            }
        )
        rows.append(row)
    return rows


def expected_unlimited_misses(trace: RoutingTrace, workload: Workload) -> int:
    seen: set[tuple[int, int]] = set()
    for _sequence, view in workload.iter_slices(trace):
        for index in range(view.start, view.stop):
            layer = int(trace.layer_index[index])
            seen.update((layer, int(expert)) for expert in trace.requested_expert_ids[index])
    return len(seen)


def policy_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"policy": "lru"}, {"policy": "lfu"}]
    specs.extend(
        {"policy": "lfu_decay", "alpha": float(alpha)}
        for alpha in config["lfu_decay_alphas"]
    )
    specs.append({"policy": "static_hotset"})
    specs.extend(
        {"policy": "random", "seed": int(seed)}
        for seed in config["random_policy_seeds"]
    )
    return specs
