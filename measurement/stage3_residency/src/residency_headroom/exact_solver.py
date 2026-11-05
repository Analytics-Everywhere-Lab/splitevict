from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ExactSolution:
    total_cost: float
    miss_cost: float
    switch_cost: float
    misses: int
    admissions: int
    states: tuple[frozenset[int], ...]


@dataclass(frozen=True)
class _DPValue:
    total_cost: float
    miss_cost: float
    switch_cost: float
    misses: int
    admissions: int
    states: tuple[frozenset[int], ...]


def solve_exact(
    requests: Sequence[Iterable[int]],
    capacity: int,
    num_experts: int,
    *,
    miss_costs: np.ndarray | None = None,
    admission_costs: np.ndarray | None = None,
    switch_lambda: float = 0.0,
) -> ExactSolution:
    """Exact DP over feasible post-event cache states for tiny validation traces."""

    if capacity < 0 or capacity > num_experts:
        raise ValueError("Invalid cache capacity")
    if switch_lambda < 0:
        raise ValueError("switch_lambda must be nonnegative")
    miss_values = _cost_vector(miss_costs, num_experts)
    admission_values = _cost_vector(admission_costs, num_experts)
    normalized = tuple(frozenset(map(int, request)) for request in requests)
    for request in normalized:
        if not request:
            raise ValueError("Atomic requests cannot be empty")
        if min(request) < 0 or max(request) >= num_experts:
            raise ValueError("Request contains an out-of-range expert")
        if capacity > 0 and len(request) > capacity:
            raise ValueError("Atomic request exceeds positive cache capacity")

    current: dict[frozenset[int], _DPValue] = {
        frozenset(): _DPValue(0.0, 0.0, 0.0, 0, 0, tuple())
    }
    tolerance = 1e-12
    for request in normalized:
        following: dict[frozenset[int], _DPValue] = {}
        for before, value in current.items():
            misses = request - before
            event_miss_cost = float(sum(miss_values[item] for item in misses))
            if capacity == 0:
                candidates = (frozenset(),)
            else:
                old = sorted(before - request)
                spare = capacity - len(request)
                candidates = tuple(
                    request | frozenset(extra)
                    for size in range(min(spare, len(old)) + 1)
                    for extra in combinations(old, size)
                )
            for after in candidates:
                admitted = after - before
                event_switch = float(sum(admission_values[item] for item in admitted))
                miss_total = value.miss_cost + event_miss_cost
                switch_total = value.switch_cost + event_switch
                total = miss_total + switch_lambda * switch_total
                candidate = _DPValue(
                    total,
                    miss_total,
                    switch_total,
                    value.misses + len(misses),
                    value.admissions + len(admitted),
                    value.states + (after,),
                )
                previous = following.get(after)
                if previous is None or total < previous.total_cost - tolerance or (
                    abs(total - previous.total_cost) <= tolerance
                    and _state_key(candidate.states) < _state_key(previous.states)
                ):
                    following[after] = candidate
        current = following
    if not current:
        raise RuntimeError("Exact solver produced no feasible cache state")
    best = min(current.values(), key=lambda item: (item.total_cost, _state_key(item.states)))
    return ExactSolution(
        total_cost=best.total_cost,
        miss_cost=best.miss_cost,
        switch_cost=best.switch_cost,
        misses=best.misses,
        admissions=best.admissions,
        states=best.states,
    )


def _cost_vector(values: np.ndarray | None, num_experts: int) -> np.ndarray:
    if values is None:
        return np.ones(num_experts, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (num_experts,):
        raise ValueError(f"Cost vector has shape {array.shape}; expected {(num_experts,)}")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Cost vector must be finite and nonnegative")
    return array


def _state_key(states: tuple[frozenset[int], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(sorted(state)) for state in states)
