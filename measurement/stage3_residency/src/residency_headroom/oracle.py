from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable, Sequence

import numpy as np

from .exact_solver import solve_exact
from .policies import CacheTransition


class FarthestFutureOracle:
    """Exact equal-cost offline policy for atomic mandatory-admission requests."""

    name = "oracle"

    def __init__(
        self, requests: Sequence[Iterable[int]], capacity: int, num_experts: int
    ) -> None:
        if capacity < 0 or capacity > num_experts:
            raise ValueError("Invalid cache capacity")
        self.requests = tuple(frozenset(map(int, request)) for request in requests)
        self.capacity = int(capacity)
        self.num_experts = int(num_experts)
        self.resident: frozenset[int] = frozenset()
        self.position = 0
        self.future = [deque() for _ in range(num_experts)]
        for position, request in enumerate(self.requests):
            if not request:
                raise ValueError("Atomic requests cannot be empty")
            if min(request) < 0 or max(request) >= num_experts:
                raise ValueError("Request contains an out-of-range expert")
            if capacity > 0 and len(request) > capacity:
                raise ValueError("Atomic request exceeds positive cache capacity")
            for expert in request:
                self.future[expert].append(position)

    def process(self, request: Iterable[int]) -> CacheTransition:
        if self.position >= len(self.requests):
            raise RuntimeError("Oracle received more events than its future trace")
        requested = frozenset(map(int, request))
        if requested != self.requests[self.position]:
            raise RuntimeError("Oracle event differs from its frozen future trace")
        before = self.resident
        hits = requested & before
        misses = requested - before
        for expert in requested:
            if not self.future[expert] or self.future[expert][0] != self.position:
                raise RuntimeError("Oracle future-use accounting is inconsistent")
            self.future[expert].popleft()
        if self.capacity == 0:
            after = frozenset()
        else:
            spare = self.capacity - len(requested)
            old = before - requested
            infinity = len(self.requests) + 1
            ranked = sorted(
                old,
                key=lambda expert: (
                    self.future[expert][0] if self.future[expert] else infinity,
                    expert,
                ),
            )
            after = requested | frozenset(ranked[:spare])
        admissions = after - before
        evictions = before - after
        if self.capacity > 0 and admissions != misses:
            raise RuntimeError("Oracle violated mandatory atomic admission")
        self.resident = after
        self.position += 1
        return CacheTransition(
            request=requested,
            before=before,
            after=after,
            hits=hits,
            misses=misses,
            admissions=admissions,
            evictions=evictions,
        )

    def finish(self) -> None:
        if self.position != len(self.requests):
            raise RuntimeError(
                f"Oracle consumed {self.position}/{len(self.requests)} frozen events"
            )


def oracle_solution(
    requests: Sequence[Iterable[int]], capacity: int, num_experts: int
) -> tuple[int, int, tuple[frozenset[int], ...]]:
    oracle = FarthestFutureOracle(requests, capacity, num_experts)
    misses = 0
    admissions = 0
    states: list[frozenset[int]] = []
    for request in requests:
        transition = oracle.process(request)
        misses += len(transition.misses)
        admissions += len(transition.admissions)
        states.append(transition.after)
    oracle.finish()
    return misses, admissions, tuple(states)


@dataclass(frozen=True)
class OracleValidation:
    passed: bool
    exhaustive_cases: int
    random_cases: int
    lambda_values: tuple[float, ...]
    maximum_cost_difference: float
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "exhaustive_cases": self.exhaustive_cases,
            "random_cases": self.random_cases,
            "lambda_values": list(self.lambda_values),
            "maximum_cost_difference": self.maximum_cost_difference,
            "seed": self.seed,
        }


def validate_oracle(
    *,
    random_cases: int = 500,
    seed: int = 20260819,
    lambda_values: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    exhaustive_max_events: int = 4,
) -> OracleValidation:
    """Compare the scalable oracle with exact DP on exhaustive and random cases."""

    checked = 0
    maximum = 0.0
    lambdas = tuple(map(float, lambda_values))
    for num_experts in range(1, 5):
        universe = tuple(range(num_experts))
        for capacity in range(1, num_experts + 1):
            for request_size in range(1, min(2, capacity, num_experts) + 1):
                choices = tuple(frozenset(value) for value in combinations(universe, request_size))
                for length in range(1, exhaustive_max_events + 1):
                    for requests in product(choices, repeat=length):
                        maximum = max(
                            maximum,
                            _compare_one(requests, capacity, num_experts, lambdas),
                        )
                        checked += 1

    rng = np.random.default_rng(seed)
    for _ in range(random_cases):
        num_experts = int(rng.integers(2, 9))
        capacity = int(rng.integers(1, min(4, num_experts) + 1))
        length = int(rng.integers(1, 16))
        requests = []
        for _position in range(length):
            size = int(rng.integers(1, min(3, capacity) + 1))
            requests.append(
                frozenset(map(int, rng.choice(num_experts, size=size, replace=False)))
            )
        maximum = max(
            maximum, _compare_one(tuple(requests), capacity, num_experts, lambdas)
        )
    return OracleValidation(True, checked, random_cases, lambdas, maximum, seed)


def _compare_one(
    requests: Sequence[frozenset[int]],
    capacity: int,
    num_experts: int,
    lambdas: tuple[float, ...],
) -> float:
    misses, admissions, _states = oracle_solution(requests, capacity, num_experts)
    maximum = 0.0
    for switch_lambda in lambdas:
        exact = solve_exact(
            requests,
            capacity,
            num_experts,
            switch_lambda=switch_lambda,
        )
        oracle_cost = misses + switch_lambda * admissions
        difference = abs(float(oracle_cost) - exact.total_cost)
        maximum = max(maximum, difference)
        if difference > 1e-10:
            raise AssertionError(
                "Farthest-future oracle differs from exact optimum: "
                f"N={num_experts}, C={capacity}, lambda={switch_lambda}, "
                f"oracle={oracle_cost}, exact={exact.total_cost}, requests={requests}"
            )
    return maximum
