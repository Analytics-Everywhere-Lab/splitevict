"""Offline ranking diagnostics.

`FutureOccurrences` holds the complete same-layer future of a frozen workload path. The
simulator reads next-use distances from it only for offline labels and oracle-referenced
ranking measurements; nothing it returns re-enters policy state.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import Workload


class FutureOccurrences:
    """Per-layer next-use lookup built once from the frozen workload path."""

    def __init__(self, streams: Sequence[np.ndarray], num_experts: int) -> None:
        self.num_experts = int(num_experts)
        self._flat: list[np.ndarray] = []
        self._offsets: list[np.ndarray] = []
        self.sentinel: list[int] = []
        for requests in streams:
            values = np.asarray(requests, dtype=np.int64)
            length = int(values.shape[0])
            sentinel = length + 1
            experts = values.ravel()
            positions = np.repeat(np.arange(length, dtype=np.int64), values.shape[1])
            order = np.lexsort((positions, experts))
            sorted_experts = experts[order]
            sorted_positions = positions[order]
            counts = np.bincount(sorted_experts, minlength=self.num_experts).astype(np.int64)
            offsets = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(counts + 1))
            )
            flat = np.full(int(offsets[-1]), sentinel, dtype=np.int64)
            within = np.arange(sorted_positions.size, dtype=np.int64) - np.repeat(
                np.cumsum(counts) - counts, counts
            )
            flat[np.repeat(offsets[: self.num_experts], counts) + within] = sorted_positions
            self._flat.append(flat)
            self._offsets.append(offsets)
            self.sentinel.append(sentinel)
        self._cursor = [offsets[: self.num_experts].copy() for offsets in self._offsets]

    def reset(self) -> None:
        self._cursor = [offsets[: self.num_experts].copy() for offsets in self._offsets]

    def advance(self, layer_ordinal: int, position: int, request: np.ndarray) -> np.ndarray:
        """Consume the current event and return next-use distances for every expert."""

        flat = self._flat[layer_ordinal]
        cursor = self._cursor[layer_ordinal]
        if not np.all(flat[cursor[request]] == position):
            raise RuntimeError("Observer future-use accounting diverged from the trace")
        cursor[request] += 1
        return flat[cursor] - position

    def finish(self) -> None:
        for layer, cursor in enumerate(self._cursor):
            flat = self._flat[layer]
            if not np.all(flat[cursor] == self.sentinel[layer]):
                raise RuntimeError("Observer did not consume the complete same-layer future")


def build_future_occurrences(
    trace: RoutingTrace, workload: Workload, layer_streams: Sequence[np.ndarray]
) -> FutureOccurrences:
    return FutureOccurrences(
        [trace.requested_expert_ids[indices] for indices in layer_streams],
        trace.num_experts,
    )
