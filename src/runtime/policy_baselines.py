"""Eviction rules of two published MoE offloading systems, in the per-layer cache model of ``policy.py``.

Both follow ``policy.CachePolicy``: mandatory atomic admission of the layer's request, the other residents
kept in the policy's order, prefetching through ``CachePolicy.prefetch`` (``keep_order``). Ties break on
recency (the last event of the layer that used the expert, over all sequences), then the smaller expert id,
as the learned policies' ties do.

* ``HobbitPolicy`` -- HOBBIT's sequence-level multidimensional caching policy (Tang et al. 2024, Sec. 4.3):
  priority p_t = w_lru R_t/T + w_lfu F_t/T + w_lhu H_t/T + w_fld p_t^fld, with the LRU, LFU and LHU records
  reset at the start of every sequence. With one precision every use is high precision, so H_t = F_t and the
  LHU term merges into the LFU term; with B slots per layer every candidate of an eviction shares the layer,
  so the layer-distance term p^fld is a constant. The ranking is therefore by w R_t + (1 - w) F_t, where R_t
  is the token of the expert's last use in this sequence and F_t its uses in this sequence (T, the token
  count, divides every candidate alike). HOBBIT fits the weights on calibration data; ``w`` is fitted the
  same way.
* ``MoEInfinityPolicy`` -- MoE-Infinity's activation-aware expert cache (Xue et al. 2024, Algorithm 1):
  the request's expert activation matrix (EAM, layers x experts, tokens routed to each expert so far) is
  matched to the nearest stored request EAM by cosine distance, and a cached expert's priority is
  (p_eam[l, e] + eps) (1 - l / L) / n_token(l), where p_eam is the matched EAM. With B slots per layer the
  layer factor and n_token(l) are constants of the eviction, so the ranking is by the matched EAM's count
  for (l, e). Stored EAMs are the calibration sequences'; with ``online`` each finished sequence's EAM is
  added, as MoE-Infinity records the requests it serves. Before the first event of a sequence the summed
  store is the prediction.
"""

from __future__ import annotations

import time

import numpy as np

from policy import CachePolicy, StepResult

MOE_INFINITY_EPS = 1e-6


class _RecencyTies(CachePolicy):
    """Shared bookkeeping: the per-layer event clock and each expert's last use (never reset)."""

    def __init__(self, num_layers: int, num_experts: int, capacity: int, top_k: int) -> None:
        super().__init__(num_layers, num_experts, capacity, top_k)
        self.clock = np.zeros(self.L, dtype=np.int64)
        self.last_use = np.full((self.L, self.E), -1, dtype=np.int64)

    def _score(self, layer: int, candidates: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def keep_order(self, layer: int, candidates: np.ndarray) -> np.ndarray:
        """Larger score first; ties to the more recent last use, then the smaller expert id."""

        return np.lexsort((candidates, -self.last_use[layer, candidates], -self._score(layer, candidates)))

    def _observe(self, layer: int, request: np.ndarray) -> None:
        raise NotImplementedError

    def step(self, layer: int, token: int, request: np.ndarray, gates: np.ndarray) -> StepResult:
        start = time.perf_counter_ns()
        request = np.asarray(request, dtype=np.int64)
        self.clock[layer] += 1
        self._observe(layer, request)
        self.last_use[layer, request] = self.clock[layer]
        row = self.resident[layer]
        mask = np.zeros(self.E, dtype=bool)
        mask[request] = True
        old = np.flatnonzero(row & ~mask)
        decision = old.size > self.spare
        if decision:
            order = self.keep_order(layer, old)
            if self.pf_active[layer] and self.wrong_guess_rule == "last":
                order = self._wrong_guesses_last(layer, old, order)
            keep = old[order[: self.spare]]
            self.eviction_decisions += 1
        else:
            keep = old
        evicted, loaded, misses = self._apply(layer, request, keep)
        self.position[layer] += 1
        return StepResult(misses, evicted, loaded, decision, timings_ns={"policy": time.perf_counter_ns() - start})


class HobbitPolicy(_RecencyTies):
    name = "hobbit"

    def __init__(self, num_layers: int, num_experts: int, capacity: int, top_k: int, *, w: float) -> None:
        super().__init__(num_layers, num_experts, capacity, top_k)
        if not 0.0 <= w <= 1.0:
            raise ValueError("w must lie in [0, 1]")
        self.w = float(w)
        self.token = np.zeros(self.L, dtype=np.int64)  # tokens of this sequence seen by each layer (T)
        self.recent = np.zeros((self.L, self.E), dtype=np.float64)  # R_t: token of the last use, this sequence
        self.count = np.zeros((self.L, self.E), dtype=np.float64)  # F_t: uses in this sequence

    def begin_sequence(self) -> None:
        self.token.fill(0)
        self.recent.fill(0.0)
        self.count.fill(0.0)

    def _score(self, layer: int, candidates: np.ndarray) -> np.ndarray:
        return self.w * self.recent[layer, candidates] + (1.0 - self.w) * self.count[layer, candidates]

    def _observe(self, layer: int, request: np.ndarray) -> None:
        self.token[layer] += 1
        self.recent[layer, request] = float(self.token[layer])
        self.count[layer, request] += 1.0


class MoEInfinityPolicy(_RecencyTies):
    name = "moe_infinity"

    def __init__(self, num_layers: int, num_experts: int, capacity: int, top_k: int, *, store: np.ndarray,
                 online: bool = True) -> None:
        super().__init__(num_layers, num_experts, capacity, top_k)
        store = np.asarray(store, dtype=np.float64)
        if store.ndim != 3 or store.shape[1:] != (self.L, self.E) or store.shape[0] < 1:
            raise ValueError("store must be shaped [requests, layers, experts]")
        self.online = bool(online)
        self.store = [m.copy() for m in store]
        self._flat = store.reshape(store.shape[0], -1).copy()  # one row per stored request EAM
        self._norm = np.linalg.norm(self._flat, axis=1)
        self.eam = np.zeros((self.L, self.E), dtype=np.float64)  # this request's EAM so far
        self._dot = np.zeros(len(self.store), dtype=np.float64)  # <eam, stored EAM> per stored request
        self._sq = 0.0  # ||eam||^2
        self.matched = store.sum(axis=0)  # the prediction before the request's first event
        self.match_index = -1

    def begin_sequence(self) -> None:
        if self.online and self._sq > 0.0:
            self._add(self.eam.copy())
        self.eam.fill(0.0)
        self._dot = np.zeros(len(self.store), dtype=np.float64)
        self._sq = 0.0
        self.matched = np.sum(self.store, axis=0)
        self.match_index = -1

    def _add(self, eam: np.ndarray) -> None:
        self.store.append(eam)
        self._flat = np.vstack((self._flat, eam.reshape(1, -1)))
        self._norm = np.append(self._norm, np.linalg.norm(eam))

    def _score(self, layer: int, candidates: np.ndarray) -> np.ndarray:
        return self.matched[layer, candidates] + MOE_INFINITY_EPS

    def _observe(self, layer: int, request: np.ndarray) -> None:
        # incremental cosine: one routed token adds 1 to each requested (layer, expert) cell
        row = self.eam[layer]
        self._sq += float(2.0 * row[request].sum() + request.size)
        row[request] += 1.0
        base = layer * self.E
        self._dot += self._flat[:, base + request].sum(axis=1)
        cosine = self._dot / (np.sqrt(self._sq) * np.maximum(self._norm, 1e-12))
        best = int(np.argmax(cosine))  # first on ties: the earliest stored request
        if best != self.match_index:
            self.match_index = best
            self.matched = self.store[best]


def request_eams(trace, workload, layer_order: dict[int, int]) -> np.ndarray:
    """EAM of every sequence of ``workload`` (``iter_slices``), shaped [sequences, layers, experts]."""

    L, E = int(trace.num_layers), int(trace.num_experts)
    requests = trace.requested_expert_ids
    out = []
    for _sequence, view in workload.iter_slices(trace):
        eam = np.zeros((L, E), dtype=np.float64)
        for index in range(view.start, view.stop):
            np.add.at(eam[layer_order[int(trace.layer_index[index])]], np.asarray(requests[index], dtype=np.int64), 1.0)
        out.append(eam)
    return np.stack(out)
