"""Copy of ``src/runtime/splitphase_service.py`` with a worker slowdown, for the worker-slowdown emulation.

Each worker busy-waits after every scorer call so that the call takes ``SPLITPHASE_SLOWDOWN`` (an environment
variable, default 1) times its measured time. The copy has no prefetch support, CPU pinning or idle parking;
the rest follows the service, whose docstring describes the design. Decisions are deterministic and equal
``analysis/splitphase.py --source lag_req --mix-layer0`` with the same bundles.

    SPLITPHASE_SLOWDOWN=4 python analysis/emulate_decode.py --policy splitphase --slow-service --capacity 32 \
        --layer-ms 2.6 --workers 1 --limit 10 --p2 data/rankers/olmoe_dev_lag_req_trees200.pkl \
        --p1 data/rankers/olmoe_dev_lag_tok_trees200.pkl --output results/simulation/worker_slack/slow_x4.json
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from policy import CachePolicy, StepResult  # noqa: E402

TOKEN_ROWS = slice(45, 50)
RING = 1024
SEQ, TOK, REQ, STOP = 1, 2, 3, 4


def _load_bundle(path):
    from policy import PolicyBundle, _ExportUnpickler

    with open(path, "rb") as handle:
        payload = _ExportUnpickler(handle).load()
    return PolicyBundle(payload, Path(path))


class _Shared:
    """Named shared-memory arrays; created by the service, attached by the workers."""

    def __init__(self, L: int, E: int, k: int, W: int, create: bool, names: dict | None = None):
        self.specs = {
            "ctrl": ((4,), np.int64),            # [write index, unused...]
            "cursor": ((W,), np.int64),          # per-worker read cursor
            "busy": ((W,), np.int64),            # per-worker busy ns
            "prof": ((W, 4), np.int64),          # per-worker [features ns, score ns, publishes, idle spins]
            "kind": ((RING,), np.int64),
            "layer": ((RING,), np.int64),
            "token": ((RING,), np.int64),
            "req": ((RING, k), np.int64),
            "gates": ((RING, k), np.float64),
            "resmask": ((RING, E), np.uint8),
            "scores": ((L, E), np.float64),
            "pos": ((L,), np.int64),
            "gen": ((L,), np.int64),
            "chk": ((L,), np.float64),
        }
        self.blocks = {}
        self.names = {}
        for key, (shape, dtype) in self.specs.items():
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if create:
                block = shared_memory.SharedMemory(create=True, size=size)
            else:
                block = shared_memory.SharedMemory(name=names[key])
            self.blocks[key] = block
            self.names[key] = block.name
            arr = np.ndarray(shape, dtype=dtype, buffer=block.buf)
            if create:
                arr[...] = 0
            setattr(self, key, arr)
        if create:
            self.pos[:] = -1

    def close(self, unlink: bool) -> None:
        for key in list(self.blocks):
            delattr(self, key)
        for block in self.blocks.values():
            block.close()
            if unlink:
                block.unlink()


def _worker(index: int, workers: int, p2_path: str, p1_path: str, capacity: int, backend: str,
            dims: tuple[int, int, int], names: dict, syspath: list[str]) -> None:
    sys.path[:] = syspath
    import os

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    from policy import GlobalScorer, LiveTokens

    L, E, k = dims
    sh = _Shared(L, E, k, workers, create=False, names=names)
    p2 = _load_bundle(p2_path)
    p1 = _load_bundle(p1_path)
    tokens = LiveTokens(L)
    state = p2.new_state(tokens)
    state.reset()
    scorer_p2 = GlobalScorer(p2.payload["global_models"][capacity], backend)
    scorer_p1 = GlobalScorer(p1.payload["global_models"][capacity], backend)
    mine = [layer for layer in range(L) if layer % workers == index]
    position = {layer: 0 for layer in mine}
    prev_req = {layer: None for layer in mine}
    prev_gates = {layer: None for layer in mine}
    prev_tok = {layer: None for layer in mine}
    resident = {layer: np.zeros(0, dtype=np.int64) for layer in mine}
    generation = 0
    warm = np.zeros((50, E))
    scorer_p2(warm[:, :4].T)
    scorer_p1(warm[:, :4].T)
    sh.cursor[index] = -1  # ready

    import os as _os
    slowdown = float(_os.environ.get("SPLITPHASE_SLOWDOWN", "1"))

    def publish(layer, p, block, scorer):
        scores = np.full(E, np.nan)
        ids = resident[layer]
        if block is not None and ids.size:
            t0 = time.perf_counter_ns()
            scores[ids] = scorer(block[:, ids].T)
            spent = time.perf_counter_ns() - t0
            if slowdown > 1.0:
                end = time.perf_counter_ns() + int((slowdown - 1.0) * spent)
                while time.perf_counter_ns() < end:
                    pass
            sh.prof[index, 1] += time.perf_counter_ns() - t0
        sh.prof[index, 2] += 1
        sh.scores[layer] = scores
        sh.chk[layer] = np.nansum(scores)
        sh.gen[layer] = generation
        sh.pos[layer] = p

    def p1_scores(layer):
        p = position[layer]
        if prev_req[layer] is None:
            publish(layer, p, None, scorer_p1)
            return
        r = prev_req[layer]
        tokens.push(layer, -1)  # placeholder; the token rows are replaced below
        try:
            block = state.features(layer, r, prev_gates[layer], np.sort(r), p)
        finally:
            tokens.streams[layer].pop()
        block = block.copy()
        block[TOKEN_ROWS] = prev_tok[layer]
        publish(layer, p, block, scorer_p1)

    cursor = 0
    sh.cursor[index] = 0
    busy = 0
    while True:
        spins = 0
        while sh.ctrl[0] <= cursor:
            spins += 1
        sh.prof[index, 3] += spins
        start = time.perf_counter_ns()
        slot = cursor % RING
        kind = int(sh.kind[slot])
        if kind == STOP:
            sh.busy[index] = busy
            sh.cursor[index] = cursor + 1
            sh.close(unlink=False)
            return
        if kind == SEQ:
            state.begin_sequence()
            generation += 1
            if 0 in mine:
                p1_scores(0)
        elif kind == TOK:
            token = int(sh.token[slot])
            for layer in mine:
                tokens.push(layer, token)
                if layer == 0:
                    continue
                p = position[layer]
                if prev_req[layer] is None:
                    publish(layer, p, None, scorer_p2)
                    continue
                r = prev_req[layer]
                t0 = time.perf_counter_ns()
                block = state.features(layer, r, prev_gates[layer], np.sort(r), p)
                sh.prof[index, 0] += time.perf_counter_ns() - t0
                publish(layer, p, block, scorer_p2)
        elif kind == REQ:
            layer = int(sh.layer[slot])
            if layer in position:
                r = sh.req[slot].copy()
                g = sh.gates[slot].copy()
                p = position[layer]
                if layer == 0:
                    fresh = state.features(layer, r, g, np.sort(r), p)
                    prev_tok[layer] = np.array(fresh[TOKEN_ROWS])
                state.absorb(layer, r, g, p)
                prev_req[layer] = r
                prev_gates[layer] = g
                resident[layer] = np.flatnonzero(sh.resmask[slot])
                position[layer] = p + 1
                if layer == 0:
                    p1_scores(0)
        cursor += 1
        sh.cursor[index] = cursor
        busy += time.perf_counter_ns() - start


class SplitPhaseService:
    def __init__(self, p2_bundle: str, p1_bundle: str, capacity: int, *, workers: int = 4,
                 num_layers: int, num_experts: int, top_k: int, backend: str = "numba",
                 start_timeout: float = 300.0) -> None:
        self.W, self.L, self.E, self.k = int(workers), int(num_layers), int(num_experts), int(top_k)
        self.sh = _Shared(self.L, self.E, self.k, self.W, create=True)
        self.sh.cursor[:] = -2  # not started
        ctx = mp.get_context("spawn")
        self.procs = []
        for j in range(self.W):
            proc = ctx.Process(target=_worker, args=(j, self.W, str(p2_bundle), str(p1_bundle), int(capacity),
                                                     backend, (self.L, self.E, self.k), dict(self.sh.names),
                                                     list(sys.path)), daemon=True)
            proc.start()
            self.procs.append(proc)
        deadline = time.time() + start_timeout
        while (self.sh.cursor < 0).any():
            if time.time() > deadline or any(not p.is_alive() for p in self.procs):
                raise RuntimeError("split-phase workers failed to start")
            time.sleep(0.01)
        self.write = 0
        self.generation = 0
        self.closed = False

    def _slot(self) -> int:
        while self.write - int(self.sh.cursor.min()) >= RING - 1:
            pass  # ring full: workers are behind (never observed in practice)
        return self.write % RING

    def _publish(self) -> None:
        self.write += 1
        self.sh.ctrl[0] = self.write

    def begin_sequence(self) -> None:
        self.generation += 1
        slot = self._slot()
        self.sh.kind[slot] = SEQ
        self._publish()

    def begin_token(self, token: int) -> None:
        slot = self._slot()
        self.sh.kind[slot] = TOK
        self.sh.token[slot] = int(token)
        self._publish()

    def post(self, layer: int, request: np.ndarray, gates: np.ndarray, resident_row: np.ndarray) -> None:
        slot = self._slot()
        self.sh.kind[slot] = REQ
        self.sh.layer[slot] = int(layer)
        self.sh.req[slot] = request
        self.sh.gates[slot] = gates
        self.sh.resmask[slot] = resident_row
        self._publish()

    def scores(self, layer: int, position: int) -> tuple[np.ndarray, int]:
        """Spin until the owner has published (layer, position, current generation)."""

        t0 = time.perf_counter_ns()
        sh = self.sh
        while True:
            if sh.pos[layer] == position and sh.gen[layer] == self.generation:
                s = sh.scores[layer].copy()
                if np.nansum(s) == sh.chk[layer] and sh.pos[layer] == position:
                    return s, time.perf_counter_ns() - t0

    def close(self) -> list[float]:
        if self.closed:
            return []
        slot = self._slot()
        self.sh.kind[slot] = STOP
        self._publish()
        for proc in self.procs:
            proc.join(timeout=10)
        busy = [float(b) for b in self.sh.busy]
        self.profile = self.sh.prof.copy().tolist()
        self.sh.close(unlink=True)
        self.closed = True
        return busy


class SplitPhasePolicy(CachePolicy):
    """Critical-path half of split-phase eviction: mask, order by published scores, keep top spare."""

    name = "splitphase"

    def __init__(self, service: SplitPhaseService, num_layers: int, num_experts: int, capacity: int,
                 top_k: int) -> None:
        super().__init__(num_layers, num_experts, capacity, top_k)
        self.service = service
        self.last_used = np.full((self.L, self.E), -1, dtype=np.int64)
        self.mask = np.zeros(self.E, dtype=bool)
        self.wait_ns = 0
        self.decide_ns = 0
        self.post_ns = 0

    def begin_sequence(self) -> None:
        self.service.begin_sequence()

    def begin_token(self, token: int) -> None:
        self.service.begin_token(token)

    def step(self, layer: int, token: int, request: np.ndarray, gates: np.ndarray) -> StepResult:
        t0 = time.perf_counter_ns()
        request = np.asarray(request, dtype=np.int64)
        p = int(self.position[layer])
        self.mask[request] = True
        self.last_used[layer, request] = p
        row = self.resident[layer]
        candidates = np.flatnonzero(row & ~self.mask)
        decision = candidates.size > self.spare
        waited = 0
        if decision:
            self.eviction_decisions += 1
            scores, waited = self.service.scores(layer, p)
            s = scores[candidates]
            if np.isnan(s).any():
                raise RuntimeError(f"worker scores missing for resident candidates at layer {layer} p={p}")
            order = np.lexsort((candidates, -self.last_used[layer, candidates], -s))
            keep = candidates[order[: self.spare]]
        else:
            keep = candidates
        evicted, loaded, misses = self._apply(layer, request, keep)
        self.mask[request] = False
        t1 = time.perf_counter_ns()
        self.service.post(layer, request, gates, self.resident[layer])
        t2 = time.perf_counter_ns()
        self.position[layer] += 1
        self.wait_ns += waited
        self.decide_ns += (t1 - t0) - waited
        self.post_ns += t2 - t1
        return StepResult(misses, evicted, loaded, decision,
                          timings_ns={"policy": t2 - t0, "wait": waited, "decide": (t1 - t0) - waited,
                                      "post": t2 - t1})
