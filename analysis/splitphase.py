"""Stale learned orders and cutoff-local fresh corrections, simulated on the development fold (Section 4).

One event-level policy covers every arm of the staleness simulations (Table 2):

  order source   fresh   : synchronous ranker, fresh features
                 lag_req : block computed after event p-1 with request R_{p-1} and the current token (P2)
                 lag_tok : as lag_req, with the token rows taken from event p-1's block (P1)
                 lfu     : decayed LFU (alpha 0.95), tie-break (-metric, id)
  window         none    : order used as is
                 fresh   : the 2w candidates at positions spare-w .. spare+w-1 are rescored
                           with fresh features and reordered among themselves
                 oracle  : the same window reordered by true capped next-use (upper bound)

Learned orders use the simulator's tie-break (-score, -last_used, id). For ``fresh`` the replay
reproduces ``run_caer.evaluate_workload`` exactly (a sanity check).

Diagnostics (lagged and lfu sources): at every eviction decision the fresh order on the same
candidates is also computed. When the kept sets differ, the fresh-order rank offset (rank - spare)
of each expert in the symmetric difference is recorded, with whether the window covered all of them.

    python analysis/splitphase.py --capacity 32 --source lag_req --window none \
        --output results/simulation/staleness/lag_req_none_B32.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(ROOT / "src" / "runtime"))
OPT = Path(__file__).resolve().parents[1] / "measurement"

DISTANCE_CAP = 33
TOKEN_ROWS = slice(45, 50)  # five causal token rows of the 50-row block


PREPARED_ROOT = None  # set by --prepared-root for Qwen3 / Mixtral


def load_dev(fit_name: str = "calA", deploy_name: str = "calB"):
    from replay_overhead import load_inputs  # sets up the measurement/ import paths first

    inputs = load_inputs(OPT, PREPARED_ROOT)
    from collect2 import sub

    cal = inputs.calibration
    folds = {"calA": sub(cal, 0, 40, "calA"), "calB": sub(cal, 40, 80, "calB")}
    return inputs, folds[fit_name], folds[deploy_name]


def future_distances(trace, workload):
    """Per layer ordinal: array (n_events_l, E) of distance to next use after each position."""

    layers = tuple(map(int, trace.metadata["layer_indices"]))
    l2o = {layer: i for i, layer in enumerate(layers)}
    E = int(trace.num_experts)
    streams = [[] for _ in layers]
    for _seq, view in workload.iter_slices(trace):
        for index in range(view.start, view.stop):
            streams[l2o[int(trace.layer_index[index])]].append(trace.requested_expert_ids[index])
    out = []
    big = np.iinfo(np.int32).max // 4
    for stream in streams:
        n = len(stream)
        nxt = np.full(E, big, dtype=np.int64)
        dist = np.empty((n, E), dtype=np.int64)
        for p in range(n - 1, -1, -1):
            dist[p] = nxt - p
            nxt[np.asarray(stream[p], dtype=np.int64)] = p
        out.append(dist)
    return out


class SplitPhase:
    def __init__(self, bundle, capacity: int, source: str, window: str, w: int = 2,
                 backend: str = "numba", future=None, diagnose: bool = True,
                 refresh: int = 1, touch: bool = False, mix_layer0: bool = False, lag_bundle=None,
                 lag_tok_bundle=None):
        from policy import GlobalScorer, LiveTokens

        self.bundle = bundle
        self.L, self.E, self.k = bundle.num_layers, bundle.num_experts, bundle.top_k
        self.capacity = int(capacity)
        self.spare = self.capacity - self.k
        self.source, self.window, self.w = source, window, int(w)
        self.tokens = LiveTokens(self.L)
        self.state = bundle.new_state(self.tokens)
        self.state.reset()
        self.scorer = GlobalScorer(bundle.payload["global_models"][self.capacity], backend)
        self.lag_scorer = (
            self.scorer if lag_bundle is None
            else GlobalScorer(lag_bundle.payload["global_models"][self.capacity], backend)
        )
        self.lag_tok_scorer = (
            self.lag_scorer if lag_tok_bundle is None
            else GlobalScorer(lag_tok_bundle.payload["global_models"][self.capacity], backend)
        )
        self.refresh, self.touch, self.mix_layer0 = int(refresh), bool(touch), bool(mix_layer0)
        self.cache = [None] * bundle.num_layers
        self.last_refresh = np.zeros(bundle.num_layers, dtype=np.int64)
        self.refreshes = 0
        self.touch_rows = 0
        self.resident = np.zeros((self.L, self.E), dtype=bool)
        self.last_used = np.full((self.L, self.E), -1, dtype=np.int64)
        self.position = np.zeros(self.L, dtype=np.int64)
        self.mask = np.zeros(self.E, dtype=bool)
        self.prev_request = [None] * self.L
        self.prev_gates = [None] * self.L
        self.prev_token_rows = [None] * self.L
        self.lfu = np.zeros((self.L, self.E))
        self.future = future
        self.diagnose = diagnose
        self.transfers = 0
        self.decisions = 0
        self.disagree_decisions = 0
        self.offsets = Counter()
        self.window_covers = 0
        self.window_fixed = 0
        self.timing = Counter()
        self.regret = Counter()
        # next-layer prefetching (T4), with the semantics of policy.CachePolicy.prefetch
        self.prefetched = np.zeros((self.L, self.E), dtype=bool)
        self.prefetch_rank = np.zeros((self.L, self.E), dtype=np.int64)
        self.prefetch_victims = np.zeros((self.L, self.E), dtype=bool)
        self.prefetch_transfers = 0
        self.prefetch_hits = 0
        self.prefetch_unused = 0
        self.prefetch_evicted_requested = 0

    def begin_sequence(self) -> None:
        self.state.begin_sequence()

    def _p2_scores(self, layer: int, token: int, experts) -> np.ndarray:
        """The P2 scores the worker publishes for layer's next event: state through the previous event,
        its request, and the current token (pushed for the duration of the call)."""

        p = int(self.position[layer])
        pr = self.prev_request[layer]
        self.tokens.push(layer, token)
        try:
            block = self.state.features(layer, pr, self.prev_gates[layer], np.sort(pr), p)
        finally:
            self.tokens.streams[layer].pop()
        return self.lag_scorer(block[:, experts].T)

    def prefetch(self, layer: int, token: int, guess, limit: int) -> tuple[np.ndarray, np.ndarray]:
        """policy.CachePolicy.prefetch with the published (P2) ranking; layer >= 1 only."""

        if layer == 0 or self.source not in ("lag_req",):
            raise ValueError("prefetching is simulated for split-phase P2 layers only")
        guess = np.asarray(guess, dtype=np.int64)
        row = self.resident[layer]
        want = [int(e) for e in guess if e >= 0 and not row[e]][: int(limit)]
        empty = np.zeros(0, dtype=np.int64)
        if not want:
            return empty, empty
        free = self.capacity - int(row.sum())
        need = max(0, len(want) - free)
        evicted = empty
        if need:
            in_guess = np.zeros(self.E, dtype=bool)
            in_guess[guess[guess >= 0]] = True
            eligible = np.flatnonzero(row & ~in_guess)
            if eligible.size < need:
                want = want[: free + eligible.size]
                need = eligible.size
            if need:
                if self.prev_request[layer] is None:
                    raise RuntimeError("prefetch victims needed before the layer's first event")
                scores = self._p2_scores(layer, token, eligible)
                order = self._learned_order(scores, eligible, layer)
                evicted = eligible[order[eligible.size - need:]]
                row[evicted] = False
                self.prefetched[layer, evicted] = False
                self.prefetch_victims[layer, evicted] = True
        loaded = np.asarray(want, dtype=np.int64)
        if loaded.size:
            rank = {int(e): i for i, e in enumerate(guess.tolist())}
            row[loaded] = True
            self.prefetched[layer, loaded] = True
            self.prefetch_rank[layer, loaded] = [rank[int(e)] for e in loaded]
            self.prefetch_transfers += int(loaded.size)
        return evicted, loaded

    def _wrong_guesses_last(self, layer: int, candidates, order):
        flags = self.prefetched[layer, candidates[order]]
        if not flags.any():
            return order
        wrong = order[flags]
        wrong = wrong[np.argsort(self.prefetch_rank[layer, candidates[wrong]], kind="stable")]
        return np.concatenate((order[~flags], wrong))

    # --------------------------------------------------------------- orders
    def _learned_order(self, scores, candidates, layer):
        return np.lexsort((candidates, -self.last_used[layer, candidates], -scores))

    def _lfu_order(self, candidates, layer):
        return np.lexsort((candidates, -self.lfu[layer, candidates]))

    def step(self, layer: int, token: int, request, gates) -> int:
        request = np.asarray(request, dtype=np.int64)
        gates = np.asarray(gates, dtype=np.float64)
        p = int(self.position[layer])
        self.tokens.push(layer, token)
        need_fresh = self.source == "fresh" or self.window == "fresh" or self.diagnose
        fresh_block = self.state.features(layer, request, gates, np.sort(request), p)
        lag_block = None
        source = self.source
        if self.mix_layer0 and layer == 0 and source == "lag_req":
            source = "lag_tok"
        if source in ("lag_req", "lag_tok"):
            if self.prev_request[layer] is None:
                lag_block = fresh_block
            else:
                t0 = time.perf_counter_ns()
                pr = self.prev_request[layer]
                lag_block = self.state.features(layer, pr, self.prev_gates[layer], np.sort(pr), p)
                if source == "lag_tok":
                    lag_block = lag_block.copy()
                    lag_block[TOKEN_ROWS] = self.prev_token_rows[layer]
                self.timing["lag_features_ns"] += time.perf_counter_ns() - t0
            if self.refresh > 1 or self.touch:
                stale = self.cache[layer] is None or p - self.last_refresh[layer] >= self.refresh
                if stale:
                    self.cache[layer] = self.lag_scorer(lag_block.T)
                    self.last_refresh[layer] = p
                    self.refreshes += 1
                elif self.touch and self.prev_request[layer] is not None:
                    touched = self.prev_request[layer]
                    self.cache[layer][touched] = self.lag_scorer(lag_block[:, touched].T)
                    self.touch_rows += int(touched.size)
        # decayed LFU counts the request before choosing, as policy.SimplePolicy does
        self.lfu[layer] *= 0.95
        self.lfu[layer, request] += 1.0

        self.mask[request] = True
        self.last_used[layer, request] = p
        row = self.resident[layer]
        misses = int(request.size) - int(row[request].sum())
        candidates = np.flatnonzero(row & ~self.mask)
        if candidates.size > self.spare:
            self.decisions += 1
            fresh_scores = None
            if fresh_block is not None and need_fresh:
                fresh_scores = self.scorer(fresh_block[:, candidates].T)
            if self.source == "fresh":
                order = self._learned_order(fresh_scores, candidates, layer)
            elif self.source == "lfu":
                order = self._lfu_order(candidates, layer)
            else:
                t0 = time.perf_counter_ns()
                if self.refresh > 1 or self.touch:
                    lag_scores = self.cache[layer][candidates]
                else:
                    scorer = self.lag_tok_scorer if (self.mix_layer0 and layer == 0) else self.lag_scorer
                    lag_scores = scorer(lag_block[:, candidates].T)
                self.timing["lag_score_ns"] += time.perf_counter_ns() - t0
                order = self._learned_order(lag_scores, candidates, layer)
            order = self._wrong_guesses_last(layer, candidates, order)
            base = order
            lo, hi = self.spare - self.w, self.spare + self.w
            if self.window != "none" and lo >= 0 and hi <= candidates.size:
                t0 = time.perf_counter_ns()
                win = order[lo:hi]
                experts = candidates[win]
                if self.window == "fresh":
                    win_scores = self.scorer(fresh_block[:, experts].T)
                    local = np.lexsort((experts, -self.last_used[layer, experts], -win_scores))
                elif self.window == "oracle":
                    dist = np.minimum(self.future[layer][p, experts], DISTANCE_CAP)
                    local = np.lexsort((experts, -self.last_used[layer, experts], dist))
                else:
                    raise ValueError(self.window)
                order = order.copy()
                order[lo:hi] = win[local]
                self.timing["window_ns"] += time.perf_counter_ns() - t0
            if self.diagnose and self.source != "fresh" and fresh_scores is not None:
                fresh_order = self._learned_order(fresh_scores, candidates, layer)
                kept_arm = set(candidates[base[: self.spare]].tolist())
                kept_fresh = set(candidates[fresh_order[: self.spare]].tolist())
                diff = kept_arm ^ kept_fresh
                if diff:
                    self.disagree_decisions += 1
                    rank = np.empty(candidates.size, dtype=np.int64)
                    rank[fresh_order] = np.arange(candidates.size)
                    pos = {int(c): int(rank[i]) for i, c in enumerate(candidates)}
                    for e in diff:
                        self.offsets[pos[e] - self.spare] += 1
                    if lo >= 0 and hi <= candidates.size:
                        window_set = set(candidates[base[lo:hi]].tolist())
                        if diff <= window_set:
                            self.window_covers += 1
                    final_kept = set(candidates[order[: self.spare]].tolist())
                    if final_kept == kept_fresh:
                        self.window_fixed += 1
                    if self.future is not None:
                        lag_victims = sorted(kept_fresh - kept_arm, key=lambda e: pos[e])
                        fresh_victims = sorted(kept_arm - kept_fresh, key=lambda e: pos[e])
                        for vl, vf in zip(lag_victims, fresh_victims):
                            off = pos[vl] - self.spare
                            bucket = ("-1" if off == -1 else "-2" if off == -2 else "-3..-4" if off >= -4
                                      else "-5..-8" if off >= -8 else "<=-9")
                            dl = min(int(self.future[layer][p, vl]), DISTANCE_CAP)
                            df = min(int(self.future[layer][p, vf]), DISTANCE_CAP)
                            outcome = "lag_worse" if dl < df else "lag_better" if dl > df else "tie"
                            self.regret[(bucket, outcome)] += 1
            keep = candidates[order[: self.spare]]
        else:
            keep = candidates
        row.fill(False)
        row[request] = True
        row[keep] = True
        self.transfers += misses
        prefetched = self.prefetched[layer]
        if prefetched.any():
            hit = int(prefetched[request].sum())
            self.prefetch_hits += hit
            self.prefetch_unused += int(prefetched.sum()) - hit
            prefetched.fill(False)
        victims = self.prefetch_victims[layer]
        if victims.any():
            self.prefetch_evicted_requested += int(victims[request].sum())
            victims.fill(False)
        self.mask[request] = False
        if fresh_block is not None:
            self.prev_token_rows[layer] = np.array(fresh_block[TOKEN_ROWS])
        self.state.absorb(layer, request, gates, p)
        self.prev_request[layer] = request
        self.prev_gates[layer] = gates
        self.position[layer] += 1
        return misses


def load_bundle(path):
    """Policy bundles: the exported ones or raw payloads written by retrain_lag.py."""

    import pickle

    from policy import PolicyBundle, _ExportUnpickler

    with open(path, "rb") as handle:
        payload = _ExportUnpickler(handle).load()
    return PolicyBundle(payload, Path(path))


def run_arm(bundle_path: str, capacity: int, source: str, window: str, w: int,
            limit: int | None = None, diagnose: bool = True, refresh: int = 1,
            touch: bool = False, mix_layer0: bool = False, lag_bundle_path: str | None = None,
            lag_tok_bundle_path: str | None = None, regret: bool = False) -> dict:
    inputs, fit, deploy = load_dev()
    trace = inputs.trace
    bundle = load_bundle(bundle_path)
    lag_bundle = None if lag_bundle_path is None else load_bundle(lag_bundle_path)
    lag_tok_bundle = None if lag_tok_bundle_path is None else load_bundle(lag_tok_bundle_path)
    future = future_distances(trace, deploy) if (window == "oracle" or regret) else None
    arm = SplitPhase(bundle, capacity, source, window, w, future=future, diagnose=diagnose,
                     refresh=refresh, touch=touch, mix_layer0=mix_layer0, lag_bundle=lag_bundle,
                     lag_tok_bundle=lag_tok_bundle)
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    l2o = {layer: i for i, layer in enumerate(layers)}
    requests = trace.requested_expert_ids.astype(np.int64, copy=False)
    gates = trace.router_weights.astype(np.float64, copy=False)
    per_sequence = []
    start = time.perf_counter()
    for count, (sequence, view) in enumerate(deploy.iter_slices(trace)):
        if limit is not None and count >= limit:
            break
        arm.begin_sequence()
        before = arm.transfers
        for index in range(view.start, view.stop):
            arm.step(l2o[int(trace.layer_index[index])], int(trace.token_id[index]),
                     requests[index], gates[index])
        per_sequence.append({"source_sequence_id": int(sequence.source_sequence_id),
                             "position": int(sequence.position), "transfers": arm.transfers - before})
    offsets = dict(sorted(arm.offsets.items()))
    total_off = sum(offsets.values())
    near = sum(v for k, v in offsets.items() if -2 <= k <= 1)
    tags = (f"_d{refresh}" if refresh > 1 else "") + ("_touch" if touch else "") + ("_mix0" if mix_layer0 else "") \
        + ("_native" if lag_bundle_path else "")
    return {
        "arm": (f"{source}+{window}" if window != "none" else source) + tags,
        "refresh": refresh, "touch": touch, "mix_layer0": mix_layer0, "lag_bundle": lag_bundle_path,
        "refreshes": arm.refreshes, "touch_rows": arm.touch_rows,
        "source": source, "window": window, "w": w, "capacity": capacity,
        "bundle": bundle_path, "deploy": deploy.name, "sequences": len(per_sequence),
        "transfers": arm.transfers, "decisions": arm.decisions,
        "disagree_decisions": arm.disagree_decisions,
        "disagree_fraction": arm.disagree_decisions / max(arm.decisions, 1),
        "offset_histogram": {str(k): v for k, v in offsets.items()},
        "fraction_within_cutoff_w2": near / total_off if total_off else None,
        "window_covers_disagreement": arm.window_covers / max(arm.disagree_decisions, 1),
        "window_fixed_disagreement": arm.window_fixed / max(arm.disagree_decisions, 1),
        "regret": {f"{b}|{o}": v for (b, o), v in sorted(arm.regret.items())},
        "timing_ms_per_decision": {k: v / 1e6 / max(arm.decisions, 1) for k, v in arm.timing.items()},
        "wall_seconds": time.perf_counter() - start,
        "per_sequence": per_sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(ROOT / "data/rankers/olmoe_setdev_dev_trees800.pkl"))
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--source", required=True, choices=("fresh", "lag_req", "lag_tok", "lfu"))
    parser.add_argument("--window", default="none", choices=("none", "fresh", "oracle"))
    parser.add_argument("--w", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-diagnose", action="store_true")
    parser.add_argument("--refresh", type=int, default=1)
    parser.add_argument("--touch", action="store_true")
    parser.add_argument("--mix-layer0", action="store_true")
    parser.add_argument("--lag-bundle")
    parser.add_argument("--lag-tok-bundle")
    parser.add_argument("--prepared-root", help="cross-model prepared inputs (Qwen3, Mixtral)")
    parser.add_argument("--regret", action="store_true", help="pairwise next-use regret of disagreements (Figure 2)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    global PREPARED_ROOT
    if args.prepared_root:
        PREPARED_ROOT = Path(args.prepared_root).resolve()
    result = run_arm(args.bundle, args.capacity, args.source, args.window, args.w, args.limit,
                     diagnose=not args.no_diagnose, refresh=args.refresh, touch=args.touch,
                     mix_layer0=args.mix_layer0, lag_bundle_path=args.lag_bundle,
                     lag_tok_bundle_path=args.lag_tok_bundle, regret=args.regret)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1) + "\n")
    print(f"{result['arm']} B={args.capacity}: transfers={result['transfers']:,} "
          f"decisions={result['decisions']:,} disagree={100 * result['disagree_fraction']:.2f}% "
          f"near_cutoff={result['fraction_within_cutoff_w2']} [{result['wall_seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
