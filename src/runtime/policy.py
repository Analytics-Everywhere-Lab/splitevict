"""Expert-eviction policies for a live decode loop, one routing event at a time.

Each policy is the simulator's eviction rule (``run_caer.evaluate_workload``) as an object that
receives ``(layer, token, requested experts, router weights)`` and returns which experts to evict
and which to load. Nothing here reads the future. The feature state, the global token-GBT, the
stage-2 classifier and the tie-breaks are the fitted objects exported by
``run_caer.py --export-models``. ``replay_overhead.py`` checks the transfer counts against the
simulator's per-sequence records, so ``offload_runtime.py`` deploys the policy the paper measured.

Simple policies (LRU, decayed LFU, static hotset) follow ``residency_headroom.policies`` exactly:
the clock advances once per request at that layer, the metric is observed before the choice, and
ties break on the smaller expert id.
"""

from __future__ import annotations

import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    # ``src/`` holds ``caer``, ``run_caer`` and the feature modules ``features2``/``features3``.
    # The policies never import ``harness`` or ``run8``: both need the measurement code.
    sys.path.append(str(CODE))

from caer import (  # noqa: E402
    build_boundary_features,
    combine_boundary_scores,
    rerank_boundary,
)

LEARNED_VARIANTS = ("global_token_gbt", "caer_gbt", "caer_logistic")
_EMPTY = np.zeros(0, dtype=np.int64)
SIMPLE_VARIANTS = ("lru", "lfu", "lfu_decay", "static_hotset")
LFU_DECAY_ALPHA = 0.95


class MatrixModels:
    """``.matrix(horizon, layer)`` over the stacked ``[layer, horizon, from, to]`` array."""

    def __init__(self, transition: np.ndarray, horizons: tuple[int, ...]) -> None:
        self.transition = np.asarray(transition, dtype=np.float64)
        self.index = {int(h): i for i, h in enumerate(horizons)}

    def matrix(self, horizon: int, layer: int) -> np.ndarray:
        return self.transition[int(layer), self.index[int(horizon)]]


class LiveTokens:
    """Per-layer token streams that grow as decode proceeds (``tokens[layer][position]``)."""

    def __init__(self, num_layers: int) -> None:
        self.streams: list[list[int]] = [[] for _ in range(num_layers)]

    def __getitem__(self, layer: int) -> list[int]:
        return self.streams[layer]

    def push(self, layer: int, token: int) -> None:
        self.streams[layer].append(int(token))


class _ExportUnpickler(pickle.Unpickler):
    """``run_caer.py`` runs as ``__main__``, so its wrapper classes pickle under that name."""

    def find_class(self, module: str, name: str):
        if module == "__main__" and name in ("ScoredModel", "AdditiveModel"):
            import run_caer

            return getattr(run_caer, name)
        return super().find_class(module, name)


@dataclass
class PolicyBundle:
    """Everything ``run_caer.py --export-models`` wrote, with derived accessors."""

    payload: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path) -> "PolicyBundle":
        path = Path(path).resolve()
        with path.open("rb") as handle:
            payload = _ExportUnpickler(handle).load()
        if payload.get("schema_version") != "caer_policy_export_v1":
            raise ValueError(f"{path} is not a caer_policy_export_v1 bundle")
        return cls(payload, path)

    @property
    def num_layers(self) -> int:
        return int(self.payload["architecture"]["num_layers"])

    @property
    def num_experts(self) -> int:
        return int(self.payload["architecture"]["num_experts"])

    @property
    def top_k(self) -> int:
        return int(self.payload["architecture"]["top_k"])

    @property
    def layer_indices(self) -> list[int]:
        return [int(v) for v in self.payload["architecture"]["layer_indices"]]

    @property
    def capacities(self) -> list[int]:
        return [int(v) for v in self.payload["capacities"]]

    @property
    def configuration(self) -> dict[str, Any]:
        return dict(self.payload["configuration"])

    def transition_models(self) -> MatrixModels:
        return MatrixModels(self.payload["transition"], tuple(self.payload["transition_horizons"]))

    def static(self) -> np.ndarray:
        return np.asarray(self.payload["static"], dtype=np.float64)

    def feature_state_class(self):
        """The exported state: OLMoE's ``FeatureState3`` or the top-k-general variant."""

        from run_caer import causal_token_state_class

        if self.top_k == 8:
            from features3 import FeatureState3 as base
        else:
            from crossmodel import feature_state_class

            base = feature_state_class(self.top_k)
        return causal_token_state_class(base)

    def new_state(self, tokens: LiveTokens):
        state = self.feature_state_class()(
            self.transition_models(), self.num_layers, self.num_experts, self.static()
        )
        state.attach(
            tokens,
            self.payload["token_index"],
            np.asarray(self.payload["token_current"]),
            np.asarray(self.payload["token_next"]),
        )
        return state


# --------------------------------------------------------------------------- scorers


class GlobalScorer:
    """Retention score for a batch of candidate rows, through one of three backends."""

    def __init__(self, scored_model: Any, backend: str) -> None:
        if isinstance(scored_model, dict):
            self.sign = float(scored_model["sign"])
            self.inner = scored_model["inner"]
        else:
            self.sign = float(scored_model.sign)
            self.inner = scored_model.inner
        self.backend = backend
        if backend == "sklearn":
            self.engine = None
        else:
            from fast_trees import FlatForest

            flat = FlatForest.from_sklearn(self.inner)
            if backend == "numba":
                self.engine = flat.to_numba()
            elif backend == "flat":
                self.engine = flat
            else:
                raise ValueError(f"unknown backend {backend!r}")

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        if self.engine is None:
            return self.sign * self.inner.predict(rows)
        return self.sign * np.asarray(self.engine.predict(rows), dtype=np.float64)


class KeepClassifier:
    """P(KEEP) for stage-2 rows: boosted trees through the fast walker, logistic by hand."""

    def __init__(self, model: Any, backend: str) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import Pipeline

        self.model = model
        self.kind = "gbt" if isinstance(model, HistGradientBoostingClassifier) else "logistic"
        self.engine = None
        if self.kind == "gbt" and backend != "sklearn":
            from fast_trees import FlatForest

            flat = FlatForest.from_sklearn(model)
            self.engine = flat.to_numba() if backend == "numba" else flat
        if self.kind == "logistic":
            if not isinstance(model, Pipeline):
                raise TypeError("logistic boundary model must be the exported pipeline")
            scaler, logistic = model.steps[0][1], model.steps[1][1]
            classes = np.asarray(logistic.classes_)
            positive = np.flatnonzero(classes == 1)
            if positive.size != 1:
                raise RuntimeError("logistic model has no unique KEEP class")
            self.mean = np.asarray(scaler.mean_, dtype=np.float64)
            self.scale = np.asarray(scaler.scale_, dtype=np.float64)
            self.coef = np.asarray(logistic.coef_[0], dtype=np.float64)
            self.intercept = float(logistic.intercept_[0])
        else:
            classes = np.asarray(model.classes_)
            positive = np.flatnonzero(classes == 1)
            if positive.size != 1:
                raise RuntimeError("boundary classifier has no unique KEEP class")
        self.positive = int(positive[0])

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        if self.kind == "logistic":
            z = ((rows - self.mean) / self.scale) @ self.coef + self.intercept
            return 1.0 / (1.0 + np.exp(-z))
        if self.engine is None:
            return self.model.predict_proba(rows)[:, self.positive]
        return self.engine.predict_proba_positive(rows)


# --------------------------------------------------------------------------- policies


@dataclass
class StepResult:
    misses: int
    evicted: np.ndarray
    loaded: np.ndarray
    eviction_decision: bool
    stage2_rows: int = 0
    timings_ns: dict[str, int] = field(default_factory=dict)


class CachePolicy:
    """Per-layer resident sets under mandatory atomic admission of the request.

    Prefetching (``prefetch``) admits guessed experts of a layer before its router fires: each
    non-resident guess takes the slot of the policy's lowest-ranked resident outside the guess.
    A prefetched expert that the layer's next request does not use ranks last at that event's
    eviction decision, behind every other candidate (in guess order among themselves); after the
    event it is an ordinary resident. ``transfers`` counts the demand misses only (the copies on
    the critical path); ``prefetch_transfers`` the prefetch copies.
    """

    name = "base"

    def __init__(self, num_layers: int, num_experts: int, capacity: int, top_k: int) -> None:
        if not top_k <= capacity <= num_experts:
            raise ValueError("need top_k <= capacity <= num_experts")
        self.L, self.E, self.capacity, self.top_k = int(num_layers), int(num_experts), int(capacity), int(top_k)
        self.spare = self.capacity - self.top_k
        self.resident = np.zeros((self.L, self.E), dtype=bool)
        self.position = np.zeros(self.L, dtype=np.int64)
        self.transfers = 0
        self.events = 0
        self.eviction_decisions = 0
        self.prefetched = np.zeros((self.L, self.E), dtype=bool)  # admitted by prefetch, before its event
        self.prefetch_rank = np.zeros((self.L, self.E), dtype=np.int64)
        self.prefetch_transfers = 0
        self.prefetch_hits = 0  # requested experts a prefetch had admitted
        self.prefetch_unused = 0  # prefetched experts the next request did not use
        self.prefetch_evicted_requested = 0  # prefetch victims the next request asked for
        self.prefetch_victims = np.zeros((self.L, self.E), dtype=bool)  # evicted by prefetch, before the event
        self.wrong_guess_rule = "last"  # "last", or "policy": unrequested prefetched experts keep the policy's rank
        # True while a layer has prefetch state awaiting its next event; False keeps every step on the
        # frozen arithmetic (no prefetch bookkeeping on the critical path when nothing was prefetched)
        self.pf_active = [False] * self.L

    def begin_sequence(self) -> None:
        pass

    def resident_experts(self, layer: int) -> np.ndarray:
        return np.flatnonzero(self.resident[layer])

    def step(self, layer: int, token: int, request: np.ndarray, gates: np.ndarray) -> StepResult:
        raise NotImplementedError

    def keep_order(self, layer: int, candidates: np.ndarray) -> np.ndarray:
        """Indices into ``candidates``, the expert to keep first (the policy's own ranking)."""

        raise NotImplementedError

    def _wrong_guesses_last(self, layer: int, candidates: np.ndarray, order: np.ndarray) -> np.ndarray:
        """Move unrequested prefetched candidates behind the others (stable), in guess order."""

        flags = self.prefetched[layer, candidates[order]]
        if not flags.any():
            return order
        wrong = order[flags]
        wrong = wrong[np.argsort(self.prefetch_rank[layer, candidates[wrong]], kind="stable")]
        return np.concatenate((order[~flags], wrong))

    def prefetch(self, layer: int, guess: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        """Admit up to ``limit`` non-resident experts of ``guess`` (best first) at ``layer``.

        Returns (evicted, loaded). Victims are the lowest-ranked residents outside the guess.
        """

        guess = np.asarray(guess, dtype=np.int64)
        if guess.size and guess.min() < 0:
            guess = guess[guess >= 0]
        row = self.resident[layer]
        missing = np.flatnonzero(~row[guess])[: int(limit)]  # positions in the guess, best first
        if not missing.size:
            return _EMPTY, _EMPTY
        want = guess[missing]
        free = self.capacity - int(np.count_nonzero(row))
        need = int(want.size) - free
        evicted = _EMPTY
        if need > 0:
            in_guess = np.zeros(self.E, dtype=bool)
            in_guess[guess] = True
            eligible = np.flatnonzero(row & ~in_guess)
            if eligible.size < need:
                missing = missing[: free + eligible.size]
                want = guess[missing]
                need = int(eligible.size)
            if need > 0:
                order = self.keep_order(layer, eligible)
                evicted = eligible[order[eligible.size - need:]]
                row[evicted] = False
                self.prefetched[layer, evicted] = False
                self.prefetch_victims[layer, evicted] = True
        if want.size:
            row[want] = True
            self.prefetched[layer, want] = True
            self.prefetch_rank[layer, want] = missing
            self.prefetch_transfers += int(want.size)
            self.pf_active[layer] = True
        return evicted, want

    def _apply(self, layer: int, request: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        row = self.resident[layer]
        before = row.copy()
        misses = int(request.size) - int(row[request].sum())
        row.fill(False)
        row[request] = True
        row[keep] = True
        evicted = np.flatnonzero(before & ~row)
        loaded = np.flatnonzero(row & ~before)
        if loaded.size != misses:
            raise RuntimeError("admission accounting mismatch")
        self.transfers += misses
        self.events += 1
        if self.pf_active[layer]:
            prefetched = self.prefetched[layer]
            hit = int(prefetched[request].sum())
            self.prefetch_hits += hit
            self.prefetch_unused += int(prefetched.sum()) - hit
            prefetched.fill(False)
            victims = self.prefetch_victims[layer]
            self.prefetch_evicted_requested += int(victims[request].sum())
            victims.fill(False)
            self.pf_active[layer] = False
        return evicted, loaded, misses


class SimplePolicy(CachePolicy):
    """``lru`` / ``lfu`` / ``lfu_decay`` / ``static_hotset`` with the frozen tie-breaks."""

    def __init__(self, name: str, num_layers: int, num_experts: int, capacity: int, top_k: int,
                 *, alpha: float = LFU_DECAY_ALPHA, static_scores: np.ndarray | None = None) -> None:
        super().__init__(num_layers, num_experts, capacity, top_k)
        if name not in SIMPLE_VARIANTS:
            raise ValueError(f"unknown simple policy {name!r}")
        self.name = name
        self.alpha = float(alpha)
        self.clock = np.zeros(self.L, dtype=np.int64)
        self.metric = np.zeros((self.L, self.E), dtype=np.float64)
        if name == "lru":
            self.metric.fill(-1.0)
        if name == "static_hotset":
            if static_scores is None:
                raise ValueError("static_hotset needs calibration scores shaped [layer, expert]")
            scores = np.asarray(static_scores, dtype=np.float64)
            if scores.shape != (self.L, self.E):
                raise ValueError("static scores must be shaped [layer, expert]")
            self.metric = scores.copy()

    def keep_order(self, layer: int, candidates: np.ndarray) -> np.ndarray:
        """The frozen order: larger metric first, ties to the smaller expert id."""

        return np.lexsort((candidates, -self.metric[layer, candidates]))

    def step(self, layer: int, token: int, request: np.ndarray, gates: np.ndarray) -> StepResult:
        start = time.perf_counter_ns()
        request = np.asarray(request, dtype=np.int64)
        self.clock[layer] += 1
        metric = self.metric[layer]
        if self.name == "lru":
            metric[request] = float(self.clock[layer])
        elif self.name == "lfu":
            metric[request] += 1.0
        elif self.name == "lfu_decay":
            metric *= self.alpha
            metric[request] += 1.0
        row = self.resident[layer]
        mask = np.zeros(self.E, dtype=bool)
        mask[request] = True
        old = np.flatnonzero(row & ~mask)
        decision = old.size > self.spare
        if decision:
            order = np.lexsort((old, -metric[old]))  # = keep_order(layer, old)
            if self.pf_active[layer] and self.wrong_guess_rule == "last":
                order = self._wrong_guesses_last(layer, old, order)
            keep = old[order[: self.spare]]
            self.eviction_decisions += 1
        else:
            keep = old
        evicted, loaded, misses = self._apply(layer, request, keep)
        self.position[layer] += 1
        return StepResult(misses, evicted, loaded, decision,
                          timings_ns={"policy": time.perf_counter_ns() - start})


class LearnedPolicy(CachePolicy):
    """Global token-GBT ordering, optionally refined by the stage-2 KEEP classifier.

    ``variant`` is ``global_token_gbt`` (stage 1 only), ``caer_gbt`` or ``caer_logistic``
    (stage 1 + the exported classifier under the exported scope/rerank settings).
    ``score_scope`` is ``candidates`` (score only resident, unrequested experts — the
    only scores the decision reads) or ``all`` (score every expert, as the batched
    simulator does); the decisions are identical.
    """

    def __init__(self, bundle: PolicyBundle, capacity: int, variant: str, *,
                 backend: str = "numba", score_scope: str = "candidates") -> None:
        super().__init__(bundle.num_layers, bundle.num_experts, capacity, bundle.top_k)
        if variant not in LEARNED_VARIANTS:
            raise ValueError(f"unknown learned variant {variant!r}")
        if capacity not in bundle.capacities:
            raise ValueError(f"bundle has no global model for capacity {capacity}")
        if score_scope not in ("candidates", "all"):
            raise ValueError("score_scope must be 'candidates' or 'all'")
        self.name = variant
        self.bundle = bundle
        self.variant = variant
        self.score_scope = score_scope
        self.config = bundle.configuration
        self.tokens = LiveTokens(self.L)
        self.state = bundle.new_state(self.tokens)
        self.state.reset()
        self.global_scorer = GlobalScorer(bundle.payload["global_models"][capacity], backend)
        self.classifier = (
            None if variant == "global_token_gbt"
            else KeepClassifier(bundle.payload["boundary_models"][variant], backend)
        )
        self.last_used = np.full((self.L, self.E), -1, dtype=np.int64)
        self.request_mask = np.zeros(self.E, dtype=bool)
        self.radius = int(self.config["window_radius"])
        self.scope = str(self.config["stage2_scope"])
        self.rerank_mode = str(self.config["rerank_mode"])
        self.residual_strength = float(self.config["residual_strength"])
        self.set_context = bool(self.config["stage2_set_context"])
        self.rich_context = bool(self.config["stage2_rich_context"])
        if bool(self.config.get("boundary_cross_layer")):
            raise NotImplementedError("cross-layer stage-2 block is not deployed here")
        self.stage2_rows_scored = 0
        self.stage2_invocations = 0

    def begin_sequence(self) -> None:
        self.state.begin_sequence()

    def step(self, layer: int, token: int, request: np.ndarray, gates: np.ndarray) -> StepResult:
        t0 = time.perf_counter_ns()
        request = np.asarray(request, dtype=np.int64)
        gates = np.asarray(gates, dtype=np.float64)
        sorted_request = np.sort(request)
        position = int(self.position[layer])
        self.tokens.push(layer, token)
        block = self.state.features(layer, request, gates, sorted_request, position)
        t1 = time.perf_counter_ns()

        self.request_mask[request] = True
        self.last_used[layer, request] = position
        row = self.resident[layer]
        candidates = np.flatnonzero(row & ~self.request_mask)
        decision = candidates.size > self.spare
        stage2_rows = 0
        t2 = t1
        t3 = t1
        if decision:
            self.eviction_decisions += 1
            if self.score_scope == "all":
                global_score = self.global_scorer(block.T)
            else:
                global_score = np.full(self.E, np.nan)
                global_score[candidates] = self.global_scorer(block[:, candidates].T)
            t2 = time.perf_counter_ns()
            base_order = np.lexsort(
                (candidates, -self.last_used[layer, candidates], -global_score[candidates])
            )
            final_order = base_order
            if self.classifier is not None:
                from run_caer import stage2_window

                selected, _label_keep, normalizer = stage2_window(
                    base_order, self.spare, self.radius, self.scope
                )
                if selected is not None:
                    rows, window = build_boundary_features(
                        block, global_score, candidates, base_order, self.spare, self.capacity,
                        radius=normalizer, num_experts=self.E, window=selected,
                        extra_block=None, rich_context=self.rich_context,
                        set_context=self.set_context,
                    )
                    probability = self.classifier(rows)
                    if self.rerank_mode == "residual":
                        probability = probability - 0.5
                    window_scores = combine_boundary_scores(
                        global_score[candidates[window]], probability,
                        mode=self.rerank_mode, strength=self.residual_strength,
                    )
                    final_order = rerank_boundary(base_order, window, window_scores)
                    stage2_rows = int(window.size)
                    self.stage2_rows_scored += stage2_rows
                    self.stage2_invocations += 1
            t3 = time.perf_counter_ns()
            keep = candidates[final_order[: self.spare]]
        else:
            keep = candidates
        evicted, loaded, misses = self._apply(layer, request, keep)
        self.request_mask[request] = False
        self.state.absorb(layer, request, gates, position)
        self.position[layer] += 1
        t4 = time.perf_counter_ns()
        return StepResult(
            misses, evicted, loaded, decision, stage2_rows,
            timings_ns={
                "features": t1 - t0,
                "global_score": t2 - t1,
                "stage2": t3 - t2,
                "bookkeeping": t4 - t3,
                "policy": t4 - t0,
            },
        )


def make_policy(name: str, *, num_layers: int, num_experts: int, capacity: int, top_k: int,
                bundle: PolicyBundle | None = None, backend: str = "numba",
                score_scope: str = "candidates", static_scores: np.ndarray | None = None,
                alpha: float = LFU_DECAY_ALPHA) -> CachePolicy:
    if name in LEARNED_VARIANTS:
        if bundle is None:
            raise ValueError(f"{name} needs an exported policy bundle")
        return LearnedPolicy(bundle, capacity, name, backend=backend, score_scope=score_scope)
    if name in SIMPLE_VARIANTS:
        scores = static_scores
        if name == "static_hotset" and scores is None and bundle is not None:
            scores = bundle.static()
        return SimplePolicy(name, num_layers, num_experts, capacity, top_k,
                            alpha=alpha, static_scores=scores)
    raise ValueError(f"unknown policy {name!r}")
