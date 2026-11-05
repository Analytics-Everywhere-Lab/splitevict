"""Fit and evaluate the learned ranker and the second-stage reranker in the trace simulator.

The learned ranker (``global_token_gbt``) is a gradient-boosted tree over 50 causal routing
and token features. The second-stage reranker (``caer_gbt``; CAER, cutoff-aware expert
reranking) rescores the w=2 candidates on each side of the ranker's eviction cutoff.
``--phase development`` fits on calA (calibration sequences 0--39) and deploys on calB
(40--79); ``reverse`` swaps them; ``evaluation`` fits on all calibration sequences. The
runtime imports this module's feature state and loads the models from ``--export-models``.

    python src/run_caer.py --repository-root measurement --phase development
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from caer import (
    boundary_feature_names,
    extract_boundary_window,
    build_boundary_features,
    combine_boundary_scores,
    offline_keep_labels,
    rerank_boundary,
    selected_keep_labels,
)


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPACITIES = (24, 32)
DEFAULT_RADIUS = 2
DISTANCE_CAP = 33
_EVALUATION_WORKER_CONTEXT: dict[str, Any] | None = None


def configure_repository(root: Path) -> Path:
    """Put the measurement codebase's simulator packages on ``sys.path``; nothing is written."""

    repository = root.expanduser().resolve()
    required = (
        repository / "stage3_residency/traces/full/routing_trace.metadata.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "--repository-root is not the measurement codebase with its data; missing: "
            + ", ".join(missing)
        )
    paths = (
        "stage3_residency/stage3_ranking/src",
        "stage3_residency/stage2_race/src",
        "stage3_residency/stage1_prediction/src",
        "stage3_residency/src",
        "src",
    )
    for relative in reversed(paths):
        value = str(repository / relative)
        if value not in sys.path:
            sys.path.insert(0, value)
    return repository


def causal_token_state_class(base=None):
    """Create the deployable 45+5 feature state with no look-ahead feature access.

    ``base`` defaults to the OLMoE ``features3.FeatureState3``; the cross-model
    path passes ``crossmodel.feature_state_class(top_k)``, which is the same class with
    the request-scope rate scale read from the trace instead of hard-coded to 8.
    """

    if base is None:
        from features3 import FeatureState3 as base

    class CausalTokenFeatureState(base):
        def attach(self, tokens, index, current, nxt) -> None:
            self.tokens = tokens
            self.token_index = index
            self.token_current = current
            self.token_next = nxt
            self.prior = self.static / np.maximum(
                self.static.sum(axis=1, keepdims=True), 1.0
            )

        def begin_sequence(self) -> None:
            super().begin_sequence()
            self.token_last_position = [dict() for _ in range(self.L)]
            self.token_last_request = [dict() for _ in range(self.L)]

        def features(self, layer, request, gates, sorted_request, position):
            base = super().features(layer, request, gates, sorted_request, position)
            token = int(self.tokens[layer][position])
            previous_position = self.token_last_position[layer].get(token)
            previous_request = self.token_last_request[layer].get(token)
            repeated = 0.0 if previous_position is None else 1.0
            gap = (
                np.log1p(1024.0)
                if previous_position is None
                else np.log1p(position - previous_position)
            )
            previous_vector = (
                np.zeros(self.E, dtype=np.float64)
                if previous_request is None
                else previous_request
            )
            row = self.token_index.get(token)
            if row is None:
                current_propensity = self.prior[layer]
                next_propensity = self.prior[layer]
            else:
                current_propensity = self.token_current[row, layer]
                next_propensity = self.token_next[row, layer]
            return np.concatenate(
                (
                    base,
                    np.stack(
                        (
                            np.full(self.E, repeated),
                            np.full(self.E, gap),
                            previous_vector,
                            current_propensity,
                            next_propensity,
                        )
                    ),
                )
            )

        def absorb(self, layer, request, gates, position) -> None:
            super().absorb(layer, request, gates, position)
            token = int(self.tokens[layer][position])
            vector = np.zeros(self.E, dtype=np.float64)
            vector[request] = 1.0
            self.token_last_position[layer][token] = position
            self.token_last_request[layer][token] = vector

    return CausalTokenFeatureState


def stage2_window(order, keep_count: int, radius: int, scope: str):
    """Return the stage-2 candidate set, its label KEEP count, and the normalizer.

    ``window`` is the reranker's normal scope: the symmetric radius window around the
    predicted cutoff. ``all`` applies the identical stage-2 model to every contestable
    candidate, as a comparison arm ("global-rich").
    """

    ranked = np.asarray(order, dtype=np.int64)
    keep = int(keep_count)
    if scope == "all":
        return ranked.copy(), keep, max(keep, ranked.size - keep, 1)
    if scope != "window":
        raise ValueError("scope must be 'window' or 'all'")
    selected = extract_boundary_window(ranked, keep, int(radius))
    if selected is None:
        return None, 0, int(radius)
    return selected, int(radius), int(radius)


def group_slices(groups: np.ndarray) -> Iterable[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(groups)) + 1
    return zip(np.r_[0, edges], np.r_[edges, groups.size])


@dataclass
class ScoredModel:
    """Wrap a fitted estimator so ``predict`` always returns a retention score.

    Higher means keep.  ``sign`` is -1 for targets where a small value means soon
    reuse (rank of next-use distance) and +1 for targets that already point the
    right way (probability that this candidate belongs in the KEEP set).
    """

    inner: Any
    sign: float
    target: str

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return self.sign * self.inner.predict(rows)


def rank_target(target: np.ndarray, groups: np.ndarray) -> np.ndarray:
    output = np.empty(target.shape[0], dtype=np.float64)
    for lower, upper in group_slices(groups):
        size = upper - lower
        output[lower:upper] = np.argsort(np.argsort(target[lower:upper])) / max(
            size - 1, 1
        )
    return output


@dataclass
class BoundaryDataset:
    capacity: int
    feature_names: tuple[str, ...]
    feature_blocks: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    groups: list[np.ndarray] = field(default_factory=list)
    keeps: list[int] = field(default_factory=list)
    eligible_events: int = 0
    ineligible_events: int = 0
    eviction_events: int = 0

    def append(self, features: np.ndarray, labels: np.ndarray, keep: int) -> None:
        if features.shape[0] != labels.size:
            raise ValueError("boundary feature rows and labels do not align")
        group = self.eligible_events
        self.feature_blocks.append(features.astype(np.float32))
        self.labels.append(labels.astype(np.uint8))
        self.groups.append(np.full(labels.size, group, dtype=np.int32))
        self.keeps.append(int(keep))
        self.eligible_events += 1

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.feature_blocks:
            raise RuntimeError(f"No eligible boundary windows at capacity {self.capacity}")
        return (
            np.concatenate(self.feature_blocks),
            np.concatenate(self.labels),
            np.concatenate(self.groups),
        )

    def summary(self) -> dict[str, Any]:
        rows = sum(int(block.size) for block in self.labels)
        positives = sum(int(block.sum()) for block in self.labels)
        return {
            "capacity": self.capacity,
            "rows": rows,
            "groups": self.eligible_events,
            "positive_rows": positives,
            "positive_fraction": positives / rows if rows else None,
            "eviction_events": self.eviction_events,
            "ineligible_incomplete_window": self.ineligible_events,
            "eligible_fraction": (
                self.eligible_events / self.eviction_events if self.eviction_events else None
            ),
        }


@dataclass
class PolicyCounters:
    misses: int = 0
    ranking_concordant: int = 0
    ranking_comparable: int = 0
    eviction_decisions: int = 0
    boundary_invocations: int = 0
    boundary_candidates_scored: int = 0
    boundary_correct: int = 0
    boundary_total: int = 0
    boundary_exact: int = 0
    boundary_events: int = 0
    boundary_scoring_seconds: float = 0.0

    def as_dict(self, total_events: int, num_experts: int) -> dict[str, Any]:
        return {
            "expert_transfers": self.misses,
            "ranking_concordant_pairs": self.ranking_concordant,
            "aggregate_applied_ranking_accuracy": (
                self.ranking_concordant / self.ranking_comparable
                if self.ranking_comparable
                else None
            ),
            "ranking_comparable_pairs": self.ranking_comparable,
            "boundary_correct_candidates": self.boundary_correct,
            "boundary_total_candidates": self.boundary_total,
            "boundary_candidate_accuracy": (
                self.boundary_correct / self.boundary_total if self.boundary_total else None
            ),
            "boundary_exact_events": self.boundary_exact,
            "boundary_exact_event_accuracy": (
                self.boundary_exact / self.boundary_events if self.boundary_events else None
            ),
            "boundary_events_measured": self.boundary_events,
            "eviction_decisions": self.eviction_decisions,
            "stage2_invocations": self.boundary_invocations,
            "stage2_invocation_fraction_of_evictions": (
                self.boundary_invocations / self.eviction_decisions
                if self.eviction_decisions
                else None
            ),
            "stage1_candidates_scored": int(total_events * num_experts),
            "stage2_candidates_scored": self.boundary_candidates_scored,
            "stage2_candidates_per_event": (
                self.boundary_candidates_scored / total_events if total_events else 0.0
            ),
            "measured_boundary_scoring_seconds": self.boundary_scoring_seconds,
        }


def make_state_factory(
    state_type,
    models,
    trace,
    static: np.ndarray,
    token_index: Mapping[int, int],
    token_current: np.ndarray,
    token_next: np.ndarray,
    layer_streams,
):
    def build(workload):
        tokens, _requests, _gates = layer_streams(trace, workload)
        state = state_type(models, trace.num_layers, trace.num_experts, static)
        state.attach(tokens, token_index, token_current, token_next)
        return state

    return build


def fit_global_models(
    trace,
    workload,
    capacities: Sequence[int],
    state_factory,
    *,
    base_feature_count: int,
    seed_weights: np.ndarray,
    simulate_multi,
    stride: int,
    max_iter: int,
) -> tuple[dict[int, Any], dict[int, int], float]:
    """Fit the learned ranker (global token-GBT), one model per capacity.

    Training rows come from replaying ``workload`` under the seed linear scorer. The
    target is the rank of the next-use distance within each eviction decision.
    """

    from sklearn.ensemble import HistGradientBoostingRegressor

    start = time.perf_counter()
    buckets = [[[], [], [], 0] for _ in capacities]
    scorer = lambda features: seed_weights @ features[:base_feature_count]
    simulate_multi(
        trace,
        workload,
        capacities,
        state_factory(workload),
        [scorer] * len(capacities),
        collect_to=buckets,
        stride=stride,
    )
    fitted: dict[int, Any] = {}
    row_counts: dict[int, int] = {}
    for index, capacity in enumerate(capacities):
        features = np.concatenate(buckets[index][0]).astype(np.float64)
        target = np.concatenate(buckets[index][1]).astype(np.float64)
        groups = np.concatenate(buckets[index][2])
        model = HistGradientBoostingRegressor(
            max_leaf_nodes=127,
            max_iter=max_iter,
            learning_rate=0.06,
            l2_regularization=1.0,
            early_stopping=False,
            min_samples_leaf=40,
            random_state=0,
        )
        labels = rank_target(target, groups)
        model.fit(features, labels)
        fitted[int(capacity)] = ScoredModel(model, -1.0, "rank")
        row_counts[int(capacity)] = int(features.shape[0])
        print(
            f"  fitted global token-GBT B={capacity}: {features.shape[0]:,} rows, "
            f"{features.shape[1]} causal features, target=rank",
            flush=True,
        )
        buckets[index] = None
        del features, target, groups
    return fitted, row_counts, time.perf_counter() - start


def feature_batches(trace, workload, state):
    """Yield one decode sequence of causal feature blocks at a time."""

    layers = tuple(map(int, trace.metadata["layer_indices"]))
    layer_to_ordinal = {layer: ordinal for ordinal, layer in enumerate(layers)}
    requests = trace.requested_expert_ids.astype(np.int64, copy=False)
    sorted_requests = np.sort(requests, axis=1)
    gates = trace.router_weights.astype(np.float64, copy=False)
    positions = np.zeros(trace.num_layers, dtype=np.int64)
    state.reset()
    for sequence, view in workload.iter_slices(trace):
        state.begin_sequence()
        count = view.stop - view.start
        blocks = None
        ordinals = np.empty(count, dtype=np.int16)
        steps = np.empty(count, dtype=np.int64)
        indices = np.arange(view.start, view.stop, dtype=np.int64)
        for offset, index in enumerate(indices):
            ordinal = layer_to_ordinal[int(trace.layer_index[index])]
            step = int(positions[ordinal])
            block = state.features(
                ordinal,
                requests[index],
                gates[index],
                sorted_requests[index],
                step,
            )
            if blocks is None:
                blocks = np.empty(
                    (count, block.shape[0], trace.num_experts), dtype=np.float64
                )
            blocks[offset] = block
            ordinals[offset] = ordinal
            steps[offset] = step
            state.absorb(ordinal, requests[index], gates[index], step)
            positions[ordinal] += 1
        if blocks is None:
            continue
        yield sequence, indices, ordinals, steps, blocks


def predict_global_batches(
    models: Mapping[int, Any], blocks: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    events, features, experts = blocks.shape
    rows = blocks.transpose(0, 2, 1).reshape(events * experts, features)
    output: dict[int, np.ndarray] = {}
    elapsed: dict[int, float] = {}
    for capacity, model in models.items():
        start = time.perf_counter()
        output[int(capacity)] = model.predict(rows).reshape(events, experts)
        elapsed[int(capacity)] = time.perf_counter() - start
    return output, elapsed


def collect_boundary_dataset(
    trace,
    workload,
    capacities: Sequence[int],
    state,
    global_models: Mapping[int, Any],
    feature_names: tuple[str, ...],
    *,
    radius: int,
    scope: str = "window",
    set_context: bool = False,
    same_layer_indices,
    build_future_occurrences,
) -> tuple[dict[int, BoundaryDataset], dict[int, float], float]:
    """Replay the fitted learned ranker and collect every eligible boundary window."""

    start = time.perf_counter()
    datasets = {
        int(capacity): BoundaryDataset(int(capacity), feature_names)
        for capacity in capacities
    }
    residents = {
        int(capacity): np.zeros(
            (trace.num_layers, trace.num_experts), dtype=bool
        )
        for capacity in capacities
    }
    last_used = np.full(
        (trace.num_layers, trace.num_experts), -1, dtype=np.int64
    )
    request_mask = np.zeros(trace.num_experts, dtype=bool)
    requests = trace.requested_expert_ids.astype(np.int64, copy=False)
    future = build_future_occurrences(
        trace, workload, same_layer_indices(trace, workload)
    )
    global_seconds = {int(capacity): 0.0 for capacity in capacities}

    for _sequence, indices, ordinals, steps, blocks in feature_batches(
        trace, workload, state
    ):
        predictions, elapsed = predict_global_batches(global_models, blocks)
        for capacity in capacities:
            global_seconds[int(capacity)] += elapsed[int(capacity)]
        for offset, index in enumerate(indices):
            ordinal = int(ordinals[offset])
            step = int(steps[offset])
            request = requests[index]
            distance = future.advance(ordinal, step, request)
            request_mask[request] = True
            last_used[ordinal, request] = step
            for capacity in capacities:
                cap = int(capacity)
                spare = cap - trace.top_k
                row = residents[cap][ordinal]
                candidates = np.flatnonzero(row & ~request_mask)
                if candidates.size > spare:
                    dataset = datasets[cap]
                    dataset.eviction_events += 1
                    score = predictions[cap][offset]
                    order = np.lexsort(
                        (candidates, -last_used[ordinal, candidates], -score[candidates])
                    )
                    selected, label_keep, normalizer = stage2_window(
                        order, spare, radius, scope
                    )
                    built = (
                        None
                        if selected is None
                        else build_boundary_features(
                            blocks[offset],
                            score,
                            candidates,
                            order,
                            spare,
                            cap,
                            radius=normalizer,
                            num_experts=trace.num_experts,
                            window=selected,
                            set_context=set_context,
                        )
                    )
                    if built is None:
                        dataset.ineligible_events += 1
                    else:
                        features, window = built
                        experts = candidates[window]
                        labels = offline_keep_labels(
                            experts,
                            distance[experts],
                            last_used[ordinal, experts],
                            keep_count=label_keep,
                            distance_cap=DISTANCE_CAP,
                        )
                        dataset.append(features, labels, label_keep)
                    row.fill(False)
                    row[request] = True
                    row[candidates[order[:spare]]] = True
                else:
                    row[request] = True
            request_mask[request] = False
    future.finish()
    return datasets, global_seconds, time.perf_counter() - start


def dataset_content_sha256(
    datasets: Mapping[int, BoundaryDataset]
) -> str:
    digest = hashlib.sha256()
    for capacity in sorted(datasets):
        dataset = datasets[capacity]
        features, labels, groups = dataset.arrays()
        digest.update(str(capacity).encode())
        digest.update(json.dumps(dataset.feature_names).encode())
        for array in (features, labels, groups):
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def save_boundary_dataset(
    path: Path,
    datasets: Mapping[int, BoundaryDataset],
    radius: int = DEFAULT_RADIUS,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for capacity in sorted(datasets):
        features, labels, groups = datasets[capacity].arrays()
        payload[f"X_B{capacity}"] = features
        payload[f"y_B{capacity}"] = labels
        payload[f"group_B{capacity}"] = groups
    first = datasets[sorted(datasets)[0]]
    payload["feature_names"] = np.asarray(first.feature_names)
    payload["capacities"] = np.asarray(sorted(datasets), dtype=np.int16)
    payload["window_radius"] = np.asarray([int(radius)], dtype=np.int16)
    np.savez_compressed(path, **payload)
    return dataset_content_sha256(datasets)


def boundary_group_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    keep: int | np.ndarray,
) -> dict[str, Any]:
    correct = total = exact = events = 0
    keeps = np.asarray(keep, dtype=np.int64).reshape(-1)
    for index, (lower, upper) in enumerate(group_slices(groups)):
        keep = int(keeps[index]) if keeps.size > 1 else int(keeps[0])
        probability = probabilities[lower:upper]
        truth = labels[lower:upper]
        order = np.lexsort((np.arange(probability.size), -probability))
        prediction = np.zeros(probability.size, dtype=np.uint8)
        prediction[order[:keep]] = 1
        matches = prediction == truth
        correct += int(matches.sum())
        total += int(matches.size)
        exact += int(bool(matches.all()))
        events += 1
    return {
        "candidate_accuracy": correct / total if total else None,
        "exact_event_accuracy": exact / events if events else None,
        "events": events,
    }


def train_boundary_models(
    datasets: Mapping[int, BoundaryDataset], *, max_iter: int, radius: int
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Fit the logistic and GBT KEEP classifiers on the boundary windows.

    One classifier per type is pooled across capacities, with capacity as a feature.
    """

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    start = time.perf_counter()
    features, labels, groups, keeps = [], [], [], []
    offset = 0
    for capacity in sorted(datasets):
        x, y, g = datasets[capacity].arrays()
        features.append(x.astype(np.float64))
        labels.append(y)
        groups.append(g + offset)
        keeps.extend(datasets[capacity].keeps)
        offset += int(g.max()) + 1
    x_all = np.concatenate(features)
    y_all = np.concatenate(labels)
    g_all = np.concatenate(groups)
    keep_all = np.asarray(keeps, dtype=np.int64)
    models: dict[str, Any] = {
        "caer_logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=0),
        ),
        "caer_gbt": HistGradientBoostingClassifier(
            max_leaf_nodes=31,
            max_iter=max_iter,
            learning_rate=0.08,
            l2_regularization=1.0,
            early_stopping=False,
            min_samples_leaf=40,
            random_state=0,
        ),
    }
    diagnostics: dict[str, Any] = {}
    for name, model in models.items():
        model_start = time.perf_counter()
        model.fit(x_all, y_all)
        probability = positive_probability(model, x_all)
        diagnostics[name] = {
            **boundary_group_metrics(probability, y_all, g_all, keep_all),
            "fit_seconds": time.perf_counter() - model_start,
        }
        print(
            f"  fitted {name}: {x_all.shape[0]:,} rows, "
            f"train boundary accuracy={100 * diagnostics[name]['candidate_accuracy']:.2f}%",
            flush=True,
        )
    diagnostics["pooled_rows"] = int(x_all.shape[0])
    diagnostics["features"] = int(x_all.shape[1])
    return models, diagnostics, time.perf_counter() - start


def positive_probability(model, features: np.ndarray) -> np.ndarray:
    probability = model.predict_proba(features)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if positive.size != 1:
        raise RuntimeError("boundary classifier does not expose a unique KEEP class")
    return probability[:, int(positive[0])]


def ordering_stats(
    capped_distances: np.ndarray, order: np.ndarray
) -> tuple[int, int]:
    distance = np.asarray(capped_distances)[np.asarray(order, dtype=np.int64)]
    concordant = comparable = 0
    for index in range(distance.size - 1):
        remainder = distance[index + 1 :]
        concordant += int(np.count_nonzero(distance[index] < remainder))
        comparable += int(np.count_nonzero(distance[index] != remainder))
    return concordant, comparable


def evaluate_workload(
    trace,
    workload,
    capacities: Sequence[int],
    state,
    global_models: Mapping[int, Any],
    boundary_models: Mapping[str, Any],
    *,
    radius: int,
    rerank_mode: str,
    residual_strength: float,
    same_layer_indices,
    build_future_occurrences,
    scope: str = "window",
    set_context: bool = False,
) -> tuple[dict[str, dict[int, PolicyCounters]], dict[int, float], int, list[dict]]:
    methods = ("global_token_gbt",) + tuple(boundary_models)
    residents = {
        method: {
            int(capacity): np.zeros(
                (trace.num_layers, trace.num_experts), dtype=bool
            )
            for capacity in capacities
        }
        for method in methods
    }
    counters = {
        method: {int(capacity): PolicyCounters() for capacity in capacities}
        for method in methods
    }
    requests = trace.requested_expert_ids.astype(np.int64, copy=False)
    last_used = np.full(
        (trace.num_layers, trace.num_experts), -1, dtype=np.int64
    )
    request_mask = np.zeros(trace.num_experts, dtype=bool)
    future = build_future_occurrences(
        trace, workload, same_layer_indices(trace, workload)
    )
    global_seconds = {int(capacity): 0.0 for capacity in capacities}
    total_events = 0
    sequence_rows: list[dict[str, Any]] = []

    for sequence, indices, ordinals, steps, blocks in feature_batches(
        trace, workload, state
    ):
        before = {
            method: {
                int(capacity): counters[method][int(capacity)].misses
                for capacity in capacities
            }
            for method in methods
        }
        predictions, elapsed = predict_global_batches(global_models, blocks)
        for capacity in capacities:
            global_seconds[int(capacity)] += elapsed[int(capacity)]
        total_events += int(indices.size)
        for offset, index in enumerate(indices):
            ordinal = int(ordinals[offset])
            step = int(steps[offset])
            request = requests[index]
            distance = future.advance(ordinal, step, request)
            request_mask[request] = True
            last_used[ordinal, request] = step
            for capacity in capacities:
                cap = int(capacity)
                spare = cap - trace.top_k
                global_score = predictions[cap][offset]
                for method in methods:
                    count = counters[method][cap]
                    row = residents[method][cap][ordinal]
                    count.misses += int(request.size) - int(row[request].sum())
                    candidates = np.flatnonzero(row & ~request_mask)
                    if candidates.size <= spare:
                        row[request] = True
                        continue
                    count.eviction_decisions += 1
                    base_order = np.lexsort(
                        (
                            candidates,
                            -last_used[ordinal, candidates],
                            -global_score[candidates],
                        )
                    )
                    selected, label_keep, normalizer = stage2_window(
                        base_order, spare, radius, scope
                    )
                    built = (
                        None
                        if selected is None
                        else build_boundary_features(
                            blocks[offset],
                            global_score,
                            candidates,
                            base_order,
                            spare,
                            cap,
                            radius=normalizer,
                            num_experts=trace.num_experts,
                            window=selected,
                            set_context=set_context,
                        )
                    )
                    final_order = base_order
                    window = None
                    if built is not None:
                        boundary_features, window = built
                        if method != "global_token_gbt":
                            boundary_start = time.perf_counter()
                            probability = positive_probability(
                                boundary_models[method], boundary_features
                            )
                            if rerank_mode == "residual":
                                probability = probability - 0.5
                            window_scores = combine_boundary_scores(
                                global_score[candidates[window]],
                                probability,
                                mode=rerank_mode,
                                strength=residual_strength,
                            )
                            final_order = rerank_boundary(
                                base_order, window, window_scores
                            )
                            count.boundary_scoring_seconds += (
                                time.perf_counter() - boundary_start
                            )
                            count.boundary_invocations += 1
                            count.boundary_candidates_scored += int(window.size)

                        experts = candidates[window]
                        oracle_labels = offline_keep_labels(
                            experts,
                            distance[experts],
                            last_used[ordinal, experts],
                            keep_count=label_keep,
                            distance_cap=DISTANCE_CAP,
                        )
                        predicted_labels = selected_keep_labels(
                            window, final_order, spare
                        )
                        matches = predicted_labels == oracle_labels
                        count.boundary_correct += int(matches.sum())
                        count.boundary_total += int(matches.size)
                        count.boundary_exact += int(bool(matches.all()))
                        count.boundary_events += 1

                    concordant, comparable = ordering_stats(
                        np.minimum(distance[candidates], DISTANCE_CAP), final_order
                    )
                    count.ranking_concordant += concordant
                    count.ranking_comparable += comparable
                    row.fill(False)
                    row[request] = True
                    row[candidates[final_order[:spare]]] = True
            request_mask[request] = False

        for method in methods:
            for capacity in capacities:
                cap = int(capacity)
                sequence_rows.append(
                    {
                        "workload": workload.name,
                        "regime": workload.regime,
                        "sequence_position": int(sequence.position),
                        "source_sequence_id": int(sequence.source_sequence_id),
                        "domain": str(sequence.domain),
                        "method": method,
                        "capacity": cap,
                        "expert_transfers": counters[method][cap].misses
                        - before[method][cap],
                    }
                )
    future.finish()
    return counters, global_seconds, total_events, sequence_rows


def harness_bindings(prepared_root: Path | None):
    """Return ``(load, references, seed_weights, simulate_multi, state_base)``.

    With no ``prepared_root`` these are the OLMoE ranking modules in ``src/`` (``harness``,
    ``committed``, ``run4``).  With one, they are the ``crossmodel`` equivalents fitted by
    ``crossmodel.prepare`` on another trace's calibration split; see that module for the
    two generalized rules.
    """

    if prepared_root is None:
        from committed import seed_weights
        from harness import load, references
        from run4 import simulate_multi

        return load, references, seed_weights, simulate_multi, None

    # ``crossmodel`` comes from ``src/``, which is on sys.path as the script directory,
    # behind the measurement-codebase paths added by ``configure_repository``.
    import crossmodel

    root = Path(prepared_root).resolve()

    def load():
        return crossmodel.load(root)

    def seed_weights():
        return crossmodel.seed_weights(root)

    def state_base(trace):
        return crossmodel.feature_state_class(trace.top_k)

    return load, crossmodel.references, seed_weights, crossmodel.simulate_multi, state_base


def initialize_evaluation_worker(context: dict[str, Any]) -> None:
    """Load immutable fitted state once in each workload worker."""

    global _EVALUATION_WORKER_CONTEXT
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    configure_repository(Path(context["repository"]))
    _EVALUATION_WORKER_CONTEXT = context


def evaluate_workload_worker(index: int):
    """Evaluate one frozen workload path in an isolated process."""

    context = _EVALUATION_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("evaluation worker was not initialized")
    from race_stage1.models import same_layer_indices
    from race_stage2.diagnostics import build_future_occurrences
    from run8 import layer_streams

    load, references, _seed, _simulate, state_base = harness_bindings(
        context.get("prepared_root")
    )
    inputs, transition_models = load()
    trace = inputs.trace
    workload = inputs.workloads[int(index)]
    state_type = causal_token_state_class(
        None if state_base is None else state_base(trace)
    )
    state_factory = make_state_factory(
        state_type,
        transition_models,
        trace,
        context["static"],
        context["token_index"],
        context["token_current"],
        context["token_next"],
        layer_streams,
    )
    start = time.perf_counter()
    counters, global_seconds, events, sequence_rows = evaluate_workload(
        trace,
        workload,
        context["capacities"],
        state_factory(workload),
        context["global_models"],
        context["boundary_models"],
        radius=context["radius"],
        rerank_mode=context["rerank_mode"],
        residual_strength=context["residual_strength"],
        same_layer_indices=same_layer_indices,
        build_future_occurrences=build_future_occurrences,
        scope=str(context.get("scope", "window")),
        set_context=bool(context.get("set_context")),
    )
    refs = references(inputs, workload, context["capacities"])
    record = workload_record(
        workload,
        counters,
        refs,
        events,
        trace.num_experts,
        global_seconds,
        context["capacities"],
    )
    return int(index), record, sequence_rows, time.perf_counter() - start


def workload_record(
    workload,
    counters: Mapping[str, Mapping[int, PolicyCounters]],
    references: Mapping[str, Any],
    total_events: int,
    num_experts: int,
    global_seconds: Mapping[int, float],
    capacities: Sequence[int],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "workload": workload.name,
        "regime": workload.regime,
        "events": total_events,
        "references": {
            "simple": {str(cap): int(references["simple"][cap]) for cap in capacities},
            "oracle": {str(cap): int(references["oracle"][cap]) for cap in capacities},
            "simple_policy": {
                str(cap): references["simple_which"][cap] for cap in capacities
            },
        },
        "global_batched_scoring_seconds": {
            str(cap): global_seconds[int(cap)] for cap in capacities
        },
        "methods": {},
    }
    for method, by_capacity in counters.items():
        record["methods"][method] = {}
        for capacity in capacities:
            cap = int(capacity)
            simple = int(references["simple"][cap])
            oracle = int(references["oracle"][cap])
            item = by_capacity[cap].as_dict(total_events, num_experts)
            item["oracle_gap_closed"] = (
                (simple - item["expert_transfers"]) / (simple - oracle)
                if simple != oracle
                else None
            )
            record["methods"][method][str(cap)] = item
    return record


def aggregate_records(
    records: Sequence[dict[str, Any]],
    capacities: Sequence[int],
    num_experts: int,
) -> dict[str, Any]:
    methods = tuple(records[0]["methods"])
    total_events = sum(int(record["events"]) for record in records)
    aggregate: dict[str, Any] = {
        "events": total_events,
        "references": {"simple": {}, "oracle": {}},
        "methods": {method: {} for method in methods},
    }
    for capacity in capacities:
        key = str(int(capacity))
        simple = sum(int(record["references"]["simple"][key]) for record in records)
        oracle = sum(int(record["references"]["oracle"][key]) for record in records)
        aggregate["references"]["simple"][key] = simple
        aggregate["references"]["oracle"][key] = oracle
        base_cost = sum(
            int(record["methods"]["global_token_gbt"][key]["expert_transfers"])
            for record in records
        )
        base_gap = (simple - base_cost) / (simple - oracle)
        for method in methods:
            cost = sum(
                int(record["methods"][method][key]["expert_transfers"])
                for record in records
            )
            concordant = sum(
                int(record["methods"][method][key]["ranking_concordant_pairs"])
                for record in records
            )
            comparable = sum(
                int(record["methods"][method][key]["ranking_comparable_pairs"])
                for record in records
            )
            boundary_correct = sum(
                int(record["methods"][method][key]["boundary_correct_candidates"])
                for record in records
            )
            boundary_total = sum(
                int(record["methods"][method][key]["boundary_total_candidates"])
                for record in records
            )
            exact = sum(
                int(record["methods"][method][key]["boundary_exact_events"])
                for record in records
            )
            boundary_events = sum(
                int(record["methods"][method][key]["boundary_events_measured"])
                for record in records
            )
            invocations = sum(
                int(record["methods"][method][key]["stage2_invocations"])
                for record in records
            )
            evictions = sum(
                int(record["methods"][method][key]["eviction_decisions"])
                for record in records
            )
            stage2_candidates = sum(
                int(record["methods"][method][key]["stage2_candidates_scored"])
                for record in records
            )
            boundary_seconds = sum(
                float(record["methods"][method][key]["measured_boundary_scoring_seconds"])
                for record in records
            )
            gap = (simple - cost) / (simple - oracle)
            aggregate["methods"][method][key] = {
                "expert_transfers": cost,
                "relative_transfer_reduction_vs_global": (
                    (base_cost - cost) / base_cost if method != "global_token_gbt" else 0.0
                ),
                "oracle_gap_closed": gap,
                "oracle_gap_points_over_global": gap - base_gap,
                "aggregate_applied_ranking_accuracy": (
                    concordant / comparable if comparable else None
                ),
                "ranking_concordant_pairs": concordant,
                "ranking_comparable_pairs": comparable,
                "boundary_correct_candidates": boundary_correct,
                "boundary_total_candidates": boundary_total,
                "boundary_candidate_accuracy": (
                    boundary_correct / boundary_total if boundary_total else None
                ),
                "boundary_exact_events": exact,
                "boundary_exact_event_accuracy": (
                    exact / boundary_events if boundary_events else None
                ),
                "boundary_events_measured": boundary_events,
                "eviction_decisions": evictions,
                "stage2_invocations": invocations,
                "stage2_invocation_fraction_of_evictions": (
                    invocations / evictions if evictions else None
                ),
                "stage1_candidates_scored": total_events * int(num_experts),
                "stage2_candidates_scored": stage2_candidates,
                "stage2_candidates_per_event": (
                    stage2_candidates / total_events if total_events else 0.0
                ),
                "measured_boundary_scoring_seconds": boundary_seconds,
                "workloads_improved": sum(
                    int(
                        record["methods"][method][key]["expert_transfers"]
                        < record["methods"]["global_token_gbt"][key][
                            "expert_transfers"
                        ]
                    )
                    for record in records
                )
                if method != "global_token_gbt"
                else 0,
                "workload_count": len(records),
            }
    return aggregate


def development_identity(
    aggregate: Mapping[str, Any], capacities: Sequence[int], enabled: bool
) -> dict[str, Any]:
    if not enabled:
        return {"checked": False, "reason": "non-default protocol"}
    artifact = ARTIFACT_ROOT / "results/ranker/run8_deploy.json"
    frozen = json.loads(artifact.read_text())["deployments"]
    rows = {}
    passed = True
    for capacity in capacities:
        key = str(int(capacity))
        observed = int(
            aggregate["methods"]["global_token_gbt"][key]["expert_transfers"]
        )
        expected = int(frozen[f"B{capacity}_token"]["cost"])
        rows[key] = {"observed": observed, "expected": expected, "match": observed == expected}
        passed &= observed == expected
    if not passed:
        raise RuntimeError(
            "The batched CAER harness did not reproduce the frozen global token-GBT cost"
        )
    return {"checked": True, "passed": True, "capacities": rows}


def select_workloads(inputs, sub, phase: str, smoke_sequences: int | None):
    calibration = inputs.calibration
    cal_a = sub(calibration, 0, 40, "calA")
    cal_b = sub(calibration, 40, 80, "calB")
    if phase == "development":
        fit = cal_a
        deploy = [cal_b]
    elif phase == "reverse":
        fit = cal_b
        deploy = [cal_a]
    elif phase == "evaluation":
        fit = calibration
        deploy = list(inputs.workloads)
    else:
        raise ValueError(f"unknown phase {phase}")
    if smoke_sequences is not None:
        if phase == "evaluation":
            raise SystemExit("--smoke-sequences is only supported for calibration phases")
        count = int(smoke_sequences)
        if count <= 0:
            raise SystemExit("--smoke-sequences must be positive")
        fit = sub(fit, 0, min(count, len(fit.sequences)), f"{fit.name}_smoke")
        target = deploy[0]
        deploy = [
            sub(
                target,
                0,
                min(count, len(target.sequences)),
                f"{target.name}_smoke",
            )
        ]
    return fit, deploy


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("development", "reverse", "evaluation"), default="development"
    )
    parser.add_argument("--capacities", nargs="+", type=int, default=list(DEFAULT_CAPACITIES))
    parser.add_argument("--window-radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--global-stride", type=int, default=3)
    parser.add_argument("--global-max-iter", type=int, default=800)
    parser.add_argument("--boundary-max-iter", type=int, default=250)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="evaluation workload workers (0 = 10 for evaluation, 1 otherwise)",
    )
    parser.add_argument(
        "--rerank-mode", choices=("replacement", "residual"), default="replacement"
    )
    parser.add_argument("--residual-strength", type=float, default=1.0)
    parser.add_argument(
        "--stage2-set-context",
        action="store_true",
        help="append per-feature deviation from the candidate-set mean, making the "
        "stage-2 scorer multivariate rather than univariate",
    )
    parser.add_argument(
        "--stage2-scope",
        choices=("window", "all"),
        default="window",
        help="apply the stage-2 model to the cutoff window (default) or to every candidate",
    )
    parser.add_argument("--smoke-sequences", type=int)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        help="run on a non-OLMoE trace prepared by crossmodel.py (split, transition "
        "models and seed weights are read from here instead of the measurement codebase)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument(
        "--export-models",
        type=Path,
        help="pickle the fitted global and stage-2 models plus every input the "
        "deployed policy needs (read by src/runtime/policy.py); written after fitting, "
        "before deployment",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="stop after fitting (and exporting); write no result file",
    )
    return parser.parse_args(argv)


def export_fitted_models(
    path: Path,
    *,
    args: argparse.Namespace,
    trace,
    fit_workload,
    transition_models,
    static: np.ndarray,
    token_index: Mapping[int, int],
    token_current: np.ndarray,
    token_next: np.ndarray,
    global_models: Mapping[int, Any],
    boundary_models: Mapping[str, Any],
    causal_names: Sequence[str],
    boundary_names: Sequence[str],
    trace_record: dict[str, Any] | None,
) -> None:
    """Pickle the deployed policy: models, tables and configuration.

    The transition models are stored as the stacked ``[layer, horizon, from, to]``
    array the feature state builds from them, so loading needs no measurement-codebase class.
    """

    import pickle

    from features2 import HORIZONS

    layers = int(trace.num_layers)
    transition = np.stack(
        [
            np.stack([transition_models.matrix(h, layer) for h in HORIZONS])
            for layer in range(layers)
        ]
    )
    payload = {
        "schema_version": "caer_policy_export_v1",
        "phase": args.phase,
        "fit_workload": fit_workload.name,
        "fit_sequences": len(fit_workload.sequences),
        "model": (
            str(trace.metadata.get("model"))
            if trace_record is None
            else trace_record["model"]
        ),
        "trace_hash": str(trace.trace_hash),
        "trace_record": trace_record,
        "architecture": {
            "num_layers": layers,
            "num_experts": int(trace.num_experts),
            "top_k": int(trace.top_k),
            "layer_indices": [int(v) for v in trace.metadata["layer_indices"]],
        },
        "capacities": [int(v) for v in args.capacities],
        "configuration": {
            "window_radius": int(args.window_radius),
            "stage2_scope": args.stage2_scope,
            "rerank_mode": args.rerank_mode,
            "residual_strength": float(args.residual_strength),
            "stage2_set_context": bool(args.stage2_set_context),
            "stage2_rich_context": False,  # src/runtime/policy.py reads this key
            "distance_cap": DISTANCE_CAP,
        },
        "causal_feature_names": list(causal_names),
        "boundary_feature_names": list(boundary_names),
        "transition_horizons": list(HORIZONS),
        "transition": transition,
        "static": np.asarray(static, dtype=np.float64),
        "token_index": dict(token_index),
        "token_current": np.asarray(token_current),
        "token_next": np.asarray(token_next),
        "global_models": {
            int(k): {"inner": v.inner, "sign": float(v.sign), "target": str(v.target)}
            for k, v in global_models.items()
        },
        "boundary_models": dict(boundary_models),
    }
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"  exported fitted policy to {path} (sha256 {digest[:12]}...)", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.window_radius < 1:
        raise SystemExit("--window-radius must be a positive integer")
    capacities = tuple(int(value) for value in args.capacities)
    if len(set(capacities)) != len(capacities):
        raise SystemExit("capacities must be unique")
    cross_model = args.prepared_root is not None
    if not cross_model and any(value not in (12, 16, 24, 32) for value in capacities):
        raise SystemExit("supported non-degenerate capacities are 12, 16, 24, and 32")

    repository = configure_repository(args.repository_root)
    from collect2 import sub
    from features3 import NAMES as BASE_NAMES
    from race_stage1.models import same_layer_indices
    from race_stage2.diagnostics import build_future_occurrences
    from residency_headroom.workloads import calibration_frequency_scores
    from run8 import TOKEN_NAMES, fit_token_tables, layer_streams

    load, references, seed_weights, simulate_multi, state_base = harness_bindings(
        args.prepared_root
    )
    inputs, transition_models = load()
    trace = inputs.trace
    if cross_model and any(
        not (trace.top_k < value < trace.num_experts) for value in capacities
    ):
        raise SystemExit(
            f"cross-model capacities must satisfy top_k={trace.top_k} < B < "
            f"E={trace.num_experts}"
        )
    state_type = causal_token_state_class(
        None if state_base is None else state_base(trace)
    )
    trace_record = None
    if cross_model:
        prepared = inputs.prepared
        trace_record = {
            "model": prepared["model"],
            "trace_root": prepared["trace_root"],
            "trace_hash": prepared["trace_hash"],
            "architecture": prepared["architecture"],
            "events": prepared["events"],
            "prepared_root": str(Path(args.prepared_root).resolve()),
            "frozen_evaluation_config_hash": prepared["frozen_evaluation_config_hash"],
            "seed_weights_sha256": prepared["seed_weights_sha256"],
            "transition_model_sha256": prepared["transition_models"].get("npz_sha256"),
            "generalisations": prepared["generalisations"],
        }
        print(
            f"cross-model trace {prepared['model']}: E={trace.num_experts} "
            f"k={trace.top_k} L={trace.num_layers}, {trace.num_events:,} events, "
            f"trace_hash {trace.trace_hash[:12]}...",
            flush=True,
        )
    fit_workload, deploy_workloads = select_workloads(
        inputs, sub, args.phase, args.smoke_sequences
    )
    print(
        f"CAER(w=2) phase={args.phase}: fit={fit_workload.name} "
        f"({len(fit_workload.sequences)} sequences), deploy="
        + ", ".join(
            f"{workload.name}({len(workload.sequences)})" for workload in deploy_workloads
        ),
        flush=True,
    )

    static = calibration_frequency_scores(trace, fit_workload).astype(np.float64)
    token_start = time.perf_counter()
    token_index, token_current, token_next = fit_token_tables(
        trace,
        fit_workload,
        trace.num_layers,
        trace.num_experts,
        static,
    )
    token_seconds = time.perf_counter() - token_start
    state_factory = make_state_factory(
        state_type,
        transition_models,
        trace,
        static,
        token_index,
        token_current,
        token_next,
        layer_streams,
    )
    causal_names = tuple(BASE_NAMES) + tuple(TOKEN_NAMES)
    if len(causal_names) != 50:
        raise RuntimeError(f"Expected 50 causal global features, found {len(causal_names)}")
    complete_boundary_names = boundary_feature_names(
        causal_names, set_context=args.stage2_set_context
    )
    print(
        f"  fitted token tables: {len(token_index):,} token ids [{token_seconds:.1f}s]",
        flush=True,
    )

    global_models, global_rows, global_fit_seconds = fit_global_models(
        trace,
        fit_workload,
        capacities,
        state_factory,
        base_feature_count=len(BASE_NAMES),
        seed_weights=seed_weights(),
        simulate_multi=simulate_multi,
        stride=args.global_stride,
        max_iter=args.global_max_iter,
    )
    print(f"  global fitting complete [{global_fit_seconds:.1f}s]", flush=True)

    datasets, boundary_collection_global_seconds, collection_seconds = (
        collect_boundary_dataset(
            trace,
            fit_workload,
            capacities,
            state_factory(fit_workload),
            global_models,
            complete_boundary_names,
            radius=args.window_radius,
            scope=args.stage2_scope,
            set_context=args.stage2_set_context,
            same_layer_indices=same_layer_indices,
            build_future_occurrences=build_future_occurrences,
        )
    )
    for capacity in capacities:
        summary = datasets[int(capacity)].summary()
        print(
            f"  boundary data B={capacity}: {summary['groups']:,} windows, "
            f"{summary['rows']:,} rows, eligible on "
            f"{100 * summary['eligible_fraction']:.2f}% of eviction decisions",
            flush=True,
        )

    results_root = (
        ARTIFACT_ROOT / "results"
        if not cross_model
        else Path(args.prepared_root).resolve().parent
    )
    default_output = results_root / f"caer_{args.phase}.json"
    output_path = (args.output or default_output).resolve()
    radius_suffix = (
        "" if args.window_radius == DEFAULT_RADIUS else f"_w{args.window_radius}"
    )
    default_dataset = (
        results_root
        / f"caer_boundary_train_{fit_workload.name}{radius_suffix}.npz"
    )
    dataset_path = (args.dataset_output or default_dataset).resolve()
    data_start = time.perf_counter()
    content_hash = save_boundary_dataset(dataset_path, datasets, args.window_radius)
    dataset_save_seconds = time.perf_counter() - data_start
    print(
        f"  saved boundary dataset {dataset_path} "
        f"(content sha256 {content_hash[:12]}...) [{dataset_save_seconds:.1f}s]",
        flush=True,
    )

    boundary_models, boundary_fit, boundary_fit_seconds = train_boundary_models(
        datasets, max_iter=args.boundary_max_iter, radius=args.window_radius
    )
    print(f"  boundary fitting complete [{boundary_fit_seconds:.1f}s]", flush=True)

    if args.export_models is not None:
        export_fitted_models(
            args.export_models,
            args=args,
            trace=trace,
            fit_workload=fit_workload,
            transition_models=transition_models,
            static=static,
            token_index=token_index,
            token_current=token_current,
            token_next=token_next,
            global_models=global_models,
            boundary_models=boundary_models,
            causal_names=causal_names,
            boundary_names=complete_boundary_names,
            trace_record=trace_record,
        )
    if args.fit_only:
        print("  --fit-only: no deployment, no result file written", flush=True)
        return

    records = []
    all_sequence_rows: list[dict[str, Any]] = []
    evaluation_seconds = 0.0

    def report_workload(record: dict[str, Any], elapsed: float) -> None:
        workload_name = record["workload"]
        for capacity in capacities:
            cap = str(int(capacity))
            base = record["methods"]["global_token_gbt"][cap]
            logistic = record["methods"]["caer_logistic"][cap]
            gbt = record["methods"]["caer_gbt"][cap]
            print(
                f"    B={capacity}: global={base['expert_transfers']:,}; "
                f"CAER-log={logistic['expert_transfers']:,} "
                f"({100 * (base['expert_transfers'] - logistic['expert_transfers']) / base['expert_transfers']:+.3f}%); "
                f"CAER-GBT={gbt['expert_transfers']:,} "
                f"({100 * (base['expert_transfers'] - gbt['expert_transfers']) / base['expert_transfers']:+.3f}%)",
                flush=True,
            )
        print(f"    completed {workload_name} [{elapsed:.1f}s]", flush=True)

    requested_workers = int(args.workers)
    workers = (
        min(10, len(deploy_workloads))
        if requested_workers == 0 and args.phase == "evaluation"
        else (1 if requested_workers == 0 else requested_workers)
    )
    if workers <= 0:
        raise SystemExit("--workers must be positive or zero for automatic selection")

    if workers > 1 and args.phase == "evaluation":
        print(
            f"  evaluating {len(deploy_workloads)} frozen workload paths with "
            f"{workers} workers...",
            flush=True,
        )
        context = {
            "repository": str(repository),
            "prepared_root": (
                None if args.prepared_root is None else str(args.prepared_root.resolve())
            ),
            "capacities": capacities,
            "static": static,
            "token_index": token_index,
            "token_current": token_current,
            "token_next": token_next,
            "global_models": global_models,
            "boundary_models": boundary_models,
            "radius": args.window_radius,
            "rerank_mode": args.rerank_mode,
            "residual_strength": args.residual_strength,
            "scope": args.stage2_scope,
            "set_context": args.stage2_set_context,
        }
        by_index: dict[int, tuple[dict[str, Any], list[dict[str, Any]], float]] = {}
        # Spawned workers inherit this environment at process start, before their
        # OpenMP runtime initializes; ``initialize_evaluation_worker`` runs too late for
        # that.  One thread per worker: ten workers at the parent's thread count thrash
        # in spin-wait/yield and spend ~90% of their time in the kernel.
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = "1"
        wall_start = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_evaluation_worker,
            initargs=(context,),
        ) as pool:
            futures = {
                pool.submit(evaluate_workload_worker, index): index
                for index in range(len(deploy_workloads))
            }
            for future in as_completed(futures):
                index, record, sequence_rows, elapsed = future.result()
                by_index[index] = (record, sequence_rows, elapsed)
                report_workload(record, elapsed)
        evaluation_seconds = time.perf_counter() - wall_start
        for index in range(len(deploy_workloads)):
            record, sequence_rows, _elapsed = by_index[index]
            records.append(record)
            all_sequence_rows.extend(sequence_rows)
    else:
        for workload in deploy_workloads:
            print(f"  evaluating {workload.name}...", flush=True)
            evaluation_start = time.perf_counter()
            counters, global_seconds, events, sequence_rows = evaluate_workload(
                trace,
                workload,
                capacities,
                state_factory(workload),
                global_models,
                boundary_models,
                radius=args.window_radius,
                rerank_mode=args.rerank_mode,
                residual_strength=args.residual_strength,
                same_layer_indices=same_layer_indices,
                build_future_occurrences=build_future_occurrences,
                scope=args.stage2_scope,
                set_context=args.stage2_set_context,
            )
            refs = references(inputs, workload, capacities)
            record = workload_record(
                workload,
                counters,
                refs,
                events,
                trace.num_experts,
                global_seconds,
                capacities,
            )
            records.append(record)
            all_sequence_rows.extend(sequence_rows)
            elapsed = time.perf_counter() - evaluation_start
            evaluation_seconds += elapsed
            report_workload(record, elapsed)

    aggregate = aggregate_records(records, capacities, trace.num_experts)
    default_protocol = (
        not cross_model
        and args.phase == "development"
        and args.smoke_sequences is None
        and capacities == DEFAULT_CAPACITIES
        and args.global_stride == 3
        and args.global_max_iter == 800
        and args.stage2_scope == "window"
        and not args.stage2_set_context
    )
    identity = (
        {"checked": False, "reason": "cross-model trace has no frozen OLMoE deployment"}
        if cross_model
        else development_identity(aggregate, capacities, default_protocol)
    )
    status = "NO_DEVELOPMENT_GAIN"
    if any(
        aggregate["methods"]["caer_gbt"][str(cap)][
            "relative_transfer_reduction_vs_global"
        ]
        > 0
        for cap in capacities
    ):
        status = (
            "DEVELOPMENT_SIGNAL_REQUIRES_REPLICATION"
            if args.phase != "evaluation"
            else "EVALUATION_GAIN"
        )

    result = {
        "schema_version": "caer_w2_v1",
        "method": "CAER — Cutoff-Aware Expert Reranking",
        "method_status": status,
        "fact_type": "DEPLOYABLE_METHOD",
        "phase": args.phase,
        "split": {
            "fit": fit_workload.name,
            "fit_sequences": len(fit_workload.sequences),
            "deploy": [workload.name for workload in deploy_workloads],
            "deploy_sequences": sum(len(workload.sequences) for workload in deploy_workloads),
            "evaluation_split_touched": args.phase == "evaluation",
        },
        "repository_root_runtime_dependency": str(repository),
        "frozen_artifacts_modified": False,
        "trace": trace_record,
        "causality": {
            "deployable_features": list(causal_names),
            "boundary_context_features": list(
                complete_boundary_names[len(causal_names) :]
            ),
            "future_information_use": "offline KEEP/EVICT labels and reported diagnostics only",
            "inference_future_information": False,
            "token_propensity_tables_fit_on": fit_workload.name,
        },
        "configuration": {
            "capacities": list(capacities),
            "window_radius": args.window_radius,
            "window_candidates": 2 * args.window_radius,
            "incomplete_windows": "fall back to unchanged global ordering",
            "global_model": {
                "type": "HistGradientBoostingRegressor",
                "max_leaf_nodes": 127,
                "max_iter": args.global_max_iter,
                "learning_rate": 0.06,
                "l2_regularization": 1.0,
                "min_samples_leaf": 40,
                "random_state": 0,
                "collection_stride": args.global_stride,
                "training_target": "rank",
            },
            "boundary_models": {
                "caer_logistic": "StandardScaler + LogisticRegression(C=1)",
                "caer_gbt": {
                    "type": "HistGradientBoostingClassifier",
                    "max_leaf_nodes": 31,
                    "max_iter": args.boundary_max_iter,
                    "learning_rate": 0.08,
                    "l2_regularization": 1.0,
                    "min_samples_leaf": 40,
                    "random_state": 0,
                },
                "scope": "one pooled classifier across capacities with capacity as a feature",
            },
            "rerank_mode": args.rerank_mode,
            "residual_strength": args.residual_strength,
            "stage2_scope": args.stage2_scope,
            "stage2_set_context": args.stage2_set_context,
            "offline_label_distance_cap": DISTANCE_CAP,
        },
        "training": {
            "token_table_seconds": token_seconds,
            "global_fit_rows": global_rows,
            "global_fit_seconds": global_fit_seconds,
            "boundary_collection_seconds": collection_seconds,
            "boundary_collection_global_batched_scoring_seconds": boundary_collection_global_seconds,
            "boundary_datasets": {
                str(capacity): datasets[int(capacity)].summary()
                for capacity in capacities
            },
            "boundary_dataset_path": str(dataset_path),
            "boundary_dataset_content_sha256": content_hash,
            "boundary_dataset_save_seconds": dataset_save_seconds,
            "boundary_fit": boundary_fit,
            "boundary_fit_seconds": boundary_fit_seconds,
        },
        "baseline_identity": identity,
        "evaluation_seconds": evaluation_seconds,
        "workloads": records,
        "aggregate": aggregate,
        "per_sequence": all_sequence_rows,
        "interpretation_guardrails": {
            "primary_metric": "expert transfers",
            "aggregate_accuracy_is_not_primary": True,
            "latency_claim": False,
            "hardware_speedup_claim": False,
            "measured_scoring_times_are_batched_simulator_times": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jsonable(result), indent=1) + "\n")
    print(f"wrote {output_path}", flush=True)
    for capacity in capacities:
        key = str(int(capacity))
        base = aggregate["methods"]["global_token_gbt"][key]
        for method in ("caer_logistic", "caer_gbt"):
            row = aggregate["methods"][method][key]
            print(
                f"  aggregate B={capacity} {method}: transfers={row['expert_transfers']:,}, "
                f"reduction={100 * row['relative_transfer_reduction_vs_global']:+.3f}%, "
                f"gap={100 * row['oracle_gap_closed']:.2f}% "
                f"({100 * row['oracle_gap_points_over_global']:+.2f} points vs global), "
                f"boundary_acc={100 * row['boundary_candidate_accuracy']:.2f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
