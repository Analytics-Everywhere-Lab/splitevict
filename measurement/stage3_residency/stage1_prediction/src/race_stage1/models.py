from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from residency_headroom.common import atomic_save_npz, atomic_write_json, read_json, sha256_file
from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import Workload


MODEL_SCHEMA_VERSION = "race_stage1_transition_models_v1"


@dataclass(frozen=True)
class TransitionModels:
    horizons: tuple[int, ...]
    probabilities: np.ndarray
    conditioning_counts: np.ndarray
    trace_hash: str
    calibration_workload_hash: str
    smoothing: str = "independent_beta_1_1_binary_event_smoothing"

    def __post_init__(self) -> None:
        if self.probabilities.ndim != 4:
            raise ValueError("Transition probabilities must have [H, layer, source, target]")
        if self.probabilities.shape[0] != len(self.horizons):
            raise ValueError("Horizon count does not match transition tensor")
        if self.conditioning_counts.shape != self.probabilities.shape[:-1]:
            raise ValueError("Conditioning count shape does not match transition tensor")
        if not np.isfinite(self.probabilities).all():
            raise ValueError("Transition probabilities are not finite")
        if np.any(self.probabilities < 0) or np.any(self.probabilities > 1):
            raise ValueError("Transition probabilities lie outside [0, 1]")

    @property
    def num_layers(self) -> int:
        return int(self.probabilities.shape[1])

    @property
    def num_experts(self) -> int:
        return int(self.probabilities.shape[2])

    def matrix(self, horizon: int, layer_ordinal: int) -> np.ndarray:
        try:
            index = self.horizons.index(int(horizon))
        except ValueError as exc:
            raise ValueError(f"No fitted transition model for H={horizon}") from exc
        return self.probabilities[index, int(layer_ordinal)]

    def save(self, path: Path) -> dict[str, Any]:
        atomic_save_npz(
            path,
            schema_version=np.asarray(MODEL_SCHEMA_VERSION),
            horizons=np.asarray(self.horizons, dtype=np.int64),
            probabilities=self.probabilities.astype(np.float32, copy=False),
            conditioning_counts=self.conditioning_counts.astype(np.int64, copy=False),
        )
        metadata = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "trace_hash": self.trace_hash,
            "calibration_workload_hash": self.calibration_workload_hash,
            "smoothing": self.smoothing,
            "horizons": list(self.horizons),
            "npz_sha256": sha256_file(path),
        }
        atomic_write_json(path.with_suffix(".metadata.json"), metadata)
        return metadata

    @classmethod
    def load(cls, path: Path) -> "TransitionModels":
        metadata = read_json(path.with_suffix(".metadata.json"))
        if metadata["npz_sha256"] != sha256_file(path):
            raise ValueError("Transition-model archive hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            schema = str(archive["schema_version"].item())
            if schema != MODEL_SCHEMA_VERSION:
                raise ValueError(f"Unsupported transition-model schema {schema}")
            model = cls(
                horizons=tuple(map(int, archive["horizons"])),
                probabilities=np.asarray(archive["probabilities"], dtype=np.float64),
                conditioning_counts=np.asarray(archive["conditioning_counts"], dtype=np.int64),
                trace_hash=str(metadata["trace_hash"]),
                calibration_workload_hash=str(metadata["calibration_workload_hash"]),
                smoothing=str(metadata["smoothing"]),
            )
        model.probabilities.flags.writeable = False
        model.conditioning_counts.flags.writeable = False
        return model


def same_layer_indices(trace: RoutingTrace, workload: Workload) -> tuple[np.ndarray, ...]:
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    chunks: list[list[np.ndarray]] = [[] for _ in layers]
    for _sequence, view in workload.iter_slices(trace):
        length = view.stop - view.start
        if length % len(layers):
            raise ValueError("Sequence event count is not divisible by the layer count")
        for ordinal, layer in enumerate(layers):
            indices = np.arange(view.start + ordinal, view.stop, len(layers), dtype=np.int64)
            if not np.all(trace.layer_index[indices] == layer):
                raise ValueError("Trace no longer follows frozen same-layer event ordering")
            chunks[ordinal].append(indices)
    return tuple(np.concatenate(values) for values in chunks)


def fit_transition_models(
    trace: RoutingTrace, workload: Workload, horizons: Sequence[int]
) -> TransitionModels:
    requested_horizons = tuple(sorted(set(map(int, horizons))))
    if not requested_horizons or requested_horizons[0] < 1:
        raise ValueError("Transition horizons must be positive")
    layer_indices = same_layer_indices(trace, workload)
    shape = (len(requested_horizons), trace.num_layers, trace.num_experts, trace.num_experts)
    probabilities = np.empty(shape, dtype=np.float64)
    denominators = np.empty(shape[:-1], dtype=np.int64)
    for layer_ordinal, indices in enumerate(layer_indices):
        requests = np.asarray(trace.requested_expert_ids[indices], dtype=np.int64)
        for horizon_ordinal, horizon in enumerate(requested_horizons):
            positive, conditioning = _fit_one(requests, trace.num_experts, horizon)
            probabilities[horizon_ordinal, layer_ordinal] = (
                positive.astype(np.float64) + 1.0
            ) / (conditioning[:, None].astype(np.float64) + 2.0)
            denominators[horizon_ordinal, layer_ordinal] = conditioning
    probabilities.flags.writeable = False
    denominators.flags.writeable = False
    return TransitionModels(
        horizons=requested_horizons,
        probabilities=probabilities,
        conditioning_counts=denominators,
        trace_hash=trace.trace_hash,
        calibration_workload_hash=workload.hash,
    )


def _fit_one(
    requests: np.ndarray, num_experts: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    if requests.ndim != 2 or requests.shape[0] < 2:
        raise ValueError("Transition fitting requires at least two atomic events")
    conditioning = np.zeros(num_experts, dtype=np.int64)
    positive = np.zeros((num_experts, num_experts), dtype=np.int64)
    future_counts = np.zeros(num_experts, dtype=np.int64)
    initial_stop = min(requests.shape[0], horizon + 1)
    for index in range(1, initial_stop):
        np.add.at(future_counts, requests[index], 1)
    for position in range(requests.shape[0] - 1):
        source = requests[position]
        target = np.flatnonzero(future_counts)
        np.add.at(conditioning, source, 1)
        positive[np.ix_(source, target)] += 1
        np.add.at(future_counts, requests[position + 1], -1)
        entering = position + horizon + 1
        if entering < requests.shape[0]:
            np.add.at(future_counts, requests[entering], 1)
        if np.any(future_counts < 0):
            raise RuntimeError("Sliding Markov-H future window became negative")
    if np.any(positive > conditioning[:, None]):
        raise RuntimeError("Binary Markov-H positives exceed conditioning events")
    return positive, conditioning
