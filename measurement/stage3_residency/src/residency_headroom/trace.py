from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import TRACE_SCHEMA_VERSION
from .common import atomic_save_npz, atomic_write_json, hash_arrays, read_json


TRACE_ARRAY_NAMES = (
    "event_index",
    "sequence_id",
    "domain_id",
    "prompt_index",
    "generated_token_index",
    "layer_index",
    "requested_expert_ids",
    "router_weights",
    "token_id",
    "prompt_length",
    "generation_length",
)


@dataclass(frozen=True)
class RoutingTrace:
    event_index: np.ndarray
    sequence_id: np.ndarray
    domain_id: np.ndarray
    prompt_index: np.ndarray
    generated_token_index: np.ndarray
    layer_index: np.ndarray
    requested_expert_ids: np.ndarray
    router_weights: np.ndarray
    token_id: np.ndarray
    prompt_length: np.ndarray
    generation_length: np.ndarray
    metadata: dict[str, Any]

    @property
    def num_events(self) -> int:
        return int(self.event_index.size)

    @property
    def top_k(self) -> int:
        return int(self.requested_expert_ids.shape[1])

    @property
    def num_layers(self) -> int:
        return len(self.metadata["layer_indices"])

    @property
    def num_experts(self) -> int:
        return int(self.metadata["num_experts"])

    @property
    def trace_hash(self) -> str:
        return self.logical_hash()

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in TRACE_ARRAY_NAMES}

    def logical_hash(self) -> str:
        return hash_arrays(
            self.arrays(),
            {
                "schema_version": self.metadata.get("schema_version"),
                "hash_basis": self.metadata.get("hash_basis"),
                "sequences": self.metadata.get("sequences"),
                "domains": self.metadata.get("domains"),
                "layer_indices": self.metadata.get("layer_indices"),
                "num_experts": self.metadata.get("num_experts"),
                "top_k": self.metadata.get("top_k"),
                "expert_bytes_by_layer": self.metadata.get("expert_bytes_by_layer"),
            },
        )

    def sequence_slices(self) -> dict[int, slice]:
        ids = self.sequence_id
        if ids.size == 0:
            return {}
        changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate((np.asarray([0]), changes))
        stops = np.concatenate((changes, np.asarray([len(ids)])))
        return {
            int(ids[start]): slice(int(start), int(stop))
            for start, stop in zip(starts, stops)
        }

    def validate(self, verify_hash: bool = True) -> dict[str, Any]:
        arrays = self.arrays()
        lengths = {name: int(array.shape[0]) for name, array in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Trace arrays have inconsistent event counts: {lengths}")
        events = self.num_events
        if events == 0:
            raise ValueError("A routing trace must contain at least one event")
        if self.metadata.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported trace schema {self.metadata.get('schema_version')!r}"
            )
        if not np.array_equal(self.event_index, np.arange(events, dtype=self.event_index.dtype)):
            raise ValueError("event_index must be contiguous, ordered, and zero based")
        if self.requested_expert_ids.ndim != 2:
            raise ValueError("requested_expert_ids must have shape [event, top_k]")
        if self.router_weights.shape != self.requested_expert_ids.shape:
            raise ValueError("router_weights and requested_expert_ids shapes differ")
        if self.top_k != int(self.metadata.get("top_k", -1)):
            raise ValueError("Trace top-k does not match metadata")
        if np.any(self.requested_expert_ids < 0) or np.any(
            self.requested_expert_ids >= self.num_experts
        ):
            raise ValueError("Trace contains an out-of-range expert ID")
        sorted_ids = np.sort(self.requested_expert_ids.astype(np.int64), axis=1)
        if self.top_k > 1 and np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
            raise ValueError("An atomic request contains a duplicate expert ID")
        if not np.isfinite(self.router_weights).all() or np.any(self.router_weights < 0):
            raise ValueError("Trace contains invalid router weights")
        layer_indices = np.asarray(self.metadata["layer_indices"], dtype=np.int64)
        if len(layer_indices) != len(np.unique(layer_indices)):
            raise ValueError("Layer indices in trace metadata are not unique")
        if not np.isin(self.layer_index, layer_indices).all():
            raise ValueError("Trace contains an unknown layer index")

        sequences = self.metadata.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("Trace metadata contains no sequence manifest")
        slices = self.sequence_slices()
        manifest_ids = [int(item["sequence_id"]) for item in sequences]
        if list(slices) != manifest_ids:
            raise ValueError("Sequence array order differs from the sequence manifest")
        domain_names = list(self.metadata.get("domains", []))
        total_tokens = 0
        for item in sequences:
            sequence = int(item["sequence_id"])
            view = slices[sequence]
            generation_length = int(item["generation_length"])
            expected = generation_length * len(layer_indices)
            if view.stop - view.start != expected:
                raise ValueError(
                    f"Sequence {sequence} has {view.stop-view.start} events; expected {expected}"
                )
            event_layers = self.layer_index[view].reshape(generation_length, -1)
            if not np.array_equal(
                event_layers, np.broadcast_to(layer_indices, event_layers.shape)
            ):
                raise ValueError(f"Sequence {sequence} has missing or reordered layer events")
            generated = self.generated_token_index[view].reshape(generation_length, -1)
            expected_generated = np.broadcast_to(
                np.arange(generation_length, dtype=generated.dtype)[:, None], generated.shape
            )
            if not np.array_equal(generated, expected_generated):
                raise ValueError(f"Sequence {sequence} has invalid generated-token indices")
            token_ids = self.token_id[view].reshape(generation_length, -1)
            if np.any(token_ids != token_ids[:, :1]):
                raise ValueError(f"Sequence {sequence} changes token ID across layers")
            if list(map(int, token_ids[:, 0])) != list(map(int, item["generated_token_ids"])):
                raise ValueError(f"Sequence {sequence} token IDs differ from metadata")
            if np.any(self.sequence_id[view] != sequence):
                raise ValueError("Sequence slice accounting failed")
            domain_id = int(item["domain_id"])
            if domain_id < 0 or domain_id >= len(domain_names):
                raise ValueError(f"Sequence {sequence} has invalid domain ID")
            if item["domain"] != domain_names[domain_id]:
                raise ValueError(f"Sequence {sequence} domain mapping is inconsistent")
            for name, expected_value in (
                ("domain_id", domain_id),
                ("prompt_index", int(item["prompt_index"])),
                ("prompt_length", int(item["prompt_length"])),
                ("generation_length", generation_length),
            ):
                if np.any(getattr(self, name)[view] != expected_value):
                    raise ValueError(f"Sequence {sequence} has inconsistent {name}")
            total_tokens += generation_length

        actual_hash = self.logical_hash()
        recorded_hash = self.metadata.get("trace_hash")
        if verify_hash and recorded_hash != actual_hash:
            raise ValueError(
                f"Trace hash mismatch: metadata={recorded_hash}, computed={actual_hash}"
            )
        total_requests = int(self.requested_expert_ids.size)
        return {
            "passed": True,
            "trace_hash": actual_hash,
            "events": events,
            "sequences": len(sequences),
            "generated_tokens": total_tokens,
            "requested_experts": total_requests,
            "events_per_layer": {
                str(int(layer)): int(np.count_nonzero(self.layer_index == layer))
                for layer in layer_indices
            },
        }

    def save(self, path: Path) -> None:
        metadata = dict(self.metadata)
        metadata["trace_hash"] = self.logical_hash()
        trace = RoutingTrace(**self.arrays(), metadata=metadata)
        trace.validate(verify_hash=True)
        atomic_save_npz(
            path,
            schema_version=np.asarray(TRACE_SCHEMA_VERSION),
            **trace.arrays(),
        )
        atomic_write_json(metadata_path_for(path), metadata)

    @classmethod
    def load(cls, path: Path, validate: bool = True) -> "RoutingTrace":
        metadata = read_json(metadata_path_for(path))
        with np.load(path, allow_pickle=False) as archive:
            schema = str(archive["schema_version"].item())
            if schema != TRACE_SCHEMA_VERSION:
                raise ValueError(f"Unsupported NPZ trace schema {schema!r}")
            arrays = {name: np.asarray(archive[name]) for name in TRACE_ARRAY_NAMES}
        trace = cls(**arrays, metadata=metadata)
        if validate:
            trace.validate(verify_hash=True)
        return trace

    @classmethod
    def from_mapping(
        cls, arrays: Mapping[str, Any], metadata: Mapping[str, Any], validate: bool = True
    ) -> "RoutingTrace":
        trace = cls(
            **{name: np.asarray(arrays[name]) for name in TRACE_ARRAY_NAMES},
            metadata=dict(metadata),
        )
        if validate:
            trace.validate(verify_hash=bool(trace.metadata.get("trace_hash")))
        return trace


def metadata_path_for(path: Path) -> Path:
    return path.with_suffix(".metadata.json")
