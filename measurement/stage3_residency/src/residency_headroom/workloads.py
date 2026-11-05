from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .common import sha256_json
from .trace import RoutingTrace


DEFAULT_ABRUPT_PAIRS = (
    ("general", "coding"),
    ("coding", "general"),
    ("general", "math"),
    ("math", "reasoning"),
)
DEFAULT_REPEATED_ORDER = ("general", "coding", "math", "reasoning", "general")


@dataclass(frozen=True)
class WorkloadSequence:
    source_sequence_id: int
    position: int
    segment_index: int
    segment_label: str
    domain: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_sequence_id": self.source_sequence_id,
            "position": self.position,
            "segment_index": self.segment_index,
            "segment_label": self.segment_label,
            "domain": self.domain,
        }


@dataclass(frozen=True)
class Workload:
    name: str
    regime: str
    sequences: tuple[WorkloadSequence, ...]

    @property
    def sequence_ids(self) -> tuple[int, ...]:
        return tuple(item.source_sequence_id for item in self.sequences)

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.domain for item in self.sequences))

    @property
    def hash(self) -> str:
        return sha256_json(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "regime": self.regime,
            "sequences": [item.as_dict() for item in self.sequences],
        }

    def iter_slices(
        self, trace: RoutingTrace
    ) -> Iterator[tuple[WorkloadSequence, slice]]:
        slices = trace.sequence_slices()
        for item in self.sequences:
            try:
                view = slices[item.source_sequence_id]
            except KeyError as exc:
                raise ValueError(
                    f"Workload {self.name} references missing sequence "
                    f"{item.source_sequence_id}"
                ) from exc
            yield item, view


@dataclass(frozen=True)
class SequenceSplit:
    calibration: dict[str, tuple[int, ...]]
    evaluation: dict[str, tuple[int, ...]]
    calibration_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibration_fraction": self.calibration_fraction,
            "calibration": {name: list(values) for name, values in self.calibration.items()},
            "evaluation": {name: list(values) for name, values in self.evaluation.items()},
        }


def split_sequences(trace: RoutingTrace, calibration_fraction: float) -> SequenceSplit:
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in [0, 1)")
    by_domain: dict[str, list[int]] = {domain: [] for domain in trace.metadata["domains"]}
    for item in trace.metadata["sequences"]:
        by_domain[item["domain"]].append(int(item["sequence_id"]))
    calibration: dict[str, tuple[int, ...]] = {}
    evaluation: dict[str, tuple[int, ...]] = {}
    for domain, identifiers in by_domain.items():
        if not identifiers:
            raise ValueError(f"Trace has no sequences for domain {domain}")
        count = int(np.floor(len(identifiers) * calibration_fraction))
        if calibration_fraction > 0:
            count = max(1, count)
        if count >= len(identifiers):
            raise ValueError(f"Calibration split leaves no evaluation sequence for {domain}")
        calibration[domain] = tuple(identifiers[:count])
        evaluation[domain] = tuple(identifiers[count:])
    _validate_split(calibration, evaluation)
    return SequenceSplit(calibration, evaluation, calibration_fraction)


def build_workloads(trace: RoutingTrace, split: SequenceSplit, config: Mapping[str, Any]) -> list[Workload]:
    domains = tuple(config["domains"])
    workloads: list[Workload] = []
    for domain in domains:
        identifiers = split.evaluation[domain]
        workloads.append(
            make_workload(
                f"stationary_{domain}",
                "stationary",
                [(identifier, 0, domain, domain) for identifier in identifiers],
            )
        )

    abrupt_pairs = tuple(
        tuple(pair) for pair in config.get("abrupt_pairs", DEFAULT_ABRUPT_PAIRS)
    )
    for before, after in abrupt_pairs:
        entries = [
            (identifier, 0, f"{before}_before", before)
            for identifier in split.evaluation[before]
        ] + [
            (identifier, 1, f"{after}_after", after)
            for identifier in split.evaluation[after]
        ]
        workloads.append(make_workload(f"{before}_to_{after}", "abrupt", entries))

    repeated_order = tuple(config.get("repeated_domain_order", DEFAULT_REPEATED_ORDER))
    segment_length = int(config["repeated_segment_prompts"])
    if segment_length < 1:
        raise ValueError("repeated_segment_prompts must be positive")
    used: dict[str, int] = {domain: 0 for domain in domains}
    repeated_entries: list[tuple[int, int, str, str]] = []
    for segment_index, domain in enumerate(repeated_order):
        start = used[domain]
        stop = start + segment_length
        chosen = split.evaluation[domain][start:stop]
        if len(chosen) != segment_length:
            raise ValueError(
                f"Repeated workload needs {stop} evaluation prompts for {domain}, "
                f"but only {len(split.evaluation[domain])} exist"
            )
        used[domain] = stop
        repeated_entries.extend(
            (identifier, segment_index, f"segment_{segment_index}_{domain}", domain)
            for identifier in chosen
        )
    workloads.append(make_workload("repeated_domain_cycle", "repeated", repeated_entries))

    mixed_entries = [
        (identifier, 0, "mixed", domain)
        for domain in domains
        for identifier in split.evaluation[domain]
    ]
    rng = np.random.default_rng(int(config["mixed_workload_seed"]))
    permutation = rng.permutation(len(mixed_entries))
    workloads.append(
        make_workload(
            "mixed_interleaved",
            "mixed",
            [mixed_entries[int(index)] for index in permutation],
        )
    )
    validate_workloads(trace, split, workloads)
    return workloads


def build_calibration_workload(
    trace: RoutingTrace, split: SequenceSplit, seed: int
) -> Workload:
    entries = [
        (identifier, 0, "calibration_mixed", domain)
        for domain in trace.metadata["domains"]
        for identifier in split.calibration[domain]
    ]
    if not entries:
        raise ValueError("The trace has no calibration sequences")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(entries))
    return make_workload(
        "calibration_mixed",
        "calibration",
        [entries[int(index)] for index in permutation],
    )


def make_workload(
    name: str,
    regime: str,
    entries: Sequence[tuple[int, int, str, str]],
) -> Workload:
    if not entries:
        raise ValueError(f"Workload {name} has no sequences")
    sequences = tuple(
        WorkloadSequence(
            source_sequence_id=int(identifier),
            position=position,
            segment_index=int(segment),
            segment_label=str(label),
            domain=str(domain),
        )
        for position, (identifier, segment, label, domain) in enumerate(entries)
    )
    if len(set(item.source_sequence_id for item in sequences)) != len(sequences):
        raise ValueError(f"Workload {name} repeats a source sequence")
    return Workload(name=name, regime=regime, sequences=sequences)


def validate_workloads(
    trace: RoutingTrace, split: SequenceSplit, workloads: Iterable[Workload]
) -> None:
    evaluation = {value for values in split.evaluation.values() for value in values}
    calibration = {value for values in split.calibration.values() for value in values}
    if evaluation & calibration:
        raise ValueError("Calibration/evaluation sequence leakage")
    manifest = {
        int(item["sequence_id"]): item["domain"] for item in trace.metadata["sequences"]
    }
    for workload in workloads:
        if not workload.sequences:
            raise ValueError(f"Workload {workload.name} is empty")
        for item in workload.sequences:
            if item.source_sequence_id not in evaluation:
                raise ValueError(
                    f"Workload {workload.name} uses non-evaluation sequence "
                    f"{item.source_sequence_id}"
                )
            if manifest[item.source_sequence_id] != item.domain:
                raise ValueError(f"Workload {workload.name} changes a sequence's domain")


def calibration_frequency_scores(
    trace: RoutingTrace, workload: Workload
) -> np.ndarray:
    layers = list(map(int, trace.metadata["layer_indices"]))
    layer_to_ordinal = {layer: index for index, layer in enumerate(layers)}
    scores = np.zeros((len(layers), trace.num_experts), dtype=np.int64)
    for _sequence, view in workload.iter_slices(trace):
        for index in range(view.start, view.stop):
            ordinal = layer_to_ordinal[int(trace.layer_index[index])]
            np.add.at(scores[ordinal], trace.requested_expert_ids[index], 1)
    return scores


def _validate_split(
    calibration: Mapping[str, Sequence[int]], evaluation: Mapping[str, Sequence[int]]
) -> None:
    all_calibration = [item for values in calibration.values() for item in values]
    all_evaluation = [item for values in evaluation.values() for item in values]
    if len(all_calibration) != len(set(all_calibration)):
        raise ValueError("A sequence appears twice in calibration")
    if len(all_evaluation) != len(set(all_evaluation)):
        raise ValueError("A sequence appears twice in evaluation")
    if set(all_calibration) & set(all_evaluation):
        raise ValueError("Calibration/evaluation sequence leakage")
