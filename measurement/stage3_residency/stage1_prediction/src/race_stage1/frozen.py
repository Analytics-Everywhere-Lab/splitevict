from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from residency_headroom.common import read_json, sha256_file
from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import Workload, WorkloadSequence


@dataclass(frozen=True)
class FrozenInputs:
    preregistration: dict[str, Any]
    stage0_frozen: dict[str, Any]
    trace: RoutingTrace
    calibration: Workload
    workloads: tuple[Workload, ...]
    preregistration_hash: str


def repository_root_from(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists() and (candidate / "stage3_residency").exists():
            return candidate
    raise ValueError(f"Could not locate repository root from {path}")


def load_and_verify_frozen_inputs(
    repository_root: Path, preregistration_path: Path
) -> FrozenInputs:
    repository_root = repository_root.resolve()
    preregistration_path = preregistration_path.resolve()
    preregistration = read_json(preregistration_path)
    sidecar = preregistration_path.with_suffix(".sha256")
    expected_prereg = _read_sidecar_hash(sidecar)
    actual_prereg = sha256_file(preregistration_path)
    if actual_prereg != expected_prereg:
        raise ValueError(
            f"Stage 1 preregistration hash mismatch: {actual_prereg} != {expected_prereg}"
        )

    reference = preregistration["stage0_reference"]
    path_fields = {
        "frozen_evaluation_config_path": "frozen_evaluation_config_file_sha256",
        "result_path": "results_jsonl_sha256",
        "per_sequence_result_path": "per_sequence_results_jsonl_sha256",
        "trace_path": "routing_trace_npz_sha256",
    }
    for path_field, hash_field in path_fields.items():
        path = repository_root / reference[path_field]
        actual = sha256_file(path)
        if actual != reference[hash_field]:
            raise ValueError(f"Frozen Stage 0 file changed: {path} ({actual})")
    trace_path = repository_root / reference["trace_path"]
    metadata_path = trace_path.with_suffix(".metadata.json")
    if sha256_file(metadata_path) != reference["routing_trace_metadata_sha256"]:
        raise ValueError("Frozen Stage 0 routing trace metadata changed")

    stage0_frozen = read_json(repository_root / reference["frozen_evaluation_config_path"])
    if stage0_frozen["config_hash"] != reference["stage0_frozen_evaluation_config_hash"]:
        raise ValueError("Stage 0 frozen evaluation logical hash changed")
    if stage0_frozen["trace_hash"] != reference["trace_logical_hash"]:
        raise ValueError("Stage 0 frozen trace reference changed")

    trace = RoutingTrace.load(trace_path, validate=True)
    if trace.trace_hash != reference["trace_logical_hash"]:
        raise ValueError("Loaded trace does not match the preregistered logical hash")
    calibration = workload_from_record(stage0_frozen["calibration_workload"])
    workloads = tuple(workload_from_record(item) for item in stage0_frozen["workloads"])
    expected_workloads = stage0_frozen["workload_hashes"]
    actual_workloads = {item.name: item.hash for item in workloads}
    if actual_workloads != expected_workloads:
        raise ValueError("Reconstructed Stage 0 workload hashes changed")
    _validate_split(stage0_frozen, calibration, workloads)
    return FrozenInputs(
        preregistration=preregistration,
        stage0_frozen=stage0_frozen,
        trace=trace,
        calibration=calibration,
        workloads=workloads,
        preregistration_hash=actual_prereg,
    )


def workload_from_record(record: Mapping[str, Any]) -> Workload:
    sequences = tuple(
        WorkloadSequence(
            source_sequence_id=int(item["source_sequence_id"]),
            position=int(item["position"]),
            segment_index=int(item["segment_index"]),
            segment_label=str(item["segment_label"]),
            domain=str(item["domain"]),
        )
        for item in record["sequences"]
    )
    positions = tuple(item.position for item in sequences)
    if positions != tuple(range(len(sequences))):
        raise ValueError(f"Workload {record['name']} has noncontiguous positions")
    return Workload(name=str(record["name"]), regime=str(record["regime"]), sequences=sequences)


def iter_primary_stage0_rows(path: Path) -> Iterable[dict[str, Any]]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cost_model") == "unit_miss" and float(row.get("lambda", -1)) == 0.0:
                yield row


def stage0_references(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Return the oracle and per-condition strongest legitimate simple row."""

    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    simple_names = {"lru", "lfu", "lfu_decay", "static_hotset"}
    for row in iter_primary_stage0_rows(path):
        key = (str(row["workload"]), int(row["cache_capacity"]))
        item = grouped.setdefault(key, {})
        policy = str(row["policy"])
        if policy == "oracle":
            item["oracle"] = row
        elif policy in simple_names:
            if policy == "lfu_decay" and not bool(row.get("selected_decay_alpha")):
                continue
            candidate = item.get("simple")
            ranking = (float(row["total_cost"]), policy)
            if candidate is None or ranking < (
                float(candidate["total_cost"]),
                str(candidate["policy"]),
            ):
                item["simple"] = row
    incomplete = [key for key, item in grouped.items() if set(item) != {"simple", "oracle"}]
    if incomplete:
        raise ValueError(f"Incomplete Stage 0 primary references: {incomplete[:5]}")
    return grouped


def source_bundle_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for directory in (root / "src", root / "scripts", root / "configs", root / "tests"):
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _read_sidecar_hash(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip().split()
    if len(value) < 1 or len(value[0]) != 64:
        raise ValueError(f"Invalid SHA-256 sidecar {path}")
    return value[0]


def _validate_split(
    frozen: Mapping[str, Any], calibration: Workload, workloads: Iterable[Workload]
) -> None:
    split = frozen["sequence_split"]
    expected_calibration = {
        int(value) for values in split["calibration"].values() for value in values
    }
    expected_evaluation = {
        int(value) for values in split["evaluation"].values() for value in values
    }
    actual_calibration = set(calibration.sequence_ids)
    if actual_calibration != expected_calibration:
        raise ValueError("Frozen calibration workload IDs differ from the Stage 0 split")
    if expected_calibration & expected_evaluation:
        raise ValueError("Frozen Stage 0 split contains calibration/evaluation leakage")
    for workload in workloads:
        if not set(workload.sequence_ids).issubset(expected_evaluation):
            raise ValueError(f"Workload {workload.name} leaks calibration sequences")
