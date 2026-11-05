from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import CONFIG_SCHEMA_VERSION
from .common import (
    atomic_save_npz,
    atomic_write_json,
    atomic_write_text,
    hash_arrays,
    sha256_json,
    utc_now,
)
from .oracle import validate_oracle
from .simulator import simulate_policy
from .trace import RoutingTrace
from .workloads import (
    build_calibration_workload,
    build_workloads,
    calibration_frequency_scores,
    split_sequences,
)


def validate_stage0_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Stage 0 config schema {config.get('schema_version')!r}")
    capacities = list(map(int, config["cache_capacities"]))
    if capacities != sorted(set(capacities)) or any(value < 1 for value in capacities):
        raise ValueError("Cache capacities must be unique positive values in sorted order")
    expected = config["expected_architecture"]
    if any(value < int(expected["top_k"]) for value in capacities):
        raise ValueError("A scientific cache capacity is smaller than atomic top-k")
    alphas = list(map(float, config["lfu_decay_alphas"]))
    if alphas != [0.9, 0.95, 0.99]:
        raise ValueError("The preregistered LFU-decay grid must be [0.90, 0.95, 0.99]")
    if len(config["random_policy_seeds"]) < 5:
        raise ValueError("Random eviction requires at least five fixed seeds")
    if float(config["primary_lambda"]) != 0.0:
        raise ValueError("The primary Stage 0 decision must use lambda=0")
    if config["primary_cost_model"] != "unit_miss":
        raise ValueError("The primary Stage 0 decision must use unit miss cost")
    criteria = config["go_no_go"]
    if float(criteria["strong_point_headroom"]) != 0.15:
        raise ValueError("STRONG GO point threshold must remain 15%")
    if int(criteria["strong_required_budgets"]) != 3 or int(criteria["total_budgets"]) != 5:
        raise ValueError("STRONG GO must require at least 3/5 cache budgets")
    run_kind = config.get("run_kind")
    if run_kind not in {"pilot", "full"}:
        raise ValueError("Only pilot and full configurations can be evaluation-frozen")
    if bool(config.get("decision_enabled")) != (run_kind == "full"):
        raise ValueError("Scientific decisions must be disabled for pilot and enabled for full")
    if list(config["cost_models"]) != ["unit_miss", "expert_bytes"]:
        raise ValueError("Both preregistered cost models must be preserved in order")
    if list(map(float, config["lambda_values"])) != [0.0, 0.25, 0.5, 1.0]:
        raise ValueError("The preregistered lambda grid changed")
    if config.get("bootstrap_unit") != "source_decode_sequence":
        raise ValueError("The bootstrap unit must remain the source decode sequence")
    if list(config.get("workload_bootstrap_strata", [])) != [
        "segment_index",
        "domain",
    ]:
        raise ValueError("Workload bootstrap strata must remain segment_index and domain")
    if (
        config.get("regime_bootstrap_cluster")
        != "source_sequence_id_stratified_by_domain"
    ):
        raise ValueError("Regime bootstrap must cluster repeated source sequences")


def freeze_evaluation(
    trace: RoutingTrace,
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    oracle_random_cases: int = 500,
) -> dict[str, Any]:
    validate_stage0_config(config)
    trace_validation = trace.validate(verify_hash=True)
    expected = config["expected_architecture"]
    if trace.num_layers != int(expected["num_moe_layers"]):
        raise ValueError("Trace has the wrong number of MoE layers")
    if trace.num_experts != int(expected["num_experts"]):
        raise ValueError("Trace has the wrong expert count")
    if trace.top_k != int(expected["top_k"]):
        raise ValueError("Trace has the wrong top-k")
    if list(trace.metadata["domains"]) != list(config["domains"]):
        raise ValueError("Trace domains differ from the preregistered order")
    expected_sequences = len(config["domains"]) * int(config["num_prompts_per_domain"])
    if len(trace.metadata["sequences"]) != expected_sequences:
        raise ValueError(
            f"Trace contains {len(trace.metadata['sequences'])} sequences; "
            f"expected {expected_sequences}"
        )

    split = split_sequences(trace, float(config["calibration_fraction"]))
    calibration = build_calibration_workload(
        trace, split, int(config["mixed_workload_seed"])
    )
    workloads = build_workloads(trace, split, config)
    static_scores = calibration_frequency_scores(trace, calibration)
    static_hash = hash_arrays(
        {"static_frequency": static_scores},
        {
            "trace_hash": trace.trace_hash,
            "calibration_sequence_ids": list(calibration.sequence_ids),
        },
    )
    selected_alpha, alpha_scores = select_decay_alpha(trace, calibration, config)
    oracle_validation = validate_oracle(
        random_cases=oracle_random_cases,
        seed=int(config["bootstrap_seed"]),
        lambda_values=config["lambda_values"],
    )

    config_basis: dict[str, Any] = {
        "schema_version": "race_stage0_frozen_evaluation_v1",
        "preregistered_config": dict(config),
        "preregistered_config_hash": sha256_json(config),
        "trace_hash": trace.trace_hash,
        "trace_validation": trace_validation,
        "sample_manifest": trace.metadata["sequences"],
        "sequence_split": split.as_dict(),
        "calibration_workload": calibration.as_dict(),
        "workloads": [workload.as_dict() for workload in workloads],
        "workload_hashes": {workload.name: workload.hash for workload in workloads},
        "static_hotset_score_hash": static_hash,
        "selected_lfu_decay_alpha": selected_alpha,
        "lfu_decay_calibration_scores": alpha_scores,
        "oracle_validation": oracle_validation.as_dict(),
        "cache_semantics": {
            "atomic_set_requests": True,
            "cache_scope": "independent_per_moe_layer",
            "mandatory_requested_experts_in_post_state": True,
            "prefetch": False,
            "positive_capacity_below_top_k": "invalid",
            "initial_state": "empty",
            "admissions_equal_misses": True,
        },
        "strongest_simple_candidates": [
            "lru",
            "lfu",
            f"lfu_decay(alpha={selected_alpha})",
            "static_hotset",
        ],
        "random_excluded_from_best_simple": True,
    }
    config_hash = sha256_json(config_basis)
    frozen = dict(config_basis)
    frozen["config_hash"] = config_hash
    frozen["frozen_at_utc"] = utc_now()

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "frozen_evaluation_config.json"
    hash_path = output_dir / "frozen_evaluation_config.sha256"
    scores_path = output_dir / "static_hotset_scores.npz"
    if config_path.exists():
        from .common import read_json

        previous = read_json(config_path)
        if previous.get("config_hash") != config_hash:
            raise RuntimeError(
                "A different frozen evaluation already exists. Create a separately "
                "labeled exploratory run; never alter the frozen final configuration."
            )
        return previous
    atomic_save_npz(
        scores_path,
        static_frequency=static_scores,
        score_hash=np.asarray(static_hash),
    )
    atomic_write_json(config_path, frozen)
    atomic_write_text(hash_path, f"{config_hash}  {config_path.name}\n")
    return frozen


def select_decay_alpha(
    trace: RoutingTrace,
    calibration: Any,
    config: Mapping[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for alpha in map(float, config["lfu_decay_alphas"]):
        total = 0
        by_capacity: dict[str, int] = {}
        for capacity in map(int, config["cache_capacities"]):
            result = simulate_policy(
                trace,
                calibration,
                capacity,
                "lfu_decay",
                alpha=alpha,
            )
            total += result.misses
            by_capacity[str(capacity)] = result.misses
        records.append(
            {"alpha": alpha, "summed_calibration_misses": total, "by_capacity": by_capacity}
        )
    # The numeric alpha is the deterministic tie break, fixed before evaluation.
    selected = min(records, key=lambda item: (item["summed_calibration_misses"], item["alpha"]))
    return float(selected["alpha"]), records


def load_static_scores(path: Path, expected_hash: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        recorded_hash = str(archive["score_hash"].item())
        scores = np.asarray(archive["static_frequency"], dtype=np.int64)
    if recorded_hash != expected_hash:
        raise ValueError("Static Hotset score hash differs from frozen config")
    return scores
