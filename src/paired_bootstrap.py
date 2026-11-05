"""Conditional paired bootstrap used by every test analysis (95% percentile intervals)."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

REPLICATES = 1000
SEED = 20260823


def paired_bootstrap(
    left: dict,
    right: dict,
    workloads: list[str],
    capacity: int,
) -> dict:
    """Conditional paired bootstrap, clustered by source sequence and stratified by domain.

    Returns the relative cost reduction of ``right`` against ``left`` with a 95%
    percentile interval over ``REPLICATES`` resamples.
    """

    parts, domains = [], {}
    for workload in workloads:
        a = left[(workload, capacity)]
        b = right[(workload, capacity)]
        if [(p, s, d) for p, s, d, _ in a] != [(p, s, d) for p, s, d, _ in b]:
            raise RuntimeError(f"per-sequence rows misalign for {workload}, B={capacity}")
        for _position, sequence, domain, _misses in a:
            domains.setdefault(sequence, domain)
        parts.append(
            (
                [sequence for _position, sequence, _domain, _misses in a],
                np.asarray([misses for *_prefix, misses in a], dtype=np.float64),
                np.asarray([misses for *_prefix, misses in b], dtype=np.float64),
            )
        )
    by_domain: dict[str, list[int]] = defaultdict(list)
    for sequence, domain in domains.items():
        by_domain[domain].append(sequence)
    digest = hashlib.sha256(f"paper-pack\0{SEED}\0{capacity}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    multiplicity = {}
    replicate_index = np.arange(REPLICATES, dtype=np.int64)[:, None]
    for domain in sorted(by_domain):
        ids = np.asarray(sorted(by_domain[domain]), dtype=np.int64)
        counts = np.zeros((REPLICATES, len(ids)), dtype=np.int32)
        np.add.at(
            counts,
            (replicate_index, rng.integers(0, len(ids), size=counts.shape)),
            1,
        )
        for position, sequence in enumerate(ids):
            multiplicity[int(sequence)] = counts[:, position]
    left_boot = np.zeros(REPLICATES)
    right_boot = np.zeros(REPLICATES)
    paired = []
    for sequences, left_values, right_values in parts:
        weights = np.stack([multiplicity[sequence] for sequence in sequences], axis=1)
        left_boot += (weights * left_values[None, :]).sum(axis=1)
        right_boot += (weights * right_values[None, :]).sum(axis=1)
        paired.extend(left_values - right_values)
    improvements = (left_boot - right_boot) / left_boot
    paired_values = np.asarray(paired)
    total_left = sum(float(values.sum()) for _sequences, values, _right in parts)
    total_right = sum(float(values.sum()) for _sequences, _left, values in parts)
    return {
        "capacity": capacity,
        "left_cost": int(total_left),
        "right_cost": int(total_right),
        "improvement": (total_left - total_right) / total_left,
        "ci_low": float(np.quantile(improvements, 0.025)),
        "ci_high": float(np.quantile(improvements, 0.975)),
        "sequences_improved": int(np.count_nonzero(paired_values > 0)),
        "sequences_worsened": int(np.count_nonzero(paired_values < 0)),
        "paired_units": int(paired_values.size),
        "bootstrap_replicates": REPLICATES,
    }
