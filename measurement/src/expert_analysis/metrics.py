from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_save_npz


@dataclass
class DomainStatistics:
    """Compact per-example statistics; no hidden states or token-level activations."""

    routing_counts: np.ndarray
    gate_sums: np.ndarray
    contribution_sums: np.ndarray
    token_counts: np.ndarray
    layer_names: list[str]
    gradient_sums: np.ndarray | None = None

    @classmethod
    def zeros(
        cls,
        num_examples: int,
        num_layers: int,
        num_experts: int,
        layer_names: list[str],
        compute_gradient: bool = False,
    ) -> "DomainStatistics":
        shape = (num_examples, num_layers, num_experts)
        return cls(
            routing_counts=np.zeros(shape, dtype=np.uint32),
            gate_sums=np.zeros(shape, dtype=np.float32),
            contribution_sums=np.zeros(shape, dtype=np.float32),
            token_counts=np.zeros(num_examples, dtype=np.uint32),
            layer_names=list(layer_names),
            gradient_sums=np.zeros(shape, dtype=np.float32) if compute_gradient else None,
        )

    @property
    def num_examples(self) -> int:
        return int(self.routing_counts.shape[0])

    @property
    def num_layers(self) -> int:
        return int(self.routing_counts.shape[1])

    @property
    def num_experts(self) -> int:
        return int(self.routing_counts.shape[2])

    def validate(self) -> None:
        expected = self.routing_counts.shape
        if self.gate_sums.shape != expected or self.contribution_sums.shape != expected:
            raise ValueError("Per-example metric arrays have inconsistent shapes")
        if self.gradient_sums is not None and self.gradient_sums.shape != expected:
            raise ValueError("Gradient-attribution array has an inconsistent shape")
        if self.token_counts.shape != (expected[0],):
            raise ValueError("token_counts must contain one value per example")
        if len(self.layer_names) != expected[1]:
            raise ValueError("layer_names does not match the number of layers")
        for name, array in (
            ("gate_sums", self.gate_sums),
            ("contribution_sums", self.contribution_sums),
        ):
            if not np.all(np.isfinite(array)) or np.any(array < 0):
                raise ValueError(f"{name} contains invalid values")
        if self.gradient_sums is not None and (
            not np.all(np.isfinite(self.gradient_sums)) or np.any(self.gradient_sums < 0)
        ):
            raise ValueError("gradient_sums contains invalid values")

    def save(self, path: Path) -> None:
        self.validate()
        arrays: dict[str, Any] = {
            "routing_counts": self.routing_counts,
            "gate_sums": self.gate_sums,
            "contribution_sums": self.contribution_sums,
            "token_counts": self.token_counts,
            "layer_names": np.asarray(self.layer_names, dtype=np.str_),
        }
        if self.gradient_sums is not None:
            arrays["gradient_sums"] = self.gradient_sums
        atomic_save_npz(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "DomainStatistics":
        with np.load(path, allow_pickle=False) as data:
            instance = cls(
                routing_counts=data["routing_counts"],
                gate_sums=data["gate_sums"],
                contribution_sums=data["contribution_sums"],
                token_counts=data["token_counts"],
                layer_names=[str(item) for item in data["layer_names"].tolist()],
                gradient_sums=data["gradient_sums"] if "gradient_sums" in data else None,
            )
        instance.validate()
        return instance

    def aggregate(self, indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if indices is None:
            routing = self.routing_counts.sum(axis=0, dtype=np.float64)
            gate = self.gate_sums.sum(axis=0, dtype=np.float64)
            contribution = self.contribution_sums.sum(axis=0, dtype=np.float64)
            tokens = float(self.token_counts.sum(dtype=np.float64))
            gradient = (
                self.gradient_sums.sum(axis=0, dtype=np.float64)
                if self.gradient_sums is not None
                else None
            )
        else:
            routing = self.routing_counts[indices].sum(axis=0, dtype=np.float64)
            gate = self.gate_sums[indices].sum(axis=0, dtype=np.float64)
            contribution = self.contribution_sums[indices].sum(axis=0, dtype=np.float64)
            tokens = float(self.token_counts[indices].sum(dtype=np.float64))
            gradient = (
                self.gradient_sums[indices].sum(axis=0, dtype=np.float64)
                if self.gradient_sums is not None
                else None
            )
        assignments = routing.sum(axis=-1, keepdims=True)
        result = {
            "routing_frequency": safe_divide(routing, assignments),
            "gate_mass": gate / max(tokens, 1.0),
            "functional_contribution": contribution / max(tokens, 1.0),
            "normalized_routing": normalize_rows(routing),
            "normalized_gate": normalize_rows(gate),
            "normalized_contribution": normalize_rows(contribution),
        }
        if gradient is not None:
            result["gradient_attribution"] = gradient / max(tokens, 1.0)
            result["normalized_gradient"] = normalize_rows(gradient)
        return result


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator != 0)
    return output


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    totals = values.sum(axis=-1, keepdims=True)
    return safe_divide(values, totals)
