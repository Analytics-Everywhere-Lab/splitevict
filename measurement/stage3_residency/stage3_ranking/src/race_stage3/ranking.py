"""Convex pairwise ranking fit over within-candidate-set groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize


MAX_PAIRS_PER_GROUP = 30
MAX_ITERATIONS = 500


def group_slices(groups: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, stop) spans of equal group id."""

    values = np.asarray(groups)
    if values.size == 0:
        return []
    edges = np.flatnonzero(np.diff(values)) + 1
    return list(zip(np.r_[0, edges].tolist(), np.r_[edges, values.size].tolist()))


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return mean, scale


def build_pairs(
    standardized: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    max_pairs: int = MAX_PAIRS_PER_GROUP,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Within-group ordered pairs (a, b) with target(a) < target(b).

    Each group contributes equal total weight so that large candidate sets do not
    dominate, and at most ``max_pairs`` sampled pairs with a fixed seed.
    """

    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for start, stop in group_slices(groups):
        target = targets[start:stop]
        rows = standardized[start:stop]
        left, right = np.nonzero(target[:, None] < target[None, :])
        if left.size == 0:
            continue
        if left.size > max_pairs:
            chosen = rng.choice(left.size, max_pairs, replace=False)
            left, right = left[chosen], right[chosen]
        blocks.append(rows[left] - rows[right])
        weights.append(np.full(left.size, 1.0 / left.size))
    if not blocks:
        raise ValueError("No comparable calibration pairs were collected")
    return np.concatenate(blocks).astype(np.float64), np.concatenate(weights)


def fit_pairwise_logistic(
    differences: np.ndarray, weights: np.ndarray, l2: float
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Minimize the convex weighted logistic pairwise ranking loss."""

    mass = float(weights.sum())

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        margin = differences @ theta
        value = float(weights @ np.logaddexp(0.0, -margin)) / mass + 0.5 * l2 * float(theta @ theta)
        gradient = -(differences.T @ (weights / (1.0 + np.exp(margin)))) / mass + l2 * theta
        return value, gradient

    result = minimize(
        objective,
        np.zeros(differences.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAX_ITERATIONS, "ftol": 1e-12, "gtol": 1e-10},
    )
    info = {
        "converged": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "l2": float(l2),
        "pairs": int(differences.shape[0]),
    }
    return result.x, float(result.fun), info


def pairwise_accuracy(
    scores: np.ndarray, targets: np.ndarray, groups: np.ndarray
) -> float:
    """P(score(a) > score(b) | target(a) < target(b)), half credit for exact ties."""

    concordant = 0.0
    comparable = 0.0
    for start, stop in group_slices(groups):
        target = targets[start:stop]
        value = scores[start:stop]
        better = target[:, None] < target[None, :]
        count = better.sum()
        if not count:
            continue
        difference = value[:, None] - value[None, :]
        concordant += ((difference > 0) & better).sum() + 0.5 * ((difference == 0) & better).sum()
        comparable += count
    return float(concordant / comparable) if comparable else float("nan")


@dataclass(frozen=True)
class RankingModel:
    """A deployed linear ranking function on raw (unstandardized) features."""

    weights: np.ndarray
    feature_names: tuple[str, ...]
    l2: float
    holdout_accuracy: float
    fit_info: dict[str, Any]

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.weights @ features

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": np.asarray(self.weights, dtype=float).tolist(),
            "feature_names": list(self.feature_names),
            "l2": float(self.l2),
            "holdout_pairwise_accuracy": float(self.holdout_accuracy),
            "fit_info": dict(self.fit_info),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RankingModel":
        return cls(
            weights=np.asarray(value["weights"], dtype=np.float64),
            feature_names=tuple(value["feature_names"]),
            l2=float(value["l2"]),
            holdout_accuracy=float(value["holdout_pairwise_accuracy"]),
            fit_info=dict(value.get("fit_info", {})),
        )


def fit_ranking_model(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    names: Sequence[str],
    *,
    l2_grid: Sequence[float],
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> RankingModel:
    """Fit one ranking function, selecting L2 on a held-out tail of the groups."""

    spans = group_slices(groups)
    if len(spans) < 20:
        raise ValueError("Too few calibration groups to fit a ranking model")
    cut = spans[int((1.0 - holdout_fraction) * len(spans))][0]
    mean, scale = standardize(features[:cut])
    standardized = (features - mean) / scale
    differences, weights = build_pairs(
        standardized[:cut], targets[:cut], groups[:cut], seed=seed
    )
    best: RankingModel | None = None
    for l2 in l2_grid:
        theta, _value, info = fit_pairwise_logistic(differences, weights, float(l2))
        accuracy = pairwise_accuracy(standardized[cut:] @ theta, targets[cut:], groups[cut:])
        info["holdout_groups"] = len(spans) - int(np.searchsorted([s for s, _ in spans], cut))
        candidate = RankingModel(
            weights=theta / scale,
            feature_names=tuple(names),
            l2=float(l2),
            holdout_accuracy=float(accuracy),
            fit_info=info,
        )
        if best is None or candidate.holdout_accuracy > best.holdout_accuracy:
            best = candidate
    assert best is not None
    return best
