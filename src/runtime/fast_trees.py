"""Small-batch inference for scikit-learn histogram gradient-boosting ensembles.

The deployed eviction policy scores one layer-event at a time: at most ``E`` rows
through an 800-tree regressor and a few dozen rows through a 250-tree classifier,
sixteen times per generated token.  ``HistGradientBoosting*.predict`` walks the trees in
a Python loop, one call per tree, so its cost at this batch size is dominated by
per-call overhead and not by the arithmetic.  ``FlatForest`` stores every tree of the
ensemble in one padded node table and walks all trees level-synchronously with a
handful of vectorised gathers per depth level, in numpy or compiled with numba.

Predictions are identical to scikit-learn's to floating-point summation order: the
thresholds, leaf values and missing-value routing are copied from the fitted
predictors, not re-derived.  ``raw`` returns the ensemble sum plus the baseline; the
link (identity for the regressor, logistic for the binary classifier) is applied by the
caller through ``predict`` / ``predict_proba``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _special_math_sigmoid(raw: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-raw))


@dataclass
class FlatForest:
    """Padded node tables for one fitted ``HistGradientBoosting{Regressor,Classifier}``."""

    feature: np.ndarray  # (trees, nodes) int32
    threshold: np.ndarray  # (trees, nodes) float64
    left: np.ndarray  # (trees, nodes) int32
    right: np.ndarray  # (trees, nodes) int32
    is_leaf: np.ndarray  # (trees, nodes) bool
    missing_left: np.ndarray  # (trees, nodes) bool
    value: np.ndarray  # (trees, nodes) float64
    baseline: float
    max_depth: int
    n_features: int
    link: str  # "identity" or "logistic"
    node_count: np.ndarray  # (trees,) real nodes per tree before padding

    @classmethod
    def from_sklearn(cls, model: Any) -> "FlatForest":
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        if isinstance(model, HistGradientBoostingClassifier):
            if int(model.n_trees_per_iteration_) != 1:
                raise ValueError("only binary classifiers are supported")
            link = "logistic"
            classes = np.asarray(model.classes_)
            positive = np.flatnonzero(classes == 1)
            if positive.size != 1:
                raise ValueError("classifier does not expose a unique positive class")
        elif isinstance(model, HistGradientBoostingRegressor):
            link = "identity"
        else:
            raise TypeError(f"unsupported model {type(model).__name__}")
        if getattr(model, "is_categorical_", None) is not None and np.any(
            np.asarray(model.is_categorical_)
        ):
            raise ValueError("categorical splits are not supported")

        predictors = [iteration[0] for iteration in model._predictors]
        nodes_per_tree = max(int(p.nodes.shape[0]) for p in predictors)
        trees = len(predictors)
        feature = np.zeros((trees, nodes_per_tree), dtype=np.int32)
        threshold = np.zeros((trees, nodes_per_tree), dtype=np.float64)
        left = np.zeros((trees, nodes_per_tree), dtype=np.int32)
        right = np.zeros((trees, nodes_per_tree), dtype=np.int32)
        is_leaf = np.ones((trees, nodes_per_tree), dtype=bool)
        missing_left = np.zeros((trees, nodes_per_tree), dtype=bool)
        value = np.zeros((trees, nodes_per_tree), dtype=np.float64)
        max_depth = 0
        node_count = np.zeros(trees, dtype=np.int64)
        for index, predictor in enumerate(predictors):
            table = predictor.nodes
            count = int(table.shape[0])
            node_count[index] = count
            feature[index, :count] = table["feature_idx"]
            threshold[index, :count] = table["num_threshold"]
            left[index, :count] = table["left"]
            right[index, :count] = table["right"]
            is_leaf[index, :count] = table["is_leaf"].astype(bool)
            missing_left[index, :count] = table["missing_go_to_left"].astype(bool)
            value[index, :count] = table["value"]
            max_depth = max(max_depth, int(table["depth"].max()))
        baseline = float(np.asarray(model._baseline_prediction).ravel()[0])
        return cls(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
            is_leaf=is_leaf,
            missing_left=missing_left,
            value=value,
            baseline=baseline,
            max_depth=max_depth,
            n_features=int(model.n_features_in_),
            link=link,
            node_count=node_count,
        )

    # ------------------------------------------------------------------ numpy
    def raw(self, rows: np.ndarray) -> np.ndarray:
        """Ensemble raw prediction for ``rows`` shaped ``(n, n_features)``."""

        x = np.asarray(rows, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"rows must have shape (n, {self.n_features})")
        n = x.shape[0]
        trees = self.feature.shape[0]
        tree_index = np.arange(trees, dtype=np.int32)[:, None]
        row_index = np.arange(n)[None, :]
        node = np.zeros((trees, n), dtype=np.int32)
        for _level in range(self.max_depth):
            leaf = self.is_leaf[tree_index, node]
            if leaf.all():
                break
            feature = self.feature[tree_index, node]
            observed = x[row_index, feature]
            go_left = observed <= self.threshold[tree_index, node]
            missing = np.isnan(observed)
            if missing.any():
                go_left = np.where(missing, self.missing_left[tree_index, node], go_left)
            child = np.where(go_left, self.left[tree_index, node], self.right[tree_index, node])
            node = np.where(leaf, node, child)
        leaf_value = self.value[tree_index, node]
        return self.baseline + leaf_value.sum(axis=0)

    def predict(self, rows: np.ndarray) -> np.ndarray:
        raw = self.raw(rows)
        if self.link == "identity":
            return raw
        raise ValueError("use predict_proba for a classifier")

    def predict_proba_positive(self, rows: np.ndarray) -> np.ndarray:
        if self.link != "logistic":
            raise ValueError("predict_proba_positive needs a classifier")
        return _special_math_sigmoid(self.raw(rows))

    # ------------------------------------------------------------------ numba
    def to_numba(self) -> "NumbaForest":
        return NumbaForest(self)


class NumbaForest:
    """Tree-major compiled walk: every tree's nodes stay hot while its rows are scored.

    Node tables are concatenated into flat arrays with per-tree offsets.  The walk is
    the scikit-learn one (``x <= threshold`` goes left, NaN follows ``missing_go_to_left``)
    and the leaf values are summed in tree order, so the result equals
    ``HistGradientBoosting*._raw_predict`` up to floating-point summation.
    """

    def __init__(self, forest: FlatForest) -> None:
        try:
            import numba  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("numba is required for NumbaForest; pip install numba") from exc
        real = [int(v) for v in forest.node_count]
        offsets = np.zeros(len(real) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(real)
        total = int(offsets[-1])
        self.feature = np.empty(total, dtype=np.int64)
        self.threshold = np.empty(total, dtype=np.float64)
        self.left = np.empty(total, dtype=np.int64)
        self.right = np.empty(total, dtype=np.int64)
        self.is_leaf = np.empty(total, dtype=np.bool_)
        self.missing_left = np.empty(total, dtype=np.bool_)
        self.value = np.empty(total, dtype=np.float64)
        for index, size in enumerate(real):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            self.feature[start:stop] = forest.feature[index, :size]
            self.threshold[start:stop] = forest.threshold[index, :size]
            self.left[start:stop] = forest.left[index, :size]
            self.right[start:stop] = forest.right[index, :size]
            self.is_leaf[start:stop] = forest.is_leaf[index, :size]
            self.missing_left[start:stop] = forest.missing_left[index, :size]
            self.value[start:stop] = forest.value[index, :size]
        self.offsets = offsets
        self.baseline = float(forest.baseline)
        self.n_features = int(forest.n_features)
        self.link = forest.link
        self._walk = _numba_walk()

    def raw(self, rows: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(rows, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"rows must have shape (n, {self.n_features})")
        return self._walk(
            x, self.offsets, self.feature, self.threshold, self.left, self.right,
            self.is_leaf, self.missing_left, self.value, self.baseline,
        )

    def predict(self, rows: np.ndarray) -> np.ndarray:
        if self.link != "identity":
            raise ValueError("use predict_proba_positive for a classifier")
        return self.raw(rows)

    def predict_proba_positive(self, rows: np.ndarray) -> np.ndarray:
        if self.link != "logistic":
            raise ValueError("predict_proba_positive needs a classifier")
        return _special_math_sigmoid(self.raw(rows))


_WALK = None


def _numba_walk():
    global _WALK
    if _WALK is not None:
        return _WALK
    from numba import njit

    @njit(cache=False, nogil=True)
    def walk(x, offsets, feature, threshold, left, right, is_leaf, missing_left, value, baseline):
        n = x.shape[0]
        trees = offsets.shape[0] - 1
        out = np.full(n, baseline)
        for t in range(trees):
            root = offsets[t]
            for r in range(n):
                node = root
                while not is_leaf[node]:
                    observed = x[r, feature[node]]
                    if observed != observed:
                        node = root + (left[node] if missing_left[node] else right[node])
                    elif observed <= threshold[node]:
                        node = root + left[node]
                    else:
                        node = root + right[node]
                out[r] += value[node]
        return out

    _WALK = walk
    return walk
