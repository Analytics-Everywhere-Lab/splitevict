"""Core operations of the second-stage reranker (CAER, cutoff-aware expert reranking).

The reranker (``caer_gbt``) rescores the candidates around the learned ranker's eviction
cutoff. This module is deliberately independent of the trace simulator and scikit-learn.
Deployable feature construction receives only causal feature blocks and the fitted global
ranker's scores. Future reuse distances are accepted by the separate
``offline_keep_labels`` function only, which keeps the training-label boundary explicit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


CONTEXT_FEATURE_NAMES = (
    "base_score",
    "base_rank_fraction",
    "signed_cutoff_distance",
    "margin_to_last_keep",
    "margin_to_first_evict",
    "base_cutoff_gap",
    "candidate_count_fraction",
    "cache_capacity_fraction",
)


RICH_CONTEXT_FEATURE_NAMES = (
    "base_score_zscore",
    "base_score_gap_above",
    "base_score_gap_below",
    "candidate_score_spread",
    "keep_fraction",
)

CROSS_LAYER_FEATURE_NAMES = ("xlayer_propensity_1", "xlayer_propensity_2")


def boundary_feature_names(
    base_names: Sequence[str],
    extra_names: Sequence[str] = (),
    rich_context: bool = False,
    set_context: bool = False,
) -> tuple[str, ...]:
    """Return the complete, ordered feature schema of the second-stage reranker.

    ``extra_names`` are stage-2-only blocks. They come after the cutoff context, so they
    never shift the earlier columns.
    """

    names = tuple(str(name) for name in base_names)
    return (
        names
        + CONTEXT_FEATURE_NAMES
        + (RICH_CONTEXT_FEATURE_NAMES if rich_context else ())
        + (tuple(f"setdev_{name}" for name in names) if set_context else ())
        + tuple(str(name) for name in extra_names)
    )


def extract_boundary_window(
    order: np.ndarray,
    keep_count: int,
    radius: int = 2,
) -> np.ndarray | None:
    """Select exactly ``radius`` items on each side of the predicted cutoff.

    ``order`` contains candidate-local indices from most to least preferred by the
    global ranker.  Events without a complete symmetric window are ineligible and
    return ``None``; a caller should retain the global ordering for those events.
    """

    ranked = np.asarray(order, dtype=np.int64)
    keep = int(keep_count)
    width = int(radius)
    if ranked.ndim != 1:
        raise ValueError("order must be one-dimensional")
    if width <= 0:
        raise ValueError("radius must be positive")
    if keep < 0 or keep > ranked.size:
        raise ValueError("keep_count must lie within the ranking")
    if np.unique(ranked).size != ranked.size:
        raise ValueError("order must not contain duplicate candidate indices")
    lower, upper = keep - width, keep + width
    if lower < 0 or upper > ranked.size:
        return None
    return ranked[lower:upper].copy()


def offline_keep_labels(
    expert_ids: np.ndarray,
    future_distances: np.ndarray,
    last_used: np.ndarray,
    *,
    keep_count: int = 2,
    distance_cap: int | None = 33,
) -> np.ndarray:
    """Build offline KEEP labels for one boundary window.

    This is the only operation in this module that consumes future information.
    Smaller future distance is better.  Ties follow the cache ordering convention:
    most recently used expert first, then smaller expert id.  Exactly ``keep_count``
    rows receive label one.
    """

    experts = np.asarray(expert_ids, dtype=np.int64)
    distance = np.asarray(future_distances)
    recency = np.asarray(last_used, dtype=np.int64)
    if experts.ndim != 1:
        raise ValueError("expert_ids must be one-dimensional")
    if distance.shape != experts.shape or recency.shape != experts.shape:
        raise ValueError("expert_ids, future_distances, and last_used must align")
    keep = int(keep_count)
    if keep < 0 or keep > experts.size:
        raise ValueError("keep_count must lie within the boundary window")
    if distance_cap is not None:
        distance = np.minimum(distance, int(distance_cap))
    oracle_order = np.lexsort((experts, -recency, distance))
    labels = np.zeros(experts.size, dtype=np.uint8)
    labels[oracle_order[:keep]] = 1
    return labels


def build_boundary_features(
    causal_block: np.ndarray,
    base_scores: np.ndarray,
    candidates: np.ndarray,
    order: np.ndarray,
    keep_count: int,
    capacity: int,
    *,
    radius: int = 2,
    num_experts: int | None = None,
    window: np.ndarray | None = None,
    extra_block: np.ndarray | None = None,
    rich_context: bool = False,
    set_context: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Construct deployable rows for a stage-2 candidate set.

    Parameters contain no future trace information.  ``causal_block`` is shaped
    ``[feature, expert]`` and ``base_scores`` is the global retention score for all
    experts.  The returned local indices address ``candidates``.

    By default the candidate set is the symmetric ``radius`` window around the predicted
    cutoff.  Passing ``window`` supplies the set explicitly, which lets the identical
    stage-2 model be applied to every candidate instead; ``radius`` then serves only to
    normalize the signed cutoff distance.
    """

    block = np.asarray(causal_block, dtype=np.float64)
    scores = np.asarray(base_scores, dtype=np.float64)
    candidate_ids = np.asarray(candidates, dtype=np.int64)
    ranked = np.asarray(order, dtype=np.int64)
    if block.ndim != 2:
        raise ValueError("causal_block must have shape [feature, expert]")
    if scores.ndim != 1 or scores.size != block.shape[1]:
        raise ValueError("base_scores must have one value per expert")
    if candidate_ids.ndim != 1:
        raise ValueError("candidates must be one-dimensional")
    if ranked.size != candidate_ids.size:
        raise ValueError("order must rank every candidate")
    if candidate_ids.size and (
        int(candidate_ids.min()) < 0 or int(candidate_ids.max()) >= block.shape[1]
    ):
        raise ValueError("candidate expert id is outside the causal feature block")

    if window is None:
        window = extract_boundary_window(ranked, keep_count, radius)
        if window is None:
            return None
    else:
        window = np.asarray(window, dtype=np.int64)
        if window.ndim != 1 or window.size == 0:
            raise ValueError("an explicit window must be a non-empty 1-D index array")

    expert_count = int(num_experts) if num_experts is not None else block.shape[1]
    if expert_count <= 0:
        raise ValueError("num_experts must be positive")
    keep = int(keep_count)
    candidate_scores = scores[candidate_ids]
    rank_of = np.empty(ranked.size, dtype=np.int64)
    rank_of[ranked] = np.arange(ranked.size, dtype=np.int64)
    local_rank = rank_of[window]
    selected_experts = candidate_ids[window]
    selected_scores = candidate_scores[window]
    last_keep_score = float(candidate_scores[ranked[keep - 1]])
    first_evict_score = float(candidate_scores[ranked[keep]])
    cutoff_gap = last_keep_score - first_evict_score
    denom = max(ranked.size - 1, 1)
    context = np.column_stack(
        (
            selected_scores,
            local_rank / denom,
            (keep - 0.5 - local_rank) / float(radius),
            selected_scores - last_keep_score,
            selected_scores - first_evict_score,
            np.full(window.size, cutoff_gap),
            np.full(window.size, ranked.size / float(expert_count)),
            np.full(window.size, int(capacity) / float(expert_count)),
        )
    )
    if rich_context:
        spread = float(candidate_scores.max() - candidate_scores.min())
        deviation = float(candidate_scores.std())
        centre = float(candidate_scores.mean())
        ordered = candidate_scores[ranked]
        above = np.empty(ranked.size, dtype=np.float64)
        below = np.empty(ranked.size, dtype=np.float64)
        above[0] = 0.0
        above[1:] = ordered[:-1] - ordered[1:]
        below[-1] = 0.0
        below[:-1] = ordered[:-1] - ordered[1:]
        gap_above = np.empty(ranked.size, dtype=np.float64)
        gap_below = np.empty(ranked.size, dtype=np.float64)
        gap_above[ranked] = above
        gap_below[ranked] = below
        scale = deviation if deviation > 0 else 1.0
        context = np.column_stack(
            (
                context,
                (selected_scores - centre) / scale,
                gap_above[window],
                gap_below[window],
                np.full(window.size, spread),
                np.full(window.size, keep / float(max(ranked.size, 1))),
            )
        )
    parts = [block[:, selected_experts].T, context]
    if set_context:
        # A univariate scorer rates each expert from its own features alone and cannot
        # express "this candidate is unusual for this candidate set".  Deviation from
        # the set mean makes the scorer multivariate at negligible cost.
        pool = block[:, candidate_ids]
        centre = pool.mean(axis=1)
        scale = pool.std(axis=1)
        scale[scale <= 0.0] = 1.0
        parts.append(
            ((block[:, selected_experts] - centre[:, None]) / scale[:, None]).T
        )
    if extra_block is not None:
        extra = np.asarray(extra_block, dtype=np.float64)
        if extra.ndim != 2 or extra.shape[1] != block.shape[1]:
            raise ValueError("extra_block must have shape [feature, expert]")
        parts.append(extra[:, selected_experts].T)
    rows = np.concatenate(parts, axis=1)
    return rows, window


def combine_boundary_scores(
    base_scores: np.ndarray,
    boundary_scores: np.ndarray,
    *,
    mode: str = "replacement",
    strength: float = 1.0,
) -> np.ndarray:
    """Return replacement or residual scores for one extracted window."""

    base = np.asarray(base_scores, dtype=np.float64)
    correction = np.asarray(boundary_scores, dtype=np.float64)
    if base.shape != correction.shape:
        raise ValueError("base_scores and boundary_scores must align")
    if mode == "replacement":
        return correction.copy()
    if mode == "residual":
        return base + float(strength) * correction
    raise ValueError("mode must be 'replacement' or 'residual'")


def rerank_boundary(
    order: np.ndarray,
    window: np.ndarray,
    boundary_scores: np.ndarray,
) -> np.ndarray:
    """Rerank only ``window`` and preserve every outside position exactly."""

    ranked = np.asarray(order, dtype=np.int64)
    selected = np.asarray(window, dtype=np.int64)
    score = np.asarray(boundary_scores, dtype=np.float64)
    if ranked.ndim != 1 or selected.ndim != 1 or score.ndim != 1:
        raise ValueError("order, window, and boundary_scores must be one-dimensional")
    if selected.size != score.size:
        raise ValueError("window and boundary_scores must align")
    positions = np.flatnonzero(np.isin(ranked, selected))
    if positions.size != selected.size or not np.array_equal(ranked[positions], selected):
        raise ValueError("window must be an ordered contiguous slice of order")
    if positions.size and not np.array_equal(
        positions, np.arange(positions[0], positions[0] + positions.size)
    ):
        raise ValueError("window must be contiguous in order")
    # Stable base-window position is the deterministic fallback for equal scores.
    local_order = np.lexsort((np.arange(selected.size), -score))
    out = ranked.copy()
    out[positions] = selected[local_order]
    return out


def selected_keep_labels(
    window: np.ndarray,
    final_order: np.ndarray,
    keep_count: int,
) -> np.ndarray:
    """Encode the final KEEP/EVICT decision for rows in ``window``."""

    selected = np.asarray(window, dtype=np.int64)
    ranked = np.asarray(final_order, dtype=np.int64)
    retained = set(int(value) for value in ranked[: int(keep_count)])
    return np.asarray([int(value) in retained for value in selected], dtype=np.uint8)
