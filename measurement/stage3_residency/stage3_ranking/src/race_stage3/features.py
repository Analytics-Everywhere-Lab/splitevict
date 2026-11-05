"""Causal per-expert features for the Stage 3 ranking model.

Every value is derived only from same-layer events that have already been observed,
plus immutable calibration-fitted artifacts. The state object is advanced by
``absorb`` strictly after ``features`` has been read for the same event, so no
feature can ever see its own event's consequences.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from race_stage1.models import TransitionModels


HORIZONS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
CAP = 33
"""Capped next-use distance for an expert unused within the next 32 same-layer events."""

RING = 32
NEVER = 4096.0
SHRINK = 24.0

CONTEXT_NAMES = (
    "mk_h1", "mk_h2", "mk_h4", "mk_h8", "mk_h16", "mk_h32",
    "band_1", "band_2_4", "band_4_8", "band_8_16", "band_16_32",
    "nor_h2", "nor_h8",
    "prevmk_h2", "prevmk_h8",
)
RENEWAL_NAMES = (
    "log_tau", "recip_tau", "never_seen",
    "log_gap1", "log_gap2", "log_overdue", "overdue_clip",
)
WINDOW_NAMES = ("cnt4", "cnt8", "cnt16", "cnt32")
DECAY_NAMES = ("freq_a90", "freq_a95", "freq_a99", "gate_ewma", "last_gate", "in_prev_request")
STATIC_NAMES = ("static_pop",)
INTERACTION_NAMES = ("mk_h2_x_recip_tau", "mk_h8_x_cnt16", "freq_a95_x_recip_tau", "mk_h2_x_freq_a95")
REQUEST_SCOPE_NAMES = (
    "log_pos_in_request", "early_in_request", "request_count", "request_rate",
    "request_rate_shrunk", "request_rate_x_mkh2", "absent_in_request",
    "request_rate_minus_static",
)

BASE_NAMES = (
    CONTEXT_NAMES + RENEWAL_NAMES + WINDOW_NAMES + DECAY_NAMES + STATIC_NAMES + INTERACTION_NAMES
)
ALL_NAMES = BASE_NAMES + REQUEST_SCOPE_NAMES
BASE_COUNT = len(BASE_NAMES)
ALL_COUNT = len(ALL_NAMES)


def feature_names(include_request_scope: bool = True) -> tuple[str, ...]:
    return ALL_NAMES if include_request_scope else BASE_NAMES


class FeatureState:
    """Per-layer causal feature state for every expert in a layer."""

    def __init__(
        self,
        models: TransitionModels,
        num_layers: int,
        num_experts: int,
        static_popularity: np.ndarray,
        include_request_scope: bool = True,
    ) -> None:
        missing = [h for h in HORIZONS if h not in models.horizons]
        if missing:
            raise ValueError(f"Transition models lack Stage 3 horizons {missing}")
        if models.num_layers != num_layers or models.num_experts != num_experts:
            raise ValueError("Transition models do not match the trace architecture")
        popularity = np.asarray(static_popularity, dtype=np.float64)
        if popularity.shape != (num_layers, num_experts):
            raise ValueError("Static popularity must be [layer, expert]")
        self.markov = np.stack(
            [np.stack([models.matrix(h, layer) for h in HORIZONS]) for layer in range(num_layers)]
        )
        self.markov.flags.writeable = False
        self.static = popularity / np.maximum(popularity.sum(axis=1, keepdims=True), 1.0)
        self.static.flags.writeable = False
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.include_request_scope = bool(include_request_scope)
        self.size = ALL_COUNT if include_request_scope else BASE_COUNT
        self.reset()

    def reset(self) -> None:
        """Clear all adaptive state at a frozen workload-path boundary."""

        layers, experts = self.num_layers, self.num_experts
        self.last = np.full((layers, experts), -1, dtype=np.int64)
        self.prev_last = np.full((layers, experts), -1, dtype=np.int64)
        self.prev2_last = np.full((layers, experts), -1, dtype=np.int64)
        self.freq90 = np.zeros((layers, experts))
        self.freq95 = np.zeros((layers, experts))
        self.freq99 = np.zeros((layers, experts))
        self.gate = np.zeros((layers, experts))
        self.last_gate = np.zeros((layers, experts))
        self.ring = np.zeros((layers, RING, experts), dtype=bool)
        self.counts = np.zeros((4, layers, experts), dtype=np.int32)
        self.previous_request: list[np.ndarray | None] = [None] * layers
        self.request_count = np.zeros((layers, experts))
        self.request_position = np.zeros(layers, dtype=np.int64)

    def begin_request(self) -> None:
        """Reset the within-decode-request statistics at a request boundary."""

        self.request_count.fill(0.0)
        self.request_position.fill(0)

    def features(
        self,
        layer: int,
        request: np.ndarray,
        gates: np.ndarray,
        sorted_request: np.ndarray,
        position: int,
    ) -> np.ndarray:
        """Feature block ``[feature, expert]`` for the decision at ``position``."""

        block = self.markov[layer][:, sorted_request, :]
        mk = block.mean(axis=1)
        previous = self.previous_request[layer]
        if previous is None:
            prev_mk2 = np.zeros(self.num_experts)
            prev_mk8 = np.zeros(self.num_experts)
            prev_indicator = np.zeros(self.num_experts)
        else:
            prev_block = self.markov[layer][:, previous, :]
            prev_mk2 = prev_block[1].mean(axis=0)
            prev_mk8 = prev_block[3].mean(axis=0)
            prev_indicator = np.zeros(self.num_experts)
            prev_indicator[previous] = 1.0

        seen = self.last[layer] >= 0
        tau = np.where(seen, np.maximum(position - self.last[layer], 1), NEVER)
        gap1 = np.where(
            self.prev_last[layer] >= 0,
            np.maximum(self.last[layer] - self.prev_last[layer], 1),
            NEVER,
        )
        gap2 = np.where(
            self.prev2_last[layer] >= 0,
            np.maximum(self.prev_last[layer] - self.prev2_last[layer], 1),
            NEVER,
        )
        overdue = tau / gap1
        counts = [self.counts[index][layer].astype(np.float64) for index in range(4)]
        freq95 = self.freq95[layer]

        rows = [
            *mk,
            mk[0], mk[2] - mk[1], mk[3] - mk[2], mk[4] - mk[3], mk[5] - mk[4],
            1.0 - np.prod(1.0 - block[1], axis=0),
            1.0 - np.prod(1.0 - block[3], axis=0),
            prev_mk2, prev_mk8,
            np.log1p(tau), 1.0 / tau, (~seen).astype(np.float64),
            np.log1p(gap1), np.log1p(gap2), np.log1p(overdue), np.minimum(overdue, 8.0),
            *counts,
            self.freq90[layer], freq95, self.freq99[layer],
            self.gate[layer], self.last_gate[layer], prev_indicator,
            self.static[layer],
            mk[1] / tau, mk[3] * counts[2], freq95 / tau, mk[1] * freq95,
        ]
        if self.include_request_scope:
            step = float(self.request_position[layer])
            inside = self.request_count[layer]
            rate = inside / max(step, 1.0)
            static_rate = self.static[layer] * 8.0
            shrunk = (inside + SHRINK * static_rate) / (max(step, 1.0) + SHRINK)
            rows += [
                np.full(self.num_experts, np.log1p(step)),
                np.full(self.num_experts, 1.0 / (1.0 + step)),
                inside,
                rate,
                shrunk,
                rate * mk[1],
                (inside == 0).astype(np.float64),
                rate - static_rate,
            ]
        return np.stack(rows)

    def absorb(self, layer: int, request: np.ndarray, gates: np.ndarray, position: int) -> None:
        """Advance the state with the event that has just been served."""

        for index, window in enumerate((4, 8, 16, 32)):
            leaving = position - window
            if leaving >= 0:
                self.counts[index][layer][self.ring[layer, leaving % RING]] -= 1
            self.counts[index][layer][request] += 1
        slot = self.ring[layer, position % RING]
        slot.fill(False)
        slot[request] = True
        self.freq90[layer] *= 0.90
        self.freq90[layer][request] += 1.0
        self.freq95[layer] *= 0.95
        self.freq95[layer][request] += 1.0
        self.freq99[layer] *= 0.99
        self.freq99[layer][request] += 1.0
        self.gate[layer] *= 0.95
        self.gate[layer][request] += 0.05 * gates
        self.last_gate[layer][request] = gates
        self.prev2_last[layer][request] = self.prev_last[layer][request]
        self.prev_last[layer][request] = self.last[layer][request]
        self.last[layer, request] = position
        self.previous_request[layer] = request
        self.request_count[layer][request] += 1.0
        self.request_position[layer] += 1


def static_popularity(trace, workload) -> np.ndarray:
    """Calibration-path expert request counts, per layer."""

    from residency_headroom.workloads import calibration_frequency_scores

    return calibration_frequency_scores(trace, workload).astype(np.float64)
