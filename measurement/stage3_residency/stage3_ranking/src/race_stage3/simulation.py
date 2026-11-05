from __future__ import annotations

import numpy as np

from race_stage1.models import TransitionModels


class Stage1WinnerScorer:
    """The frozen Stage 1 winner, reproduced with its own arithmetic and update order.

    Stage 1 computes ``0.5 * Markov-H2 + 0.5 * history`` where ``history`` is a
    request-indicator EWMA at alpha 0.95 that is updated with the current event
    *before* the retention decision. Reproducing that exactly needs Stage 1's own
    recurrence and its own summation order, so this scorer keeps its own state
    instead of reading the shared feature block. It exists to prove that the Stage 3
    eviction mechanism is the Stage 1 mechanism, and is never used as a policy.
    """

    def __init__(self, models: TransitionModels, num_layers: int, num_experts: int,
                 beta: float = 0.5, alpha: float = 0.95) -> None:
        self.matrix = np.stack([models.matrix(2, layer) for layer in range(num_layers)])
        self.matrix.flags.writeable = False
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.history = np.zeros((num_layers, num_experts), dtype=np.float64)
        self._score = np.zeros(num_experts, dtype=np.float64)

    def reset(self) -> None:
        self.history.fill(0.0)

    def observe(self, layer: int, request: np.ndarray, gates: np.ndarray,
                sorted_request: np.ndarray, position: int) -> None:
        conditional = self.matrix[layer][sorted_request].mean(axis=0)
        row = self.history[layer]
        row *= self.alpha
        row[request] += 1.0 - self.alpha
        self._score = self.beta * conditional + (1.0 - self.beta) * row

    def __call__(self, block: np.ndarray) -> np.ndarray:
        return self._score


