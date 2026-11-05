from __future__ import annotations
import numpy as np

HORIZONS = (1, 2, 4, 8, 16, 32)
CAP = 33
RING = 32
BIG = 4096.0

NAMES = (

    "mk_h1", "mk_h2", "mk_h4", "mk_h8", "mk_h16", "mk_h32",

    "band_1", "band_2_4", "band_4_8", "band_8_16", "band_16_32",

    "nor_h2", "nor_h8",

    "prevmk_h2", "prevmk_h8",

    "log_tau", "recip_tau", "never_seen",
    "log_gap1", "log_gap2", "log_overdue", "overdue_clip",

    "cnt4", "cnt8", "cnt16", "cnt32",

    "freq_a90", "freq_a95", "freq_a99", "gate_ewma", "last_gate", "in_prev_request",

    "static_pop",

    "mk_h2_x_recip_tau", "mk_h8_x_cnt16", "freq_a95_x_recip_tau", "mk_h2_x_freq_a95",
)


class FeatureState2:
    def __init__(self, models, num_layers, num_experts, static_pop):
        self.M = np.stack([np.stack([models.matrix(h, l) for h in HORIZONS])
                           for l in range(num_layers)])
        self.M.flags.writeable = False
        self.L, self.E = num_layers, num_experts
        self.static = static_pop / np.maximum(static_pop.sum(axis=1, keepdims=True), 1.0)
        self.reset()

    def reset(self):
        L, E = self.L, self.E
        self.last = np.full((L, E), -1, np.int64)
        self.prev_last = np.full((L, E), -1, np.int64)
        self.prev2_last = np.full((L, E), -1, np.int64)
        self.f90 = np.zeros((L, E)); self.f95 = np.zeros((L, E)); self.f99 = np.zeros((L, E))
        self.gate = np.zeros((L, E)); self.lastgate = np.zeros((L, E))
        self.ring = np.zeros((L, RING, E), bool)
        self.cnt = np.zeros((4, L, E), np.int32)
        self.prev_req = [None] * L

    def features(self, layer, request, gates, sorted_request, position):
        block = self.M[layer][:, sorted_request, :]
        mk = block.mean(axis=1)
        nor2 = 1.0 - np.prod(1.0 - block[1], axis=0)
        nor8 = 1.0 - np.prod(1.0 - block[3], axis=0)
        pr = self.prev_req[layer]
        if pr is None:
            pmk2 = np.zeros(self.E); pmk8 = np.zeros(self.E)
        else:
            pblock = self.M[layer][:, pr, :]
            pmk2 = pblock[1].mean(axis=0); pmk8 = pblock[3].mean(axis=0)

        seen = self.last[layer] >= 0
        tau = np.where(seen, np.maximum(position - self.last[layer], 1), BIG)
        has1 = self.prev_last[layer] >= 0
        gap1 = np.where(has1, np.maximum(self.last[layer] - self.prev_last[layer], 1), BIG)
        has2 = self.prev2_last[layer] >= 0
        gap2 = np.where(has2, np.maximum(self.prev_last[layer] - self.prev2_last[layer], 1), BIG)
        overdue = tau / gap1
        prev = np.zeros(self.E)
        if pr is not None:
            prev[pr] = 1.0
        c4, c8, c16, c32 = self.cnt[0][layer], self.cnt[1][layer], self.cnt[2][layer], self.cnt[3][layer]
        f95 = self.f95[layer]
        return np.stack([
            *mk,
            mk[0], mk[2] - mk[1], mk[3] - mk[2], mk[4] - mk[3], mk[5] - mk[4],
            nor2, nor8,
            pmk2, pmk8,
            np.log1p(tau), 1.0 / tau, (~seen).astype(np.float64),
            np.log1p(gap1), np.log1p(gap2), np.log1p(overdue), np.minimum(overdue, 8.0),
            c4.astype(np.float64), c8.astype(np.float64),
            c16.astype(np.float64), c32.astype(np.float64),
            self.f90[layer], f95, self.f99[layer],
            self.gate[layer], self.lastgate[layer], prev,
            self.static[layer],
            mk[1] / tau, mk[3] * c16, f95 / tau, mk[1] * f95,
        ])

    def absorb(self, layer, request, gates, position):
        slot = position % RING

        for i, w in enumerate((4, 8, 16, 32)):
            out = position - w
            if out >= 0:
                self.cnt[i][layer][self.ring[layer, out % RING]] -= 1
            self.cnt[i][layer][request] += 1
        self.ring[layer, slot].fill(False)
        self.ring[layer, slot][request] = True
        self.f90[layer] *= 0.90; self.f90[layer][request] += 1.0
        self.f95[layer] *= 0.95; self.f95[layer][request] += 1.0
        self.f99[layer] *= 0.99; self.f99[layer][request] += 1.0
        self.gate[layer] *= 0.95; self.gate[layer][request] += 0.05 * gates
        self.lastgate[layer][request] = gates
        self.prev2_last[layer][request] = self.prev_last[layer][request]
        self.prev_last[layer][request] = self.last[layer][request]
        self.last[layer, request] = position
        self.prev_req[layer] = request
