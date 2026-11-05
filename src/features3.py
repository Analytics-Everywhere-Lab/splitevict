from __future__ import annotations
import numpy as np
from features2 import FeatureState2, NAMES as BASE_NAMES


EXTRA = ("log_pos_in_request", "early_in_request", "request_count", "request_rate",
         "request_rate_shrunk", "request_rate_x_mkh2", "absent_in_request",
         "request_rate_minus_static")
NAMES = BASE_NAMES + EXTRA
SHRINK = 24.0


class FeatureState3(FeatureState2):
    def reset(self):
        super().reset()
        self.seq_cnt = np.zeros((self.L, self.E))
        self.seq_pos = np.zeros(self.L, np.int64)

    def begin_sequence(self):
        self.seq_cnt.fill(0.0)
        self.seq_pos.fill(0)

    def features(self, layer, request, gates, sorted_request, position):
        base = super().features(layer, request, gates, sorted_request, position)
        p = float(self.seq_pos[layer])
        cnt = self.seq_cnt[layer]
        rate = cnt / max(p, 1.0)
        static = self.static[layer]
        shrunk = (cnt + SHRINK * static * 8.0) / (max(p, 1.0) + SHRINK)
        mk_h2 = base[1]
        return np.concatenate([base, np.stack([
            np.full(self.E, np.log1p(p)),
            np.full(self.E, 1.0 / (1.0 + p)),
            cnt,
            rate,
            shrunk,
            rate * mk_h2,
            (cnt == 0).astype(np.float64),
            rate - static * 8.0,
        ])])

    def absorb(self, layer, request, gates, position):
        super().absorb(layer, request, gates, position)
        self.seq_cnt[layer][request] += 1.0
        self.seq_pos[layer] += 1
