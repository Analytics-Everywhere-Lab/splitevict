from __future__ import annotations

import numpy as np

from race_stage1.models import same_layer_indices

TOKEN_NAMES = ("tok_is_repeat", "tok_log_gap", "tok_prev_req",
               "tok_propensity", "tok_next_propensity")
SHRINK = 8.0


def layer_streams(trace, workload):
    streams = same_layer_indices(trace, workload)
    requests = trace.requested_expert_ids.astype(np.int64)
    gates = trace.router_weights.astype(np.float64)
    return ([np.asarray(trace.token_id[s], dtype=np.int64) for s in streams],
            [requests[s] for s in streams],
            [gates[s] for s in streams])


def fit_token_tables(trace, workload, num_layers, num_experts, popularity):
    toks, reqs, _gates = layer_streams(trace, workload)
    seen = np.unique(np.concatenate(toks))
    index = {int(t): i for i, t in enumerate(seen)}
    cur = np.zeros((len(seen), num_layers, num_experts))
    nxt = np.zeros((len(seen), num_layers, num_experts))
    hits = np.zeros((len(seen), num_layers))
    for layer in range(num_layers):
        tl, rl = toks[layer], reqs[layer]
        for p in range(len(tl)):
            row = index[int(tl[p])]
            cur[row, layer][rl[p]] += 1.0
            hits[row, layer] += 1.0
            if p + 1 < len(rl):
                nxt[row, layer][rl[p + 1]] += 1.0
    prior = popularity / np.maximum(popularity.sum(axis=1, keepdims=True), 1.0)
    denom = hits[:, :, None] + SHRINK
    cur = (cur + SHRINK * prior[None]) / denom
    nxt = (nxt + SHRINK * prior[None]) / denom
    return index, cur, nxt


