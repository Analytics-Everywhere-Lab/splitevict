import numpy as np
from race_stage1.models import same_layer_indices
from race_stage2.diagnostics import build_future_occurrences

def simulate_multi(trace, workload, capacities, state, scorers, collect_to=None, stride=13):
    L, E, C = trace.num_layers, trace.num_experts, len(capacities)
    req_all = trace.requested_expert_ids.astype(np.int64); srt = np.sort(req_all, axis=1)
    gate_all = trace.router_weights.astype(np.float64)
    res = np.zeros((C, L, E), bool); lu = np.full((L, E), -1, np.int64)
    pos = np.zeros(L, np.int64); mask = np.zeros(E, bool)
    spares = [c - trace.top_k for c in capacities]
    misses = np.zeros(C, np.int64)
    fut = build_future_occurrences(trace, workload, same_layer_indices(trace, workload)) \
        if collect_to is not None else None
    state.reset(); n = 0
    for _s, view in workload.iter_slices(trace):
        state.begin_sequence()
        for idx in range(view.start, view.stop):
            o = int(trace.layer_index[idx]); r = req_all[idx]; p = int(pos[o])
            td = fut.advance(o, p, r) if fut is not None else None
            f = state.features(o, r, gate_all[idx], srt[idx], p)
            scores = [sc(f) for sc in scorers]
            state.absorb(o, r, gate_all[idx], p)
            sample = collect_to is not None and (n % stride == 0) and p >= 40
            mask[r] = True; lu[o, r] = p
            for ci, sp in enumerate(spares):
                row = res[ci, o]; cand = np.flatnonzero(row & ~mask)
                misses[ci] += r.size - int(row[r].sum())
                if cand.size > sp:
                    if sample and cand.size >= 4:
                        collect_to[ci][0].append(f[:, cand].T.astype(np.float32))
                        collect_to[ci][1].append(np.minimum(td[cand], 33).astype(np.float32))
                        collect_to[ci][2].append(np.full(cand.size, collect_to[ci][3], np.int32))
                        collect_to[ci][3] += 1
                    s = scores[ci][cand]
                    ordr = np.lexsort((cand, -lu[o, cand], -s))
                    row.fill(False); row[r] = True; row[cand[ordr[:sp]]] = True
                else:
                    row[r] = True
            mask[r] = False; pos[o] += 1; n += 1
    return {int(c): int(m) for c, m in zip(capacities, misses)}

