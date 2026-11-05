r"""Prepare a non-OLMoE routing trace for the learned ranker (``run_caer.py --prepared-root``).

For OLMoE, ``measurement/`` supplies the trace, the calibration/evaluation split, the workload
paths, the transition models and the seed weights. For another model, ``prepare`` re-derives
each of them from the new trace's calibration sequences with the same functions
(``freeze_evaluation``, ``fit_transition_models``, ``fit_ranking_model``) and writes them under
``--prepared-root``. Two rules are generalized to other expert counts and top-k values; see
``feature_state_class`` and ``simulate_multi``.

    python src/crossmodel.py --repository-root measurement \
        --trace-root data/traces/mixtral_8x7b --config configs/eval_mixtral_8x7b.json \
        --prepared-root data/prepared/mixtral_8x7b
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HORIZONS = (1, 2, 4, 8, 16, 32)
SEED_STRIDE = 13
SEED_WARMUP = 40
SEED_L2_GRID = (0.03, 0.01, 0.003, 0.001)
PREPARED_SCHEMA = "caer_crossmodel_prepared_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def feature_state_class(top_k: int):
    """``features3.FeatureState3`` with the top-k read from the trace, not hard-coded.

    The request-scope features in ``features3`` scale an expert's calibration share by
    the literal ``8.0`` (OLMoE's top-k) to form its expected per-event request rate.
    ``FeatureState3K`` uses ``trace.top_k`` instead; with top-k 8 the two are identical.
    """

    from features2 import FeatureState2
    from features3 import NAMES, SHRINK

    rate_scale = float(top_k)
    request_top_k = int(top_k)

    class FeatureState3K(FeatureState2):
        top_k = request_top_k
        names = NAMES

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
            shrunk = (cnt + SHRINK * static * rate_scale) / (max(p, 1.0) + SHRINK)
            mk_h2 = base[1]
            return np.concatenate(
                [
                    base,
                    np.stack(
                        [
                            np.full(self.E, np.log1p(p)),
                            np.full(self.E, 1.0 / (1.0 + p)),
                            cnt,
                            rate,
                            shrunk,
                            rate * mk_h2,
                            (cnt == 0).astype(np.float64),
                            rate - static * rate_scale,
                        ]
                    ),
                ]
            )

        def absorb(self, layer, request, gates, position):
            super().absorb(layer, request, gates, position)
            self.seq_cnt[layer][request] += 1.0
            self.seq_pos[layer] += 1

    return FeatureState3K


def simulate_multi(trace, workload, capacities, state, scorers, collect_to=None, stride=13):
    """``run4.simulate_multi`` with the group-size rule generalized to small expert counts.

    ``run4`` adds a candidate set to the training data only when it holds at least four
    experts. With 64 experts and at least four spare slots this never binds; with
    Mixtral's eight experts and one or two spare slots it would discard every decision.
    This version requires ``spare + 1`` candidates, capped at four. That is the same rule
    on OLMoE, and on any model a group is any decision that has to evict something.
    """

    from race_stage1.models import same_layer_indices
    from race_stage2.diagnostics import build_future_occurrences

    L, E, C = trace.num_layers, trace.num_experts, len(capacities)
    req_all = trace.requested_expert_ids.astype(np.int64)
    srt = np.sort(req_all, axis=1)
    gate_all = trace.router_weights.astype(np.float64)
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    l2o = {layer: ordinal for ordinal, layer in enumerate(layers)}
    res = np.zeros((C, L, E), bool)
    lu = np.full((L, E), -1, np.int64)
    pos = np.zeros(L, np.int64)
    mask = np.zeros(E, bool)
    spares = [c - trace.top_k for c in capacities]
    min_group = [min(4, sp + 1) for sp in spares]
    misses = np.zeros(C, np.int64)
    fut = (
        build_future_occurrences(trace, workload, same_layer_indices(trace, workload))
        if collect_to is not None
        else None
    )
    state.reset()
    n = 0
    for _s, view in workload.iter_slices(trace):
        state.begin_sequence()
        for idx in range(view.start, view.stop):
            o = l2o[int(trace.layer_index[idx])]
            r = req_all[idx]
            p = int(pos[o])
            td = fut.advance(o, p, r) if fut is not None else None
            f = state.features(o, r, gate_all[idx], srt[idx], p)
            scores = [sc(f) for sc in scorers]
            state.absorb(o, r, gate_all[idx], p)
            sample = collect_to is not None and (n % stride == 0) and p >= SEED_WARMUP
            mask[r] = True
            lu[o, r] = p
            for ci, sp in enumerate(spares):
                row = res[ci, o]
                cand = np.flatnonzero(row & ~mask)
                misses[ci] += r.size - int(row[r].sum())
                if cand.size > sp:
                    if sample and cand.size >= min_group[ci]:
                        collect_to[ci][0].append(f[:, cand].T.astype(np.float32))
                        collect_to[ci][1].append(np.minimum(td[cand], 33).astype(np.float32))
                        collect_to[ci][2].append(np.full(cand.size, collect_to[ci][3], np.int32))
                        collect_to[ci][3] += 1
                    s = scores[ci][cand]
                    ordr = np.lexsort((cand, -lu[o, cand], -s))
                    row.fill(False)
                    row[r] = True
                    row[cand[ordr[:sp]]] = True
                else:
                    row[r] = True
            mask[r] = False
            pos[o] += 1
            n += 1
    if fut is not None:
        fut.finish()
    return {int(c): int(m) for c, m in zip(capacities, misses)}


@dataclass(frozen=True)
class CrossModelInputs:
    trace: Any
    calibration: Any
    workloads: tuple
    frozen: dict[str, Any]
    prepared: dict[str, Any]
    root: Path


def _workloads_from_frozen(frozen: Mapping[str, Any]):
    from race_stage1.frozen import workload_from_record

    calibration = workload_from_record(frozen["calibration_workload"])
    workloads = tuple(workload_from_record(item) for item in frozen["workloads"])
    return calibration, workloads


def load_trace(trace_root: Path):
    from residency_headroom.trace import RoutingTrace

    return RoutingTrace.load(trace_root / "routing_trace.npz", validate=True)


def prepare(trace_root: Path, config_path: Path, prepared_root: Path) -> dict[str, Any]:
    """Freeze the split, fit the transition models and the seed weights. Idempotent.

    The seed weights are a pooled pairwise linear ranker. They drive row collection for
    the learned ranker, as the OLMoE seed weights do.
    """

    from race_stage1.models import fit_transition_models
    from race_stage3.ranking import fit_ranking_model
    from race_stage3.simulation import Stage1WinnerScorer
    from residency_headroom.common import atomic_write_json, load_config
    from residency_headroom.freeze import freeze_evaluation
    from residency_headroom.workloads import calibration_frequency_scores
    from race_stage1.models import TransitionModels

    prepared_root.mkdir(parents=True, exist_ok=True)
    manifest_path = prepared_root / "prepared.json"
    trace = load_trace(trace_root)
    config = load_config(config_path)
    expected = config["expected_architecture"]
    if expected.get("num_moe_layers") is None:
        expected["num_moe_layers"] = int(trace.num_layers)
    if int(expected["num_experts"]) != trace.num_experts or int(expected["top_k"]) != trace.top_k:
        raise SystemExit("config expected_architecture does not match the trace")

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["trace_hash"] != trace.trace_hash:
            raise SystemExit(f"{manifest_path} was prepared for a different trace")
        return manifest

    print(f"[prepare] {config['model']}: {trace.num_events:,} events, "
          f"E={trace.num_experts} k={trace.top_k} L={trace.num_layers}", flush=True)
    start = time.perf_counter()
    frozen = freeze_evaluation(trace, config, prepared_root / "frozen")
    calibration, workloads = _workloads_from_frozen(frozen)
    print(f"[prepare] frozen split: calibration {len(calibration.sequences)} sequences, "
          f"{len(workloads)} workload paths [{time.perf_counter() - start:.0f}s]", flush=True)

    model_path = prepared_root / "transition_models.npz"
    start = time.perf_counter()
    models = fit_transition_models(trace, calibration, HORIZONS)
    model_meta = models.save(model_path)
    TransitionModels.load(model_path)
    print(f"[prepare] transition models H={HORIZONS} [{time.perf_counter() - start:.0f}s]",
          flush=True)

    capacities = tuple(int(v) for v in config["cache_capacities"])
    decision = tuple(c for c in capacities if trace.top_k < c < trace.num_experts)
    static = calibration_frequency_scores(trace, calibration).astype(np.float64)
    state_type = feature_state_class(trace.top_k)
    state = state_type(models, trace.num_layers, trace.num_experts, static)
    winner = Stage1WinnerScorer(models, trace.num_layers, trace.num_experts)

    class _Seed:
        # Stage 1 winner as the round-1 collection policy, fed through simulate_multi.
        def __init__(self):
            self.pending = None

        def __call__(self, features):
            return self.pending

    start = time.perf_counter()
    buckets = [[[], [], [], 0] for _ in decision]
    original_features = state.features

    def features_with_winner(layer, request, gates, sorted_request, position):
        winner.observe(layer, request, gates, sorted_request, position)
        return original_features(layer, request, gates, sorted_request, position)

    state.features = features_with_winner  # type: ignore[method-assign]
    winner.reset()
    scorer = lambda features: winner(features)
    simulate_multi(trace, calibration, decision, state, [scorer] * len(decision),
                   collect_to=buckets, stride=SEED_STRIDE)
    state.features = original_features  # type: ignore[method-assign]
    features, targets, groups = [], [], []
    offset = 0
    for bucket in buckets:
        x = np.concatenate(bucket[0]).astype(np.float64)
        y = np.concatenate(bucket[1]).astype(np.float64)
        g = np.concatenate(bucket[2])
        features.append(x)
        targets.append(y)
        groups.append(g + offset)
        offset += int(g.max()) + 1
    x_all = np.concatenate(features)
    y_all = np.concatenate(targets)
    g_all = np.concatenate(groups)
    ranking = fit_ranking_model(x_all, y_all, g_all, state_type.names, l2_grid=SEED_L2_GRID)
    seed = {
        "procedure": "round-1 pooled pairwise ranking model, collected under the Stage 1 "
                     "winner (0.5*Markov-H2 + 0.5*EWMA) on the calibration workload",
        "decision_capacities": list(decision),
        "stride": SEED_STRIDE,
        "warmup_events_skipped": SEED_WARMUP,
        "l2_grid": list(SEED_L2_GRID),
        "groups": int(offset),
        "rows": int(x_all.shape[0]),
        "model": ranking.as_dict(),
    }
    seed_path = prepared_root / "seed_weights.json"
    atomic_write_json(seed_path, seed)
    print(f"[prepare] seed weights: {offset:,} groups, {x_all.shape[0]:,} rows, "
          f"l2={ranking.l2}, holdout pairwise accuracy {100 * ranking.holdout_accuracy:.2f}% "
          f"[{time.perf_counter() - start:.0f}s]", flush=True)

    manifest = {
        "schema_version": PREPARED_SCHEMA,
        "model": config["model"],
        "trace_root": str(trace_root.resolve()),
        "trace_hash": trace.trace_hash,
        "routing_trace_npz_sha256": sha256_file(trace_root / "routing_trace.npz"),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "architecture": {
            "num_moe_layers": trace.num_layers,
            "num_experts": trace.num_experts,
            "top_k": trace.top_k,
        },
        "events": int(trace.num_events),
        "cache_capacities": list(capacities),
        "decision_capacities": list(decision),
        "frozen_evaluation_config_hash": frozen["config_hash"],
        "frozen_evaluation_config_sha256": sha256_file(
            prepared_root / "frozen" / "frozen_evaluation_config.json"
        ),
        "calibration_sequences": len(calibration.sequences),
        "calibration_workload_hash": calibration.hash,
        "workloads": {w.name: {"regime": w.regime, "sequences": len(w.sequences), "hash": w.hash}
                      for w in workloads},
        "evaluation_sequences": sum(len(w.sequences) for w in workloads),
        "transition_models": model_meta,
        "seed_weights_sha256": sha256_file(seed_path),
        "seed_holdout_pairwise_accuracy": float(ranking.holdout_accuracy),
        "seed_l2": float(ranking.l2),
        "generalisations": {
            "request_scope_rate_scale": f"static share x top_k ({trace.top_k}); OLMoE code uses 8.0",
            "collection_min_group": "min(4, spare + 1) candidates; OLMoE code uses 4",
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def load(prepared_root: Path):
    """Return ``(inputs, transition_models)`` shaped like ``harness.load()``."""

    from race_stage1.models import TransitionModels
    from residency_headroom.common import read_json

    manifest = read_json(prepared_root / "prepared.json")
    trace = load_trace(Path(manifest["trace_root"]))
    if trace.trace_hash != manifest["trace_hash"]:
        raise SystemExit("trace changed since it was prepared")
    frozen = read_json(prepared_root / "frozen" / "frozen_evaluation_config.json")
    if frozen["config_hash"] != manifest["frozen_evaluation_config_hash"]:
        raise SystemExit("frozen evaluation config changed since it was prepared")
    calibration, workloads = _workloads_from_frozen(frozen)
    models = TransitionModels.load(prepared_root / "transition_models.npz")
    inputs = CrossModelInputs(trace, calibration, workloads, frozen, manifest, prepared_root)
    return inputs, models


def seed_weights(prepared_root: Path) -> np.ndarray:
    seed = json.loads((prepared_root / "seed_weights.json").read_text())
    return np.asarray(seed["model"]["weights"], dtype=np.float64)


def references(inputs, workload, capacities):
    """Oracle and strongest simple baseline, exactly as ``harness.references``."""

    from residency_headroom.simulator import simulate_oracle, simulate_policy
    from residency_headroom.workloads import calibration_frequency_scores

    out: dict[str, Any] = {}
    out["oracle"] = {c: int(simulate_oracle(inputs.trace, workload, c).misses) for c in capacities}
    static = calibration_frequency_scores(inputs.trace, inputs.calibration).astype(np.float64)
    simple = {}
    for name, kw in (("lru", {}), ("lfu", {}), ("lfu_decay", {"alpha": 0.95}),
                     ("static_hotset", {"static_scores": static})):
        simple[name] = {c: int(simulate_policy(inputs.trace, workload, c, name, **kw).misses)
                        for c in capacities}
    out["simple_all"] = simple
    out["simple"] = {c: min(simple[n][c] for n in simple) for c in capacities}
    out["simple_which"] = {c: min(simple, key=lambda n: simple[n][c]) for c in capacities}
    return out


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="prepare a non-OLMoE trace for run_caer.py")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_caer import configure_repository

    configure_repository(args.repository_root)
    manifest = prepare(args.trace_root, args.config, args.prepared_root)
    print(json.dumps({k: manifest[k] for k in ("model", "trace_hash", "decision_capacities",
                                               "seed_holdout_pairwise_accuracy")}, indent=2))


if __name__ == "__main__":
    main()
