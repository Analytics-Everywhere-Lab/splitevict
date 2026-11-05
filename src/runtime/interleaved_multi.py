"""Several eviction policies interleaved sequence by sequence in one process, on one set of GPU slots (T3).

The drift-immune design of ``interleaved_runtime.py`` (T2), generalised in two ways:

* any number of policies (decayed LFU, split-phase, the synchronous learned ranker), with the order
  rotating every sequence (A,B,C / B,C,A / C,A,B ...), so each policy runs first, second and third
  equally often;
* one shared set of slots. Three slot sets for Qwen3-30B-A3B at B=64 would need 87 GB. Each policy
  keeps its own cache state (slot map, free slots, policy object) across its own sequences, as in a
  workload path; when the policy changes, the slots are refilled from pinned host memory with that
  policy's residents. The refill happens before the sequence's prefill, outside the timed decode.

Everything inside a sequence is ``offload_runtime.run_path``.

    python src/runtime/interleaved_multi.py --spec data/specs/qwen3_30b_a3b_t3_replay_spec.json \
        --capacity 64 --experts-on-host --policies lfu_decay splitphase global_token_gbt \
        --sync-bundle data/rankers/qwen3_30b_a3b_dev_trees800.pkl \
        --p2-bundle data/rankers/qwen3_30b_a3b_frozen_lag_req_trees200.pkl \
        --p1-bundle data/rankers/qwen3_30b_a3b_frozen_lag_tok_trees200.pkl \
        --output data/runs/t3_qwen3_30b_a3b/B64.json
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from offload_runtime import OffloadRuntime, _sha256, bandwidth_probe, load_model, run_path  # noqa: E402
from policy import LFU_DECAY_ALPHA, PolicyBundle, make_policy  # noqa: E402

POLICIES = ("lfu_decay", "splitphase", "global_token_gbt", "lru", "lfu")


class SharedSlotContext:
    """One policy's cache state; the physical slots are shared and refilled on ``use``."""

    def __init__(self, name: str, policy, runtime: OffloadRuntime):
        self.name, self.policy = name, policy
        self.slot_of = [np.full(layer.E, -1, dtype=np.int64) for layer in runtime.layers]
        self.free_slots = [list(range(runtime.capacity - 1, -1, -1)) for _ in runtime.layers]
        self.rows: list[dict] = []
        self.refills = 0
        self.refill_seconds = 0.0

    def use(self, runtime: OffloadRuntime) -> None:
        start = time.perf_counter()
        runtime.policy = self.policy
        for layer, slot_of, free in zip(runtime.layers, self.slot_of, self.free_slots):
            layer.slot_of = slot_of  # the forward mutates these objects in place
            layer.free_slots = free
            for expert in np.flatnonzero(slot_of >= 0):
                slot = int(slot_of[expert])
                layer.slot_gate_up[slot].copy_(layer.host_gate_up[int(expert)], non_blocking=True)
                layer.slot_down[slot].copy_(layer.host_down[int(expert)], non_blocking=True)
        if runtime.cuda:
            runtime.torch.cuda.synchronize()
        self.refills += 1
        self.refill_seconds += time.perf_counter() - start


def rotation(names: list[str], index: int) -> list[str]:
    shift = index % len(names)
    return names[shift:] + names[:shift]


def totals(rows: list[dict]) -> dict:
    keys = ("tokens", "expert_transfers", "bytes_transferred", "wall_ms", "policy_ms", "issue_ms", "copy_ms",
            "routing_events", "routing_matches", "token_agreements", "token_comparisons")
    out = {k: sum(r[k] for r in rows) for k in keys}
    tok = max(out["tokens"], 1)
    out.update({"ms_per_token_wall": out["wall_ms"] / tok, "ms_per_token_policy": out["policy_ms"] / tok,
                "ms_per_token_issue": out["issue_ms"] / tok, "ms_per_token_copy": out["copy_ms"] / tok,
                "transfers_per_token": out["expert_transfers"] / tok,
                "ms_per_copy": out["copy_ms"] / max(out["expert_transfers"], 1),
                "routing_match_fraction": out["routing_matches"] / max(out["routing_events"], 1),
                "token_agreement_fraction": out["token_agreements"] / max(out["token_comparisons"], 1)})
    return out


def run_interleaved(runtime: OffloadRuntime, contexts: list[SharedSlotContext], items: list[dict], decode, log=print):
    """Decode every item with every context, rotating the order; ``decode(item)`` returns one row."""

    start = time.perf_counter()
    names = [c.name for c in contexts]
    by_name = {c.name: c for c in contexts}
    order_log = []
    for i, item in enumerate(items):
        order = rotation(names, i)
        for slot, name in enumerate(order):
            ctx = by_name[name]
            ctx.use(runtime)
            row = dict(decode(item))
            row["order_slot"] = slot
            row["clock_s"] = time.perf_counter() - start
            ctx.rows.append(row)
        order_log.append(order)
        if (i + 1) % 10 == 0 or i == 0:
            parts = ", ".join(f"{c.name} {c.rows[-1]['wall_ms'] / max(c.rows[-1]['tokens'], 1):.1f} ms/tok "
                              f"({c.rows[-1]['expert_transfers']} xfer)" for c in contexts)
            log(f"seq {i + 1}/{len(items)}: {parts}")
    return order_log, time.perf_counter() - start


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--policies", nargs="+", default=["lfu_decay", "splitphase", "global_token_gbt"], choices=POLICIES)
    parser.add_argument("--sequences", default="calibration", help="'calibration' (the spec's id list) or a workload path name")
    parser.add_argument("--sync-bundle", type=Path, help="global_token_gbt: the synchronous learned ranker")
    parser.add_argument("--p2-bundle", type=Path)
    parser.add_argument("--p1-bundle", type=Path)
    parser.add_argument("--alpha", type=float, default=LFU_DECAY_ALPHA, help="decayed LFU decay (the paper's 0.95)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--experts-on-host", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    spec = json.loads(args.spec.read_text())
    arch = spec["architecture"]
    L, E, k = int(arch["num_moe_layers"]), int(arch["num_experts"]), int(arch["top_k"])
    if not k < args.capacity <= E:
        raise SystemExit("capacity must satisfy top_k < B <= E")
    by_id = {int(s["sequence_id"]): s for s in spec["sequences"]}
    if args.sequences == "calibration":
        ids = [int(v) for v in spec["calibration_sequence_ids"]]
    else:
        ids = [int(s["source_sequence_id"]) for s in {w["name"]: w for w in spec["workloads"]}[args.sequences]["sequences"]]
    if args.limit:
        ids = ids[: args.limit]
    items = [{"source_sequence_id": sid, "position": i, "domain": by_id[sid]["domain"]} for i, sid in enumerate(ids)]

    device = torch.device(args.device)
    torch.manual_seed(int(spec["trace_generation"]["seed"]))
    attn = spec["trace_generation"].get("attn_implementation", "eager")
    model = load_model(spec, device, torch.bfloat16, attn, None, args.experts_on_host, log=lambda m: print(m, flush=True))
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["model_revision"])

    service = None
    policies = {}
    for name in args.policies:
        if name == "splitphase":
            from splitphase_service import SplitPhasePolicy, SplitPhaseService

            service = SplitPhaseService(str(args.p2_bundle), str(args.p1_bundle), int(args.capacity), workers=args.workers,
                                        num_layers=L, num_experts=E, top_k=k)
            policies[name] = SplitPhasePolicy(service, L, E, int(args.capacity), k)
        elif name == "global_token_gbt":
            bundle = PolicyBundle.load(args.sync_bundle)
            policies[name] = make_policy(name, num_layers=L, num_experts=E, capacity=int(args.capacity), top_k=k,
                                         bundle=bundle, backend="numba", score_scope="candidates")
        else:
            policies[name] = make_policy(name, num_layers=L, num_experts=E, capacity=int(args.capacity), top_k=k,
                                         alpha=args.alpha)
    first = policies[args.policies[0]]
    runtime = OffloadRuntime(model, spec, device, first, int(args.capacity), compute="module",
                             experts_on_host=args.experts_on_host)
    probe = bandwidth_probe(runtime.layers[0], device)
    contexts = [SharedSlotContext(name, policies[name], runtime) for name in args.policies]

    def quiet(message: str) -> None:
        pass

    def decode(item):
        result = run_path(runtime, tokenizer, spec, {"name": f"interleaved_{args.sequences}", "regime": "interleaved",
                                                    "sequences": [item]}, teacher_force=True, limit_sequences=None, log=quiet)
        return result["sequences"][0]

    order_log, elapsed = run_interleaved(runtime, contexts, items, decode, log=lambda m: print(m, flush=True))
    busy = service.close() if service is not None else None
    host = {"hostname": socket.gethostname(), "platform": platform.platform(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "experts_on_host": bool(args.experts_on_host)}
    try:
        import transformers

        host["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        pass
    out = {
        "schema_version": "hardware_interleaved_multi_v1", "label": args.label, "model": spec["model"],
        "trace_hash": spec["trace_hash"], "capacity": int(args.capacity), "sequences": args.sequences,
        "design": "every policy decodes every sequence back to back in one process, order rotating per sequence; "
                  "one shared slot set refilled from host memory when the policy changes (untimed)",
        "host": host, "bandwidth_probe": probe, "order": order_log, "alpha": args.alpha,
        "bundles": {"sync": None if args.sync_bundle is None else {"path": str(args.sync_bundle), "sha256": _sha256(args.sync_bundle)},
                    "p2": None if args.p2_bundle is None else {"path": str(args.p2_bundle), "sha256": _sha256(args.p2_bundle)},
                    "p1": None if args.p1_bundle is None else {"path": str(args.p1_bundle), "sha256": _sha256(args.p1_bundle)}},
        "policies": {c.name: {"totals": totals(c.rows), "sequences": c.rows, "refills": c.refills,
                              "refill_seconds": c.refill_seconds} for c in contexts},
        "elapsed_seconds": elapsed,
    }
    if service is not None:
        sp = policies["splitphase"]
        tokens = max(sum(r["tokens"] for r in contexts[args.policies.index("splitphase")].rows), 1)
        out["splitphase"] = {"workers": args.workers, "wait_ms_per_token": sp.wait_ns / 1e6 / tokens,
                             "decide_ms_per_token": sp.decide_ns / 1e6 / tokens, "post_ms_per_token": sp.post_ns / 1e6 / tokens,
                             "worker_busy_ms_per_token": [b / 1e6 / tokens for b in busy]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    for name, entry in out["policies"].items():
        t = entry["totals"]
        print(f"{name}: {t['ms_per_token_wall']:.2f} ms/token wall, policy {t['ms_per_token_policy']:.2f}, copy "
              f"{t['ms_per_token_copy']:.2f} ({t['ms_per_copy']:.3f} ms/copy), {t['transfers_per_token']:.2f} xfer/tok, "
              f"routing {100 * t['routing_match_fraction']:.2f}%, refills {entry['refill_seconds']:.0f}s", flush=True)
    runtime.release()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
