"""Two eviction policies interleaved sequence by sequence in one process: a drift-immune A/B (T2).

T1 ran each policy in its own pass, and its drift check found that the host's copy speed stepped by
~17% for the last pass. Here both policies decode every sequence back to back in one process on one
model, alternating which goes first (A,B then B,A). Each policy keeps its own cache state (its own GPU
slot set and policy object) across its own sequences; the host copies of the experts are shared.
Run-level drift then hits both policies equally, and the per-sequence paired comparison measures only
the policy.

    python src/runtime/interleaved_runtime.py --spec data/specs/olmoe_t1_replay_spec.json \
        --capacity 32 --sequences calibration --output data/runs/t2/B32.json \
        --p2-bundle data/rankers/olmoe_frozen_lag_req_trees200.pkl \
        --p1-bundle data/rankers/olmoe_frozen_lag_tok_trees200.pkl
"""

from __future__ import annotations

import argparse
import copy
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

from offload_runtime import OffloadRuntime, _sha256, bandwidth_probe, run_path  # noqa: E402
from policy import make_policy  # noqa: E402

SLOT_ATTRS = ("slot_gate_up", "slot_down", "slot_experts", "slot_of", "free_slots", "pinned_index")


def new_slots(layer, capacity: int, device) -> dict:
    """A second slot set for one layer, sharing the layer's pinned host copies."""

    torch = layer.torch
    pin = device.type == "cuda"
    gate_up = torch.empty((capacity,) + tuple(layer.resident_gate_up.shape[1:]), dtype=layer.resident_gate_up.dtype, device=device)
    down = torch.empty((capacity,) + tuple(layer.resident_down.shape[1:]), dtype=layer.resident_down.dtype, device=device)
    experts = layer.block.experts
    config = copy.deepcopy(experts.config)
    for name in ("num_local_experts", "num_experts"):
        if hasattr(config, name):
            setattr(config, name, int(capacity))
    with torch.device("meta"):
        module = type(experts)(config)
    module.gate_up_proj = torch.nn.Parameter(gate_up, requires_grad=False)
    module.down_proj = torch.nn.Parameter(down, requires_grad=False)
    module.num_experts = int(capacity)
    index = torch.empty((1, int(layer.runtime.spec["architecture"]["top_k"])), dtype=torch.long)
    if pin:
        index = index.pin_memory()
    return {"slot_gate_up": gate_up, "slot_down": down, "slot_experts": module.eval(),
            "slot_of": np.full(layer.E, -1, dtype=np.int64), "free_slots": list(range(capacity - 1, -1, -1)),
            "pinned_index": index}


class Context:
    def __init__(self, name: str, policy, slots: list[dict]):
        self.name, self.policy, self.slots = name, policy, slots
        self.rows: list[dict] = []

    def use(self, runtime) -> None:
        runtime.policy = self.policy
        for layer, slots in zip(runtime.layers, self.slots):
            for attr in SLOT_ATTRS:
                setattr(layer, attr, slots[attr])

    def save(self, runtime) -> None:
        for layer, slots in zip(runtime.layers, self.slots):
            for attr in SLOT_ATTRS:
                slots[attr] = getattr(layer, attr)


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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--sequences", default="calibration", help="'calibration' or a workload path name")
    parser.add_argument("--p2-bundle", type=Path, required=True)
    parser.add_argument("--p1-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from splitphase_service import SplitPhasePolicy, SplitPhaseService

    spec = json.loads(args.spec.read_text())
    arch = spec["architecture"]
    L, E, k = int(arch["num_moe_layers"]), int(arch["num_experts"]), int(arch["top_k"])
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
    kwargs = {"revision": spec["model_revision"], "dtype": torch.bfloat16, "low_cpu_mem_usage": True,
              "attn_implementation": attn}
    if device.type != "cpu":
        kwargs["device_map"] = {"": str(device)}
    model = AutoModelForCausalLM.from_pretrained(spec["model"], **kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["model_revision"])

    lfu = make_policy("lfu_decay", num_layers=L, num_experts=E, capacity=args.capacity, top_k=k)
    runtime = OffloadRuntime(model, spec, device, lfu, int(args.capacity), compute="module")
    ctx_lfu = Context("lfu_decay", lfu, [{a: getattr(layer, a) for a in SLOT_ATTRS} for layer in runtime.layers])
    service = SplitPhaseService(str(args.p2_bundle), str(args.p1_bundle), int(args.capacity), workers=1,
                                num_layers=L, num_experts=E, top_k=k)
    sp = SplitPhasePolicy(service, L, E, int(args.capacity), k)
    ctx_sp = Context("splitphase", sp, [new_slots(layer, int(args.capacity), device) for layer in runtime.layers])

    probes = {}
    for ctx in (ctx_lfu, ctx_sp):
        ctx.use(runtime)
        probes[ctx.name] = bandwidth_probe(runtime.layers[0], device)
        for layer in runtime.layers:  # the probe wrote slot 0; start both caches cold
            layer.slot_of.fill(-1)
            layer.free_slots = list(range(int(args.capacity) - 1, -1, -1))
        ctx.save(runtime)

    def quiet(message: str) -> None:
        pass

    start = time.perf_counter()
    order_log = []
    for i, item in enumerate(items):
        order = (ctx_lfu, ctx_sp) if i % 2 == 0 else (ctx_sp, ctx_lfu)
        for ctx in order:
            ctx.use(runtime)
            result = run_path(runtime, tokenizer, spec, {"name": f"interleaved_{args.sequences}", "regime": "interleaved",
                                                        "sequences": [item]}, teacher_force=True, limit_sequences=None, log=quiet)
            ctx.save(runtime)
            row = dict(result["sequences"][0])
            row["order_slot"] = order.index(ctx)
            row["clock_s"] = time.perf_counter() - start
            ctx.rows.append(row)
        order_log.append([c.name for c in order])
        if (i + 1) % 10 == 0 or i == 0:
            a, b = ctx_lfu.rows[-1], ctx_sp.rows[-1]
            print(f"seq {i + 1}/{len(items)}: lfu {a['wall_ms'] / a['tokens']:.1f} ms/tok ({a['expert_transfers']} xfer), "
                  f"split {b['wall_ms'] / b['tokens']:.1f} ms/tok ({b['expert_transfers']} xfer)", flush=True)
    busy = service.close()
    host = {"hostname": socket.gethostname(), "platform": platform.platform(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()}
    try:
        import transformers

        host["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        pass
    out = {
        "schema_version": "hardware_interleaved_v1", "label": args.label, "model": spec["model"],
        "trace_hash": spec["trace_hash"], "capacity": int(args.capacity), "sequences": args.sequences,
        "design": "both policies decode every sequence back to back in one process, alternating which goes first",
        "host": host, "bandwidth_probes": probes, "order": order_log,
        "splitphase": {"p2_bundle": str(args.p2_bundle), "p1_bundle": str(args.p1_bundle),
                       "p2_bundle_sha256": _sha256(args.p2_bundle), "p1_bundle_sha256": _sha256(args.p1_bundle),
                       "wait_ms_per_token": sp.wait_ns / 1e6 / max(sum(r["tokens"] for r in ctx_sp.rows), 1),
                       "worker_busy_ms_per_token": [b / 1e6 / max(sum(r["tokens"] for r in ctx_sp.rows), 1) for b in busy]},
        "policies": {ctx.name: {"totals": totals(ctx.rows), "sequences": ctx.rows} for ctx in (ctx_lfu, ctx_sp)},
        "elapsed_seconds": time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    for name, entry in out["policies"].items():
        t = entry["totals"]
        print(f"{name}: {t['ms_per_token_wall']:.2f} ms/token wall, copy {t['ms_per_token_copy']:.2f} "
              f"({t['ms_per_copy']:.3f} ms/copy), {t['transfers_per_token']:.2f} xfer/tok, "
              f"routing {100 * t['routing_match_fraction']:.2f}%", flush=True)
    runtime.release()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
