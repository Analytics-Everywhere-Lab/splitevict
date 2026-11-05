"""T9 check on the GPU before calibration: graph replay reproduces the pieces run eagerly, and its speed.

On the first ``--sequences`` calibration sequences of a spec, decayed LFU decodes every sequence three ways, each
from cold slots: the graphed decode step (``interleaved_graphs``), the same pieces run eagerly without graphs, and the eager
runtime of ``interleaved_freerun.py``. Graph replay launches the kernels the pieces launch, so it must give exactly their
tokens; the check fails otherwise. The eager runtime's routing against the trace (100% expected) and the graphed
step's are reported, with the milliseconds per token of all three.

    python src/runtime/check_graphs.py --spec data/specs/olmoe_t5_replay_spec.json --capacity 32 \
        --output data/runs/t9/olmoe/graph_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import interleaved_graphs  # noqa: E402
from interleaved_freerun import free_running_run_path  # noqa: E402
from offload_runtime import load_model  # noqa: E402
from policy import LFU_DECAY_ALPHA, make_policy  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    spec = json.loads(args.spec.read_text())
    arch = spec["architecture"]
    L, E, k = int(arch["num_moe_layers"]), int(arch["num_experts"]), int(arch["top_k"])
    device = torch.device("cuda")
    torch.manual_seed(int(spec["trace_generation"]["seed"]))
    attn = spec["trace_generation"].get("attn_implementation", "eager")
    model = load_model(spec, device, torch.bfloat16, attn, None, False, log=lambda m: print(m, flush=True))
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["model_revision"])
    by_id = {int(s["sequence_id"]): s for s in spec["sequences"]}
    ids = [int(v) for v in spec["calibration_sequence_ids"]][: args.sequences]
    workload = {"name": "graph_check", "regime": "check",
                "sequences": [{"source_sequence_id": sid, "position": i, "domain": by_id[sid]["domain"]}
                              for i, sid in enumerate(ids)]}

    def lfu():
        return make_policy("lfu_decay", num_layers=L, num_experts=E, capacity=args.capacity, top_k=k, alpha=LFU_DECAY_ALPHA)

    runtime = interleaved_graphs.GraphedRuntime(model, spec, device, lfu(), args.capacity, compute="module", copy_api="driver")
    decoder = runtime.graphed_decoder
    results = {}
    for name, graphed, run_path in (("graphed", True, interleaved_graphs.graphed_run_path),
                                    ("pieces_eager", False, interleaved_graphs.graphed_run_path),
                                    ("eager_runtime", None, free_running_run_path)):
        if graphed is not None:
            decoder.graphed = graphed
        runtime.start_path(lfu())
        rows = run_path(runtime, tokenizer, spec, workload, teacher_force=False, limit_sequences=None,
                        log=lambda m: None)["sequences"]
        tokens = sum(r["tokens"] for r in rows)
        results[name] = {"ms_per_token": sum(r["wall_ms"] for r in rows) / tokens, "tokens": tokens,
                         "transfers_per_token": sum(r["expert_transfers"] for r in rows) / tokens,
                         "routing_match_fraction": sum(r["routing_matches"] for r in rows) / sum(r["routing_events"] for r in rows),
                         "token_agreement_fraction": sum(r["token_agreements"] for r in rows)
                         / max(sum(r["token_comparisons"] for r in rows), 1),
                         "generated_tokens": [r.get("generated_tokens") for r in rows]}
        print(f"{name}: {results[name]['ms_per_token']:.2f} ms/token, {results[name]['transfers_per_token']:.2f} xfer/token, "
              f"routing vs trace {100 * results[name]['routing_match_fraction']:.2f}%", flush=True)
    decoder.graphed = True
    runtime.release()
    same = results["graphed"]["generated_tokens"] == results["pieces_eager"]["generated_tokens"]
    out = {"spec": str(args.spec), "capacity": args.capacity, "sequences": ids, "capture_seconds": decoder.capture_seconds,
           "graph_equals_pieces": same, "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
           "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    print(f"graph replay reproduces the pieces' tokens: {same}; wrote {args.output}", flush=True)
    if not same:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
