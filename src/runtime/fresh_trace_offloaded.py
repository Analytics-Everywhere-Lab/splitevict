"""A new routing trace on unseen prompts (T3 to T7) and its replay spec; by default the experts stay in host memory.

The recipe of ``fresh_trace.py``: the trace generator's own decode loop in ``measurement/``, the model's
original trace config (``configs/trace_<model>.json``: pinned model and dataset revisions, bf16, eager
attention, greedy decode, 128 new tokens, seed 42), on prompts no earlier trace used: examples
``offset .. offset + n - 1`` of each domain's pinned shuffle instead of 0-99. The one difference is where
the expert weights live. The generator's model loader is wrapped in memory: the model loads on the CPU,
everything but the expert weights moves to the GPU, the experts stay pinned on the host, and
``offload_runtime.OffloadRuntime`` serves them (prefill through a staging buffer, decode through
``capacity`` slots under decayed LFU). Placement does not change the arithmetic: the experts module of
the model's own class computes every MoE output (``tests/test_hardware.py``, ``HostExpertsTests``).
``--resident`` keeps every expert on the GPU with the generator's own loader instead. Nothing in
``measurement/`` is modified.

``spec`` builds the replay spec without the ten workload paths (too few prompts for them): one path,
``fresh_interleaved``, takes the sequences in domain round-robin order.

    # on the GPU (needs measurement/, torch, transformers 5.16.1, datasets)
    python src/runtime/fresh_trace_offloaded.py generate --opt-root measurement \
        --base-config configs/trace_qwen3_30b_a3b.json --prompts-per-domain 20 --capacity 64 \
        --output-dir data/traces/qwen3_30b_a3b_t3
    # anywhere
    python src/runtime/fresh_trace_offloaded.py spec --opt-root measurement \
        --trace-root data/traces/qwen3_30b_a3b_t3 --output data/specs/qwen3_30b_a3b_t3_replay_spec.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PROMPT_OFFSET = 100
PATH_NAME = "fresh_interleaved"
_RUNTIMES: list = []  # keeps the offload runtime alive for the generator's lifetime


def _paths(opt_root: Path) -> None:
    for relative in ("stage3_residency/stage1_prediction/src", "stage3_residency/src", "src"):
        value = str((opt_root / relative).resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def fresh_config(base_config: Path, offset: int, per_domain: int, capacity: int, resident: bool = False) -> dict:
    config = json.loads(base_config.read_text())
    where = ("every expert resident (the trace repository's own loader)" if resident else
             f"experts pinned on the host (offload runtime, decayed LFU, B={capacity})")
    config["stage"] = (f"Fresh unseen trace, {config['model']}: examples {offset}-{offset + per_domain - 1} of each "
                       f"pinned shuffle; {where}")
    config["num_prompts_per_domain"] = int(per_domain)
    config["prompt_offset"] = int(offset)
    config["offloaded_generation"] = (None if resident else
                                      {"experts_on_host": True, "decode_capacity": int(capacity), "policy": "lfu_decay"})
    return config


def _layer_indices(model) -> list[int]:
    return [i for i, layer in enumerate(model.model.layers) if hasattr(getattr(layer, "mlp", None), "experts")]


def offloaded_loader(original, capacity: int):
    """Wrap the trace generator's ``load_model_and_tokenizer``: load on the CPU, then serve the
    experts from pinned host memory through the offload runtime."""

    import torch

    from offload_runtime import OffloadRuntime, place_non_experts
    from policy import make_policy

    def load(checkpoint, runtime, revision=None, cache_dir=None, local_files_only=False, attn_implementation=None):
        cpu = dataclasses.replace(runtime, device=torch.device("cpu"))
        bundle = original(checkpoint, cpu, revision=revision, cache_dir=cache_dir,
                          local_files_only=local_files_only, attn_implementation=attn_implementation)
        model = bundle.model
        layers = _layer_indices(model)
        experts = model.model.layers[layers[0]].mlp.experts
        num_experts = int(experts.gate_up_proj.shape[0])
        top_k = int(model.model.layers[layers[0]].mlp.gate.top_k)
        expert_bytes = int(experts.gate_up_proj[0].numel() * experts.gate_up_proj.element_size()
                           + experts.down_proj[0].numel() * experts.down_proj.element_size())
        counts = place_non_experts(model, runtime.device, layers)
        print(f"offloaded load: {counts['moved_bytes'] / 1e9:.2f} GB on {runtime.device}, "
              f"{counts['host_bytes'] / 1e9:.2f} GB of experts pinned on the host", flush=True)
        spec = {"architecture": {"layer_indices": layers, "num_moe_layers": len(layers), "num_experts": num_experts,
                                 "top_k": top_k, "expert_bytes": expert_bytes}}
        policy = make_policy("lfu_decay", num_layers=len(layers), num_experts=num_experts, capacity=int(capacity), top_k=top_k)
        offload = OffloadRuntime(model, spec, runtime.device, policy, int(capacity), compute="module", experts_on_host=True)
        offload.mode = "auto"  # the generator's own loop: a multi-token call is the prefill
        _RUNTIMES.append(offload)
        return dataclasses.replace(bundle, runtime=runtime)

    return load


def generate(opt_root: Path, base_config: Path, output_dir: Path, offset: int, per_domain: int, capacity: int,
             resident: bool = False) -> None:
    _paths(opt_root)
    import residency_headroom.trace_generation as tg

    original_examples = tg.load_domain_examples

    def shifted(*, domain, num_examples, seed, **kwargs):
        examples = original_examples(domain=domain, num_examples=num_examples + offset, seed=seed, **kwargs)
        if len(examples.texts) != num_examples + offset:
            raise RuntimeError(f"{domain}: only {len(examples.texts)} examples for offset {offset}")
        examples.texts = examples.texts[offset:]
        metadata = dict(examples.metadata)
        metadata["selected_example_ids"] = list(metadata["selected_example_ids"])[offset:]
        metadata["prompt_offset"] = int(offset)
        examples.metadata = metadata
        return examples

    tg.load_domain_examples = shifted
    if not resident:  # with --resident, a model that fits the GPU (OLMoE) keeps the generator's own loader, as in T1
        tg.load_model_and_tokenizer = offloaded_loader(tg.load_model_and_tokenizer, capacity)
    config = fresh_config(base_config, offset, per_domain, capacity, resident)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fresh_trace_config.json").write_text(json.dumps(config, indent=2) + "\n")
    trace = tg.generate_trace(config, output_dir.resolve())
    validation = trace.validate()
    print(f"fresh trace: {validation['sequences']} sequences, {validation['generated_tokens']} tokens, "
          f"{validation['events']} events, hash={validation['trace_hash']}", flush=True)
    prompts = {}
    for domain in config["domains"]:
        examples = shifted(domain=domain, num_examples=int(config["num_prompts_per_domain"]), seed=int(config["seed"]),
                           revision=config["dataset_revisions"][domain],
                           include_answers=bool(config["include_reference_answers"]),
                           allow_substitution=bool(config["allow_dataset_substitution"]), format_style="legacy")
        records = [s for s in trace.metadata["sequences"] if s["domain"] == domain]
        for record, text in zip(records, examples.texts):
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != record["prompt_sha256"]:
                raise RuntimeError(f"prompt hash mismatch for fresh sequence {record['sequence_id']}")
            prompts[int(record["sequence_id"])] = text
    (output_dir / "fresh_prompts.json").write_text(json.dumps(prompts) + "\n")
    print(f"wrote {output_dir / 'fresh_prompts.json'} ({len(prompts)} prompts)", flush=True)


def interleaved_order(sequences: list[dict], domains: list[str]) -> list[dict]:
    """Domain round-robin: general, coding, math, reasoning, general, ... (each domain in prompt order)."""

    by_domain = {d: [s for s in sequences if s["domain"] == d] for d in domains}
    order = []
    for i in range(max(len(v) for v in by_domain.values())):
        for d in domains:
            if i < len(by_domain[d]):
                order.append(by_domain[d][i])
    return order


def build_spec(opt_root: Path, trace_root: Path, output: Path, original_metadata: list[Path] | None = None) -> dict:
    """The hardware replay spec (``build_replay_spec.py`` schema) with the single path."""

    import numpy as np

    _paths(opt_root)
    from residency_headroom.trace import RoutingTrace

    trace = RoutingTrace.load(trace_root / "routing_trace.npz", validate=True)
    config = json.loads((trace_root / "fresh_trace_config.json").read_text())
    prompts = {int(k): v for k, v in json.loads((trace_root / "fresh_prompts.json").read_text()).items()}
    meta = trace.metadata
    records = {int(s["sequence_id"]): s for s in meta["sequences"]}
    for sid, record in records.items():
        if hashlib.sha256(prompts[sid].encode("utf-8")).hexdigest() != record["prompt_sha256"]:
            raise SystemExit(f"fresh prompt hash mismatch for sequence {sid}")
    overlap = None
    if original_metadata:
        if isinstance(original_metadata, Path):
            original_metadata = [original_metadata]
        old = set()
        for path in original_metadata:  # every earlier trace of the model: none may share a prompt
            old |= {s["prompt_sha256"] for s in json.loads(path.read_text())["sequences"]}
        overlap = len(old & {s["prompt_sha256"] for s in meta["sequences"]})
        if overlap:
            raise SystemExit(f"fresh prompts overlap an earlier trace: {overlap}")
    slices = trace.sequence_slices()
    layers = [int(v) for v in meta["layer_indices"]]
    sequences = []
    for sid, record in sorted(records.items()):
        view = slices[sid]
        sequences.append({
            "sequence_id": sid, "domain": record["domain"], "prompt_index": int(record["prompt_index"]),
            "prompt_id": str(record["prompt_id"]), "prompt_sha256": record["prompt_sha256"], "prompt": prompts[sid],
            "prompt_length": int(record["prompt_length"]), "generation_length": int(record["generation_length"]),
            "generated_token_ids": [int(v) for v in record["generated_token_ids"]],
            "stopped_on_eos": bool(record["stopped_on_eos"]), "trace_event_start": int(view.start),
            "trace_requested_expert_ids": trace.requested_expert_ids[view].astype(int).tolist(),
        })
    order = interleaved_order(sequences, list(config["domains"]))
    workload = {"name": PATH_NAME, "regime": "interleaved",
                "sequences": [{"source_sequence_id": s["sequence_id"], "position": i, "segment_index": 0,
                               "segment_label": "interleaved", "domain": s["domain"]} for i, s in enumerate(order)]}
    spec = {
        "schema_version": "hardware_replay_spec_v1",
        "model": meta["model"],
        "model_revision": meta.get("resolved_model_revision") or meta.get("requested_model_revision"),
        "trace_hash": str(trace.trace_hash),
        "trace_generation": {
            "device_description": meta.get("runtime", {}).get("device_description"),
            "precision": meta.get("precision"),
            "attn_implementation": config.get("attn_implementation", "eager"),
            "max_prompt_tokens": int(config["max_prompt_tokens"]), "max_new_tokens": int(config["max_new_tokens"]),
            "stop_on_eos": bool(config["stop_on_eos"]), "seed": int(config["seed"]),
            "prompt_offset": int(config["prompt_offset"]), "offloaded_generation": config.get("offloaded_generation"),
        },
        "architecture": {"num_moe_layers": len(layers), "layer_indices": layers, "num_experts": int(trace.num_experts),
                         "top_k": int(trace.top_k), "expert_bytes": int(np.asarray(meta["expert_bytes_by_layer"])[0][0])},
        "calibration_sequence_ids": [s["sequence_id"] for s in order],
        "prompt_overlap_with_original": overlap,
        "sequences": sequences,
        "workloads": [workload],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec) + "\n")
    print(f"wrote {output}: {len(sequences)} sequences, trace {trace.trace_hash[:12]}..., overlap {overlap}", flush=True)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--opt-root", type=Path, required=True)
    g.add_argument("--base-config", type=Path, required=True)
    g.add_argument("--output-dir", type=Path, required=True)
    g.add_argument("--offset", type=int, default=PROMPT_OFFSET)
    g.add_argument("--prompts-per-domain", type=int, required=True)
    g.add_argument("--capacity", type=int, default=0, help="decode slots per layer while generating (speed only)")
    g.add_argument("--resident", action="store_true", help="every expert on the GPU (the model fits): no offloading")
    s = sub.add_parser("spec")
    s.add_argument("--opt-root", type=Path, required=True)
    s.add_argument("--trace-root", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--original-metadata", type=Path, nargs="+",
                   help="metadata of every earlier trace of the model, to prove the prompts are disjoint")
    args = parser.parse_args()
    if args.command == "generate":
        if not args.resident and args.capacity < 1:
            raise SystemExit("--capacity is required unless --resident")
        generate(args.opt_root, args.base_config, args.output_dir, args.offset, args.prompts_per_domain, args.capacity,
                 args.resident)
    else:
        build_spec(args.opt_root, args.trace_root, args.output, args.original_metadata)


if __name__ == "__main__":
    main()
