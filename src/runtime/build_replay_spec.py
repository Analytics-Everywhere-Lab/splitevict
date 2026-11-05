"""Write everything the GPU runtime needs to replay the frozen evaluation, as one JSON.

The routing trace records what each sequence generated but not its prompt text; the
prompts come from four pinned dataset revisions through the generator's own loader.
This script re-loads them, checks every prompt's sha256 against the trace metadata, and
writes ``data/specs/<model>_replay_spec.json`` holding, per sequence, the prompt and
the recorded generated token ids, and per workload path the sequence order the
simulator evaluated. With the spec, the GPU host needs only the model weights and this
file, not the measurement code or the datasets.

    python src/runtime/build_replay_spec.py --repository-root measurement \
        --output data/specs/olmoe_replay_spec.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, help="cross-model prepared inputs")
    parser.add_argument("--trace-config", type=Path, help="generation config (defaults to the OLMoE one)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-trace-root", type=Path, help="fresh unseen trace (fresh_trace.py generate)")
    parser.add_argument("--fresh-frozen-dir", type=Path, help="its frozen workloads (fresh_trace.py prepare)")
    args = parser.parse_args(argv)

    repository = args.repository_root.expanduser().resolve()
    if str(CODE) not in sys.path:
        sys.path.append(str(CODE))
    if args.fresh_trace_root is None:
        from run_caer import configure_repository

        configure_repository(repository)
        sys.path.insert(0, str(repository / "src"))
        from expert_analysis.datasets import load_domain_examples
    else:
        # an unseen trace needs only the measurement source code, not its recorded trace data
        sys.path.insert(0, str(HERE))
        from fresh_trace import _paths

        _paths(repository)

    fresh_prompts = None
    if args.fresh_trace_root is not None:
        sys.path.insert(0, str(HERE))
        from fresh_trace import load_fresh

        inputs = load_fresh(repository, args.fresh_trace_root, args.fresh_frozen_dir)
        config_path = args.fresh_trace_root / "fresh_trace_config.json"
        fresh_prompts = {int(k): v for k, v in json.loads((args.fresh_trace_root / "fresh_prompts.json").read_text()).items()}
    elif args.prepared_root is None:
        from harness import load

        inputs, _ = load()
        config_path = args.trace_config or repository / "stage3_residency/configs/full_preregistered.json"
    else:
        import crossmodel

        inputs, _ = crossmodel.load(args.prepared_root.resolve())
        if args.trace_config is None:
            raise SystemExit("--trace-config is required with --prepared-root")
        config_path = args.trace_config
    config = json.loads(Path(config_path).read_text())
    trace = inputs.trace
    meta = trace.metadata

    records = {int(s["sequence_id"]): s for s in meta["sequences"]}
    prompts: dict[int, dict] = {}
    if fresh_prompts is not None:
        for sid, record in records.items():
            text = fresh_prompts[sid]
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != record["prompt_sha256"]:
                raise SystemExit(f"fresh prompt hash mismatch for sequence {sid}")
            prompts[sid] = {"text": text}
        print(f"{len(prompts)} fresh prompts verified against the trace", flush=True)
    for domain in ([] if fresh_prompts is not None else config["domains"]):
        examples = load_domain_examples(
            domain=domain,
            num_examples=int(config["num_prompts_per_domain"]),
            seed=int(config["seed"]),
            revision=config["dataset_revisions"][domain],
            include_answers=bool(config["include_reference_answers"]),
            allow_substitution=bool(config["allow_dataset_substitution"]),
            format_style="legacy",
        )
        domain_records = [s for s in meta["sequences"] if s["domain"] == domain]
        if len(domain_records) != len(examples.texts):
            raise SystemExit(f"{domain}: {len(examples.texts)} prompts for {len(domain_records)} sequences")
        for record, text, prompt_id in zip(domain_records, examples.texts, examples.metadata["selected_example_ids"]):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest != record["prompt_sha256"]:
                raise SystemExit(f"prompt hash mismatch for sequence {record['sequence_id']}")
            if str(prompt_id) != str(record["prompt_id"]):
                raise SystemExit(f"prompt id mismatch for sequence {record['sequence_id']}")
            prompts[int(record["sequence_id"])] = {"text": text}
        print(f"{domain}: {len(domain_records)} prompts verified against the trace", flush=True)

    # per-sequence routing, so the runtime can check its live routing against the trace
    slices = trace.sequence_slices()
    layers = [int(v) for v in meta["layer_indices"]]
    sequences = []
    for sequence_id, record in sorted(records.items()):
        view = slices[sequence_id]
        requested = trace.requested_expert_ids[view].astype(int).tolist()
        sequences.append(
            {
                "sequence_id": sequence_id,
                "domain": record["domain"],
                "prompt_index": int(record["prompt_index"]),
                "prompt_id": str(record["prompt_id"]),
                "prompt_sha256": record["prompt_sha256"],
                "prompt": prompts[sequence_id]["text"],
                "prompt_length": int(record["prompt_length"]),
                "generation_length": int(record["generation_length"]),
                "generated_token_ids": [int(v) for v in record["generated_token_ids"]],
                "stopped_on_eos": bool(record["stopped_on_eos"]),
                "trace_event_start": int(view.start),
                "trace_requested_expert_ids": requested,
            }
        )
    workloads = [
        {
            "name": w.name,
            "regime": w.regime,
            "sequences": [
                {"source_sequence_id": int(s.source_sequence_id), "position": int(s.position),
                 "segment_index": int(s.segment_index), "segment_label": str(s.segment_label),
                 "domain": str(s.domain)}
                for s in w.sequences
            ],
        }
        for w in inputs.workloads
    ]
    calibration = [int(s.source_sequence_id) for s in inputs.calibration.sequences]
    spec = {
        "schema_version": "hardware_replay_spec_v1",
        "model": meta["model"],
        "model_revision": meta.get("resolved_model_revision") or meta.get("requested_model_revision"),
        "trace_hash": str(trace.trace_hash),
        "trace_generation": {
            "device_description": meta.get("runtime", {}).get("device_description"),
            "precision": meta.get("precision"),
            "attn_implementation": config.get("attn_implementation", "eager"),
            "max_prompt_tokens": int(config["max_prompt_tokens"]),
            "max_new_tokens": int(config["max_new_tokens"]),
            "stop_on_eos": bool(config["stop_on_eos"]),
            "seed": int(config["seed"]),
        },
        "architecture": {
            "num_moe_layers": len(layers),
            "layer_indices": layers,
            "num_experts": int(trace.num_experts),
            "top_k": int(trace.top_k),
            "expert_bytes": int(np.asarray(meta["expert_bytes_by_layer"])[0][0]),
        },
        "calibration_sequence_ids": calibration,
        "sequences": sequences,
        "workloads": workloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB, sha256 {digest[:12]}...)")


if __name__ == "__main__":
    main()
