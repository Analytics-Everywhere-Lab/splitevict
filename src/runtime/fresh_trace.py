"""A new OLMoE routing trace on prompts no earlier trace used (T1).

``generate`` runs the original trace's recipe (pinned model and dataset revisions, bf16, eager attention,
greedy decode, 128 new tokens, seed 42) on the next 100 prompts of each domain's pinned shuffle: examples
100-199 instead of 0-99, so the prompt sets are disjoint by construction. ``prepare`` freezes the same ten
workload paths on the new trace with the frozen functions of ``measurement/`` (``freeze_evaluation``).
Nothing in ``measurement/`` is modified: its generator is imported and its prompt loader is wrapped in memory.

    # on the GPU (needs measurement/, torch, transformers 5.15.1, datasets)
    python src/runtime/fresh_trace.py generate --opt-root measurement --output-dir data/traces/olmoe_t1
    # anywhere (CPU)
    python src/runtime/fresh_trace.py prepare --opt-root measurement --trace-root data/traces/olmoe_t1 \
        --out data/prepared/olmoe_t1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROMPT_OFFSET = 100
BASE_CONFIG = "stage3_residency/configs/full_preregistered.json"


def _paths(opt_root: Path) -> None:
    for relative in ("stage3_residency/stage1_prediction/src", "stage3_residency/src", "src"):
        value = str((opt_root / relative).resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def fresh_config(opt_root: Path, offset: int = PROMPT_OFFSET) -> dict:
    config = json.loads((opt_root / BASE_CONFIG).read_text())
    config["stage"] = f"Fresh unseen OLMoE trace: examples {offset}-{offset + int(config['num_prompts_per_domain']) - 1} of each pinned shuffle"
    config["prompt_offset"] = int(offset)
    return config


def generate(opt_root: Path, output_dir: Path, offset: int) -> None:
    _paths(opt_root)
    import residency_headroom.trace_generation as tg

    original = tg.load_domain_examples

    def shifted(*, domain, num_examples, seed, **kwargs):
        examples = original(domain=domain, num_examples=num_examples + offset, seed=seed, **kwargs)
        if len(examples.texts) != num_examples + offset:
            raise RuntimeError(f"{domain}: only {len(examples.texts)} examples for offset {offset}")
        examples.texts = examples.texts[offset:]
        metadata = dict(examples.metadata)
        metadata["selected_example_ids"] = list(metadata["selected_example_ids"])[offset:]
        metadata["prompt_offset"] = int(offset)
        examples.metadata = metadata
        return examples

    tg.load_domain_examples = shifted
    config = fresh_config(opt_root, offset)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fresh_trace_config.json").write_text(json.dumps(config, indent=2) + "\n")
    trace = tg.generate_trace(config, output_dir.resolve())
    validation = trace.validate()
    print(f"fresh trace: {validation['sequences']} sequences, {validation['generated_tokens']} tokens, "
          f"{validation['events']} events, hash={validation['trace_hash']}", flush=True)

    # the prompt texts, for the hardware replay spec (hash-checked against the trace)
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


def prepare(opt_root: Path, trace_root: Path, out: Path, original_trace: Path | None) -> dict:
    _paths(opt_root)
    from residency_headroom.freeze import freeze_evaluation
    from residency_headroom.trace import RoutingTrace

    trace = RoutingTrace.load(trace_root / "routing_trace.npz", validate=True)
    config = json.loads((trace_root / "fresh_trace_config.json").read_text())
    report = {"trace_hash": trace.trace_hash, "sequences": len(trace.metadata["sequences"]),
              "events": int(trace.num_events), "prompt_offset": config.get("prompt_offset")}
    if original_trace is not None:
        old = json.loads(original_trace.read_text())
        old_hashes = {s["prompt_sha256"] for s in old["sequences"]}
        new_hashes = {s["prompt_sha256"] for s in trace.metadata["sequences"]}
        report["prompt_overlap_with_original"] = len(old_hashes & new_hashes)
        if report["prompt_overlap_with_original"]:
            raise SystemExit(f"fresh prompts overlap the original trace: {report['prompt_overlap_with_original']}")
    frozen = freeze_evaluation(trace, config, out)
    (out / "frozen_workloads.json").write_text(json.dumps(
        {"calibration_workload": frozen["calibration_workload"], "workloads": frozen["workloads"],
         "report": report}, indent=1) + "\n")
    print(json.dumps(report), flush=True)
    return frozen


class FreshInputs:
    def __init__(self, trace, calibration, workloads):
        self.trace, self.calibration, self.workloads = trace, calibration, workloads


def load_fresh(opt_root: Path, trace_root: Path, frozen_dir: Path) -> FreshInputs:
    _paths(opt_root)
    from race_stage1.frozen import workload_from_record
    from residency_headroom.trace import RoutingTrace

    trace = RoutingTrace.load(trace_root / "routing_trace.npz", validate=True)
    record = json.loads((frozen_dir / "frozen_workloads.json").read_text())
    if record["report"]["trace_hash"] != trace.trace_hash:
        raise SystemExit("frozen workloads were prepared for a different trace")
    calibration = workload_from_record(record["calibration_workload"])
    workloads = tuple(workload_from_record(item) for item in record["workloads"])
    return FreshInputs(trace, calibration, workloads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--opt-root", type=Path, required=True)
    g.add_argument("--output-dir", type=Path, required=True)
    g.add_argument("--offset", type=int, default=PROMPT_OFFSET)
    p = sub.add_parser("prepare")
    p.add_argument("--opt-root", type=Path, required=True)
    p.add_argument("--trace-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--original-metadata", type=Path,
                   help="original trace metadata JSON, to prove the prompt sets are disjoint")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.opt_root, args.output_dir, args.offset)
    else:
        prepare(args.opt_root, args.trace_root, args.out, args.original_metadata)


if __name__ == "__main__":
    main()
