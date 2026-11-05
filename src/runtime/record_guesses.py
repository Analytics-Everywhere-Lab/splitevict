"""Record next-layer expert guesses along a replay spec: the input of a prefetcher.

A prefetcher must name the experts of layer l+1 before layer l+1's router fires. The guess used
here is the one of Mixtral-offloading (Eliseev and Mazur, 2023): apply layer l+1's router to the
hidden state that enters layer l's MoE block. It is available at layer l's router, so the runtime
computes it in the same device-to-host sync as layer l's own routing.

This module decodes the spec's sequences teacher-forced with every expert resident, captures each
MoE block's input during decode, and records per (token, layer) the router's own top-k (checked
against the trace) and the top-``--topm`` guesses for layer l+1 under two variants:

* ``g1``   layer l+1's router on layer l's MoE input (what the runtime computes);
* ``g1n``  the same, with layer l's post-attention RMSNorm gain swapped for layer l+1's
           (folded into the router weight: every column scaled by w_{l+1} / w_l).

A third guess, layer l+1's own request at the previous token, needs no compute and is derived
from the recorded requests by the analysis.

Rows for layer 0 hold -1 (no earlier layer in the same token). The guesses depend only on the
hidden states, which are the same under every eviction policy (routing is exact), so one record per
trace serves the simulator and the exactness check of every hardware run.

    python src/runtime/record_guesses.py --spec data/specs/olmoe_replay_spec.json \
        --sequences calibration --device cuda --output data/guesses/olmoe_calibration.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

VARIANTS = ("g1", "g1n")


def sequence_ids(spec: dict, which: str) -> list[int]:
    if which == "calibration":
        return [int(v) for v in spec["calibration_sequence_ids"]]
    if which == "all":
        return [int(s["sequence_id"]) for s in spec["sequences"]]
    by_name = {w["name"]: w for w in spec["workloads"]}
    if which in by_name:
        return [int(s["source_sequence_id"]) for s in by_name[which]["sequences"]]
    return [int(v) for v in which.split(",")]


class Capture:
    """Wraps every MoE block's forward to keep its decode input (one token)."""

    def __init__(self, model, layer_indices: list[int]):
        self.blocks = [model.model.layers[i].mlp for i in layer_indices]
        self.norms = [model.model.layers[i].post_attention_layernorm.weight for i in layer_indices]
        self.active = False
        self.inputs: list = [None] * len(self.blocks)
        self.originals = []
        for ordinal, block in enumerate(self.blocks):
            original = block.forward
            self.originals.append(original)

            def forward(hidden_states, _ordinal=ordinal, _original=original):
                if self.active:
                    self.inputs[_ordinal] = hidden_states.reshape(-1, hidden_states.shape[-1])[-1:].detach()
                return _original(hidden_states)

            block.forward = forward

    def release(self) -> None:
        for block, original in zip(self.blocks, self.originals):
            block.forward = original


def guess_topm(torch, block, hidden, topm: int):
    """The router's ranking of ``block``'s experts for ``hidden``: the top m of its logits, stable on ties
    (the runtime's guess, ``offload_runtime.OffloadedMoeLayer._read_guess``)."""

    logits = torch.nn.functional.linear(hidden, block.gate.weight)[0].detach().float().cpu().numpy()
    return torch.from_numpy(np.argsort(-logits, kind="stable")[:topm].astype(np.int64))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--sequences", default="calibration")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topm", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    from offload_runtime import load_model

    spec = json.loads(args.spec.read_text())
    arch = spec["architecture"]
    L, E, k = int(arch["num_moe_layers"]), int(arch["num_experts"]), int(arch["top_k"])
    layer_indices = [int(v) for v in arch["layer_indices"]]
    device = torch.device(args.device)
    torch.manual_seed(int(spec["trace_generation"]["seed"]))
    attn = spec["trace_generation"].get("attn_implementation", "eager")
    model = load_model(spec, device, torch.bfloat16, attn, None, False, log=lambda m: print(m, flush=True))
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["model_revision"])
    capture = Capture(model, layer_indices)
    by_id = {int(s["sequence_id"]): s for s in spec["sequences"]}
    ids = sequence_ids(spec, args.sequences)
    if args.limit:
        ids = ids[: args.limit]
    gen = spec["trace_generation"]
    out: dict[str, np.ndarray] = {}
    match_events = match_hits = 0
    start = time.perf_counter()
    for count, sid in enumerate(ids):
        record = by_id[sid]
        recorded = [int(v) for v in record["generated_token_ids"]]
        T = len(recorded)
        trace_req = np.asarray(record["trace_requested_expert_ids"], dtype=np.int64).reshape(T, L, -1)
        encoded = tokenizer(record["prompt"], truncation=True, max_length=int(gen["max_prompt_tokens"]),
                            add_special_tokens=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
        if int(mask.sum().item()) != int(record["prompt_length"]):
            raise RuntimeError(f"prompt length differs from the trace for sequence {sid}")
        request = np.full((T, L, k), -1, dtype=np.int16)
        guesses = {v: np.full((T, L, args.topm), -1, dtype=np.int16) for v in VARIANTS}
        with torch.inference_mode():
            capture.active = False
            outputs = model(input_ids=input_ids, attention_mask=mask, use_cache=True, return_dict=True)
            past = outputs.past_key_values
            for step in range(T):
                mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype, device=device)), dim=1)
                capture.active = True
                outputs = model(input_ids=torch.tensor([[recorded[step]]], device=device), attention_mask=mask,
                                past_key_values=past, use_cache=True, return_dict=True)
                capture.active = False
                past = outputs.past_key_values
                rows = []
                for l in range(L):
                    h = capture.inputs[l]
                    _logits, _w, idx = capture.blocks[l].gate(h)
                    rows.append(idx[0])
                    if l + 1 < L:
                        nxt = capture.blocks[l + 1]
                        g1 = guess_topm(torch, nxt, h, args.topm)
                        # g1n as the runtime computes it: layer l+1's router weight with every column scaled by
                        # w_{l+1} / w_l (offload_runtime.OffloadRuntime.guess_weight)
                        ratio = capture.norms[l + 1].float() / capture.norms[l].float()
                        scaled = (nxt.gate.weight.float() * ratio[None, :]).to(nxt.gate.weight.dtype)
                        logits = torch.nn.functional.linear(h, scaled)[0].detach().float().cpu().numpy()
                        g1n = torch.from_numpy(np.argsort(-logits, kind="stable")[: args.topm].astype(np.int64))
                        rows.append(g1)
                        rows.append(g1n)
                packed = torch.cat([r.to(torch.int64) for r in rows]).cpu().numpy()
                at = 0
                for l in range(L):
                    request[step, l] = packed[at:at + k]
                    at += k
                    if l + 1 < L:
                        guesses["g1"][step, l + 1] = packed[at:at + args.topm]
                        at += args.topm
                        guesses["g1n"][step, l + 1] = packed[at:at + args.topm]
                        at += args.topm
                for l in range(L):
                    match_events += 1
                    match_hits += int(set(request[step, l].tolist()) == set(trace_req[step, l].tolist()))
        out[f"request/{sid}"] = request
        for v in VARIANTS:
            out[f"{v}/{sid}"] = guesses[v]
        if (count + 1) % 10 == 0 or count == 0:
            rate = (time.perf_counter() - start) / (count + 1)
            print(f"seq {count + 1}/{len(ids)} (id {sid}): routing match so far {100 * match_hits / match_events:.2f}% "
                  f"[{rate:.1f} s/seq]", flush=True)
    capture.release()
    meta = {"spec": str(args.spec), "trace_hash": spec["trace_hash"], "model": spec["model"],
            "sequences": ids, "topm": args.topm, "variants": list(VARIANTS), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "torch": torch.__version__, "routing_events": match_events, "routing_matches": match_hits}
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, meta=np.array(json.dumps(meta)), **out)
    print(f"wrote {args.output}: {len(ids)} sequences, routing match {100 * match_hits / max(match_events, 1):.2f}%",
          flush=True)


if __name__ == "__main__":
    main()
