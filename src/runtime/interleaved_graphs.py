"""Runner for T9: the decode step replayed from CUDA graphs, for a shorter policy-independent time per token (W0).

The eager runtime launches every kernel of a decode step from Python, so at batch size one most of a token's time
independent of the eviction policy goes to launching kernels. Here a decode step is cut at the points where the CPU must
read a router's choice, one per MoE layer, and each piece between two such points is captured once as a CUDA graph and
replayed: L + 1 graph launches per token. Piece 0 embeds the token, computes the rotary embedding and the attention
mask, and runs layer 0 up to its router. Piece l runs layer l-1's experts module from the slots, adds its output to the
residual and runs layer l up to its router. The last piece does the same for the last layer, then the final norm, the
head and the argmax. Between two pieces the CPU reads the router's choice, the eviction policy decides, the missing
experts are copied into their slots and the slot indices to the GPU, exactly as in ``OffloadedMoeLayer.forward``, with
the same counters and timing events. The experts module is in the pieces only if its grouped GEMM never waits on the
host (checked once at start-up with torch's synchronization debug mode): GPUs without a fused grouped kernel read the
group offsets on the host, which a graph cannot capture, and there the experts module runs eagerly between the pieces,
as in the eager runtime. The output records which (``experts_in_graphs``).

Attention reads a key/value cache of fixed size (one per layer, the longest sequence of the spec plus one), filled from
the eager prefill's cache at the start of each sequence and written at the token's position inside the graph; positions
after the token are masked. The arithmetic is the eager attention of transformers over a longer, masked key range, so
results can differ from the eager runtime in the last bits; every arm of a run uses this same path. On a CPU (tests)
the pieces run eagerly, without graphs.

Decoding is free-running for the recorded number of steps, as in ``interleaved_freerun.py``. Prefill stays the eager,
untimed forward. Each row also lists the tokens fed to the model (``generated_tokens``), so the analysis can check that
every arm decoded the same token stream. Supported: OLMoE with experts resident on the GPU, the experts module
computing from the slots, no prefetching.

    python src/runtime/interleaved_graphs.py <interleaved_h100.py arguments>
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import interleaved_freerun  # noqa: E402
import interleaved_h100  # noqa: E402
from offload_runtime import PREFETCH_COUNTERS, OffloadRuntime  # noqa: E402


class GraphedDecoder:
    """The decode step of an OLMoE model as L + 1 pieces, replayed from CUDA graphs on a GPU."""

    def __init__(self, runtime: OffloadRuntime, max_len: int) -> None:
        import torch
        from transformers.models.olmoe import modeling_olmoe

        model = runtime.model
        if model.config.model_type != "olmoe":
            raise RuntimeError("the graphed decode step is written for OLMoE")
        if runtime.compute != "module" or runtime.experts_on_host:
            raise RuntimeError("the graphed decode step needs resident non-expert weights and the experts module")
        self.torch = torch
        self.rope = modeling_olmoe.apply_rotary_pos_emb
        self.repeat_kv = modeling_olmoe.repeat_kv
        self.runtime = runtime
        self.model = model
        self.device = runtime.device
        self.graphed = runtime.cuda
        indices = [int(v) for v in runtime.spec["architecture"]["layer_indices"]]
        if indices != list(range(model.config.num_hidden_layers)):
            raise RuntimeError("the graphed decode step expects every decoder layer to be a MoE layer")
        self.decoder = [model.model.layers[i] for i in indices]
        self.moe = runtime.layers
        self.L = len(self.decoder)
        config = model.config
        attn = self.decoder[0].self_attn
        self.head_dim = attn.head_dim
        self.kv_heads = config.num_key_value_heads
        self.groups = attn.num_key_value_groups
        self.hidden = config.hidden_size
        self.top_k = int(runtime.spec["architecture"]["top_k"])
        self.max_len = int(max_len)
        dtype = next(model.parameters()).dtype
        self.dtype = dtype
        device = self.device

        def zeros(*shape, dtype=dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        self.tok = zeros(1, 1, dtype=torch.long)
        self.pos = zeros(1, dtype=torch.long)
        self.positions = torch.arange(self.max_len, device=device)
        self.mask = zeros(1, 1, 1, self.max_len)
        self.mask_min = torch.finfo(dtype).min
        self.cos = zeros(1, 1, self.head_dim)
        self.sin = zeros(1, 1, self.head_dim)
        self.keys = [zeros(1, self.kv_heads, self.max_len, self.head_dim) for _ in range(self.L)]
        self.values = [zeros(1, self.kv_heads, self.max_len, self.head_dim) for _ in range(self.L)]
        self.residual = [zeros(1, 1, self.hidden) for _ in range(self.L)]
        self.flat = [zeros(1, self.hidden) for _ in range(self.L)]
        self.weights = [zeros(1, self.top_k) for _ in range(self.L)]
        self.indices = [zeros(1, self.top_k, dtype=torch.long) for _ in range(self.L)]
        self.slot_index = [zeros(1, self.top_k, dtype=torch.long) for _ in range(self.L)]
        self.expert_out = [zeros(1, self.hidden) for _ in range(self.L)]
        self.next_token = zeros(1, dtype=torch.long)
        pin = runtime.cuda
        self.tok_host = torch.zeros((1, 1), dtype=torch.long)
        self.pos_host = torch.zeros(1, dtype=torch.long)
        if pin:
            self.tok_host, self.pos_host = self.tok_host.pin_memory(), self.pos_host.pin_memory()
        self.graphs: list = []
        self.capture_seconds = 0.0
        self.experts_in_graphs = self.graphed and self._experts_capturable()
        if self.graphed:
            self.capture()

    def _experts_capturable(self) -> bool:
        """Whether the experts module runs without waiting on the host (a graph can then hold it)."""

        torch = self.torch
        with torch.inference_mode():
            self.experts(0)  # first call: lazy initialization may synchronize once
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                self.experts(0)
                return True
            except RuntimeError:
                return False
            finally:
                torch.cuda.set_sync_debug_mode(0)
                torch.cuda.synchronize()

    # ------------------------------------------------------------------ pieces (captured)
    def _attention(self, ordinal: int, layer, x):
        torch = self.torch
        attn = layer.self_attn
        q = attn.q_norm(attn.q_proj(x))
        k = attn.k_norm(attn.k_proj(x))
        v = attn.v_proj(x)
        clip = attn.config.clip_qkv
        if clip is not None:
            q.clamp_(min=-clip, max=clip)
            k.clamp_(min=-clip, max=clip)
            v.clamp_(min=-clip, max=clip)
        q = q.view(1, 1, -1, self.head_dim).transpose(1, 2)
        k = k.view(1, 1, -1, self.head_dim).transpose(1, 2)
        v = v.view(1, 1, -1, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, self.cos, self.sin)
        self.keys[ordinal].index_copy_(2, self.pos, k)
        self.values[ordinal].index_copy_(2, self.pos, v)
        keys = self.repeat_kv(self.keys[ordinal], self.groups)
        values = self.repeat_kv(self.values[ordinal], self.groups)
        scores = torch.matmul(q, keys.transpose(2, 3)) * attn.scaling + self.mask
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, values).transpose(1, 2).contiguous().reshape(1, 1, -1)
        return attn.o_proj(out)

    def _to_router(self, ordinal: int, hidden) -> None:
        layer = self.decoder[ordinal]
        hidden = hidden + self._attention(ordinal, layer, layer.input_layernorm(hidden))
        self.residual[ordinal].copy_(hidden)
        flat = layer.post_attention_layernorm(hidden).view(-1, self.hidden)
        self.flat[ordinal].copy_(flat)
        _logits, weights, indices = layer.mlp.gate(flat)
        self.weights[ordinal].copy_(weights)
        self.indices[ordinal].copy_(indices)

    def layer_output(self, ordinal: int):
        """Layer ``ordinal``'s output: its residual plus its expert output."""

        return self.residual[ordinal] + self.expert_out[ordinal].view(1, 1, self.hidden)

    def piece(self, index: int) -> None:
        torch = self.torch
        model = self.model
        if index == 0:
            hidden = model.model.embed_tokens(self.tok)
            cos, sin = model.model.rotary_emb(hidden, self.pos.view(1, 1))
            self.cos.copy_(cos)
            self.sin.copy_(sin)
            visible = (self.positions <= self.pos).view(1, 1, 1, -1)
            self.mask.copy_(torch.where(visible, 0.0, self.mask_min).to(self.dtype))
            self._to_router(0, hidden)
        else:
            if self.experts_in_graphs:
                self.experts(index - 1)
            if index < self.L:
                self._to_router(index, self.layer_output(index - 1))
            else:
                logits = model.lm_head(model.model.norm(self.layer_output(self.L - 1)))
                self.next_token.copy_(torch.argmax(logits[:, -1, :].float(), dim=-1))

    def experts(self, ordinal: int) -> None:
        """The experts module from the slots, eagerly, into the layer's expert-output buffer."""

        out = self.moe[ordinal].slot_experts(self.flat[ordinal], self.slot_index[ordinal], self.weights[ordinal])
        self.expert_out[ordinal].copy_(out.view(1, self.hidden))

    def capture(self) -> None:
        torch = self.torch
        start = time.perf_counter()
        with torch.inference_mode():
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    for index in range(self.L + 1):
                        self.piece(index)
                        if index < self.L and not self.experts_in_graphs:
                            self.experts(index)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            pool = torch.cuda.graph_pool_handle()
            for index in range(self.L + 1):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self.piece(index)
                self.graphs.append(graph)
            torch.cuda.synchronize()
        self.capture_seconds = time.perf_counter() - start

    # ------------------------------------------------------------------ between pieces (CPU)
    def begin_sequence(self, past, prompt_length: int) -> None:
        """The prompt's keys and values from the eager prefill's cache, at positions 0 .. prompt_length - 1."""

        if prompt_length + 1 > self.max_len:
            raise RuntimeError(f"a prompt of {prompt_length} tokens does not fit the {self.max_len}-position cache")
        for ordinal in range(self.L):
            layer = past.layers[ordinal]
            self.keys[ordinal][:, :, :prompt_length].copy_(layer.keys)
            self.values[ordinal][:, :, :prompt_length].copy_(layer.values)

    def _between(self, ordinal: int) -> None:
        """``OffloadedMoeLayer.forward`` between the router and the experts: routing read, decision, copies."""

        runtime = self.runtime
        torch = self.torch
        layer = self.moe[ordinal]
        entered = time.perf_counter_ns()
        sync_start = time.perf_counter_ns()
        request = self.indices[ordinal][0].to("cpu", non_blocking=False).numpy().astype(np.int64)
        gates = self.weights[ordinal][0].float().to("cpu").numpy().astype(np.float64)
        runtime.token_sync_ns += time.perf_counter_ns() - sync_start  # the CPU waiting for the GPU
        runtime.observe_routing(ordinal, request, gates)
        t0 = time.perf_counter_ns()
        result = runtime.policy.step(ordinal, runtime.current_token, request, gates)
        t1 = time.perf_counter_ns()
        runtime.token_policy_ns += t1 - t0
        for expert in result.evicted:
            slot = int(layer.slot_of[expert])
            if slot < 0:
                raise RuntimeError("evicting an expert that has no slot")
            layer.slot_of[expert] = -1
            layer.free_slots.append(slot)
        if result.loaded.size:
            start_event = runtime.event(timing=True) if runtime.cuda else None
            end_event = runtime.event(timing=True) if runtime.cuda else None
            if start_event is not None:
                start_event.record()
            for expert in result.loaded:
                if not layer.free_slots:
                    raise RuntimeError("no free slot for an admitted expert")
                slot = layer.free_slots.pop()
                layer.slot_of[expert] = slot
                layer.copy_expert(int(expert), slot, None)
            if end_event is not None:
                end_event.record()
                runtime.token_copy_events.append((start_event, end_event))
            runtime.token_transfers += int(result.loaded.size)
            runtime.token_bytes += int(result.loaded.size) * runtime.expert_bytes
        runtime.token_issue_ns += time.perf_counter_ns() - t1  # slot bookkeeping + copy issue
        slots = layer.slot_of[request]
        if (slots < 0).any():
            raise RuntimeError("a requested expert is not resident after admission")
        layer.pinned_index[0].copy_(torch.from_numpy(slots))
        self.slot_index[ordinal].copy_(layer.pinned_index, non_blocking=True)
        runtime.token_moe_ns += time.perf_counter_ns() - entered

    def step(self, token_id: int, position: int):
        """One decode step: the token at ``position``; returns the argmax token (on the device)."""

        if position >= self.max_len:
            raise RuntimeError(f"position {position} is outside the {self.max_len}-position cache")
        self.tok_host[0, 0] = int(token_id)
        self.pos_host[0] = int(position)
        self.tok.copy_(self.tok_host, non_blocking=True)
        self.pos.copy_(self.pos_host, non_blocking=True)
        for index in range(self.L + 1):
            if self.graphed:
                self.graphs[index].replay()
            else:
                self.piece(index)
            if index < self.L:
                self._between(index)
                if not self.experts_in_graphs:
                    self.experts(index)
        return self.next_token


def max_positions(spec: dict[str, Any]) -> int:
    """The longest sequence of the spec (prompt plus generated tokens) plus one."""

    return max(int(s["prompt_length"]) + len(s["generated_token_ids"]) for s in spec["sequences"]) + 1


def graphed_run_path(runtime, tokenizer, spec: dict[str, Any], workload: dict[str, Any], *, teacher_force: bool,
                     limit_sequences: int | None, log) -> dict[str, Any]:
    """``offload_runtime.run_path`` free-running for the recorded number of steps, every decode step graphed."""

    import torch

    model = runtime.model
    device = runtime.device
    decoder: GraphedDecoder = runtime.graphed_decoder
    sequences_by_id = {int(s["sequence_id"]): s for s in spec["sequences"]}
    generation = spec["trace_generation"]
    layers = len(spec["architecture"]["layer_indices"])
    rows = []
    for count, item in enumerate(workload["sequences"]):
        if limit_sequences is not None and count >= limit_sequences:
            break
        record = sequences_by_id[int(item["source_sequence_id"])]
        runtime.policy.begin_sequence()
        encoded = tokenizer(record["prompt"], truncation=True, max_length=int(generation["max_prompt_tokens"]),
                            add_special_tokens=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
        prompt_length = int(attention_mask.sum().item())
        if prompt_length != int(record["prompt_length"]):
            raise RuntimeError(f"prompt length differs from the trace for sequence {record['sequence_id']}")
        recorded = [int(v) for v in record["generated_token_ids"]]
        trace_requests = np.asarray(record["trace_requested_expert_ids"], dtype=np.int64).reshape(len(recorded), layers, -1)

        runtime.mode = "prefill"
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            if runtime.cuda:
                torch.cuda.synchronize()
            predicted_first = int(torch.argmax(outputs.logits[:, -1, :].float(), dim=-1).item())
            decoder.begin_sequence(outputs.past_key_values, prompt_length)
            del outputs
            runtime.mode = "decode"
            wall_ns = policy_ns = issue_ns = 0
            copy_ms = 0.0
            transfers = nbytes = routing_events = routing_matches = token_agreements = tokens = 0
            sync_ns = moe_ns = 0
            token_wall: list[float] = []
            token_cpu: list[int] = []
            token_mhz: list[int] = []
            monitor = runtime.host_monitor
            busy_start = monitor.busy() if monitor is not None else []
            next_token = predicted_first
            generated: list[int] = []
            transfers_before = runtime.policy.transfers
            pf_before = {key: getattr(runtime.policy, attr, 0) for key, attr in PREFETCH_COUNTERS.items()}
            for step in range(len(recorded)):
                token_id = int(next_token)
                generated.append(token_id)
                runtime.current_token = token_id
                runtime.reset_token_counters()
                runtime.routing = []
                runtime.guesses = []
                t0 = time.perf_counter_ns()
                if hasattr(runtime.policy, "begin_token"):
                    runtime.policy.begin_token(token_id)
                    runtime.token_policy_ns += time.perf_counter_ns() - t0
                argmax = decoder.step(token_id, prompt_length + step)
                if runtime.cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                wall_ns += t1 - t0
                token_wall.append(round((t1 - t0) / 1e6, 3))
                if monitor is not None:
                    token_cpu.append(monitor.cpu())
                    token_mhz.append(monitor.mhz(token_cpu[-1]))
                policy_ns += runtime.token_policy_ns
                issue_ns += runtime.token_issue_ns
                copy_ms += runtime.copy_ms()
                transfers += runtime.token_transfers
                nbytes += runtime.token_bytes
                sync_ns += runtime.token_sync_ns
                moe_ns += runtime.token_moe_ns
                predicted = int(argmax.item())
                tokens += 1
                if len(runtime.routing) != layers:
                    raise RuntimeError(f"captured {len(runtime.routing)} MoE events for {layers} layers")
                for ordinal, request, _gates in runtime.routing:
                    routing_events += 1
                    if set(request.tolist()) == set(trace_requests[step, ordinal].tolist()):
                        routing_matches += 1
                if step + 1 < len(recorded):
                    if predicted == recorded[step + 1]:
                        token_agreements += 1
                    next_token = predicted
        rows.append({
            "sequence_position": int(item["position"]),
            "source_sequence_id": int(item["source_sequence_id"]),
            "domain": str(item["domain"]),
            "prompt_length": int(record["prompt_length"]),
            "tokens": tokens,
            "expert_transfers": int(runtime.policy.transfers - transfers_before),
            "transfers_counted_in_layers": transfers,
            "bytes_transferred": nbytes,
            "wall_ms": wall_ns / 1e6,
            "policy_ms": policy_ns / 1e6,
            "issue_ms": issue_ns / 1e6,
            "copy_ms": copy_ms,
            "routing_events": routing_events,
            "routing_matches": routing_matches,
            "token_agreements": token_agreements,
            "token_comparisons": max(tokens - 1, 0),
            "first_token_agrees": predicted_first == recorded[0],
            "prefetch_transfers": 0, "prefetch_copy_ms": 0.0, "prefetch_stall_ms": 0.0, "prefetch_policy_ms": 0.0,
            "prefetch_issue_ms": 0.0, "sync_ms": sync_ns / 1e6, "moe_ms": moe_ns / 1e6, "prefetch_slot_waits": 0,
            "prefetch_slot_stalls": 0, "prefetch_needed_stalls": 0, "guess_events": 0, "guess_matches": 0,
            "token_wall_ms": token_wall,
            "generated_tokens": generated,
        })
        if monitor is not None:
            busy_end = monitor.busy()
            rows[-1]["token_cpu"] = token_cpu
            rows[-1]["token_mhz"] = token_mhz
            rows[-1]["host_busy_jiffies"] = [b - a for a, b in zip(busy_start, busy_end)]
        for key, attr in PREFETCH_COUNTERS.items():
            rows[-1][key] = int(getattr(runtime.policy, attr, 0) - pf_before.get(key, 0))
        if rows[-1]["expert_transfers"] != transfers:
            raise RuntimeError("policy transfer count and layer-level transfer count disagree")
        if (count + 1) % 10 == 0 or count == 0:
            r = rows[-1]
            log(f"    seq {count + 1}/{len(workload['sequences'])}: {r['tokens']} tok, "
                f"{r['wall_ms'] / max(r['tokens'], 1):.1f} ms/tok wall, {r['expert_transfers']} transfers, "
                f"routing {r['routing_matches']}/{r['routing_events']}")
    return {"workload": workload["name"], "regime": workload["regime"], "sequences": rows}


RUNTIMES: list = []


class GraphedRuntime(OffloadRuntime):
    """``OffloadRuntime`` with a graphed decode step built for the spec's longest sequence."""

    def __init__(self, model, spec, *args, **kwargs) -> None:
        super().__init__(model, spec, *args, **kwargs)
        self.graphed_decoder = GraphedDecoder(self, max_positions(spec))
        RUNTIMES.append(self)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a == "--experts-on-host" for a in argv) or any(a.endswith("+pf") or "+pf:" in a for a in argv):
        raise SystemExit("T9 runs OLMoE with resident non-expert weights and no prefetching")
    output, argv = interleaved_freerun.output_path(argv)
    partial = Path(str(output) + ".partial")
    saved = (interleaved_h100.run_path, interleaved_h100.OffloadRuntime)
    interleaved_h100.run_path, interleaved_h100.OffloadRuntime = graphed_run_path, GraphedRuntime
    try:
        interleaved_h100.main(argv)
    finally:
        interleaved_h100.run_path, interleaved_h100.OffloadRuntime = saved
    decoder = RUNTIMES[-1].graphed_decoder
    out = json.loads(partial.read_text())
    out["decoding"] = "free_running"
    out["runner"] = "interleaved_graphs.py"
    out["graphs"] = {"graphed": decoder.graphed, "experts_in_graphs": decoder.experts_in_graphs, "pieces": decoder.L + 1,
                     "max_positions": decoder.max_len, "capture_seconds": decoder.capture_seconds}
    staging = Path(str(output) + ".tmp")
    staging.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(staging, output)
    partial.unlink()
    print(f"graphed decode: {decoder.L + 1} pieces, experts in graphs {decoder.experts_in_graphs}, {decoder.max_len} "
          f"positions, captured in {decoder.capture_seconds:.1f} s", flush=True)


if __name__ == "__main__":
    main()
