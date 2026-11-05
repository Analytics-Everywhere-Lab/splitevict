"""Decode a recorded workload path on a GPU with the experts offloaded to host memory.

Every MoE layer keeps ``B`` expert slots on the GPU. At each decode step the router picks its
top-k experts, the eviction policy (``policy.py``, the object the trace replay validated) decides
which residents leave, the missing experts are copied from pinned host memory into the freed
slots, and the experts run from the slots. Per generated token the runtime records the wall time
of the step, the policy's CPU time on the critical path, the GPU time of the host-to-device
copies, and the number of experts copied.

Fidelity to the simulator:

* Prompts, generated tokens and the order of sequences on each path are the recorded ones
  (``build_replay_spec.py``). By default decode is teacher-forced to the recorded tokens, so
  every policy sees the same token stream and the routing is checked event by event against
  the trace. The model's own argmax is recorded too, for token agreement.
* Prefill runs on the full expert set and is not timed; the simulator never saw prefill
  routing either. Cache state carries across the sequences of a path, as in the simulator.
* ``--policy resident`` keeps every expert on the GPU and never copies: the upper bound on
  decode speed with the same expert kernels.
* ``--experts-on-host`` is for models whose experts do not fit the GPU (Qwen3-30B-A3B,
  Mixtral-8x7B). Everything except the expert weights moves to the GPU; the expert weights stay
  pinned in host memory. Prefill streams each layer's full expert set through one staging
  buffer on the GPU and runs the model's own experts module on it, the same computation as the
  resident prefill. ``--policy resident`` is not available in this mode.

    python src/runtime/offload_runtime.py --spec data/specs/olmoe_replay_spec.json \
        --policy-bundle data/rankers/olmoe_setdev_policy.pkl --policy caer_gbt \
        --capacity 32 --output-dir data/runs/a40 --workloads stationary_general
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from policy import LEARNED_VARIANTS, SIMPLE_VARIANTS, CachePolicy, PolicyBundle, make_policy  # noqa: E402

def _sha256(path) -> str | None:
    import hashlib

    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


POLICIES = ("resident",) + SIMPLE_VARIANTS + LEARNED_VARIANTS + ("splitphase",)
PREFETCH_ROW_KEYS = ("sync_ms", "moe_ms", "prefetch_transfers", "prefetch_copy_ms", "prefetch_stall_ms", "prefetch_policy_ms",
                     "prefetch_issue_ms", "prefetch_slot_waits", "prefetch_slot_stalls", "prefetch_needed_stalls",
                     "guess_events", "guess_matches",
                     "prefetch_transfers_policy", "prefetch_hits", "prefetch_unused", "prefetch_evicted_requested")
# per-sequence row key -> policy counter (policy.CachePolicy)
PREFETCH_COUNTERS = {"prefetch_transfers_policy": "prefetch_transfers", "prefetch_hits": "prefetch_hits",
                     "prefetch_unused": "prefetch_unused", "prefetch_evicted_requested": "prefetch_evicted_requested"}


@dataclass(frozen=True)
class PrefetchConfig:
    """Next-layer prefetching (test T4).

    At layer l's router, layer l+1's router is applied to layer l's MoE input (``g1``; ``g1n`` first
    swaps layer l's post-attention RMSNorm gain for layer l+1's) and its top ``m`` experts are the
    guess; up to ``q`` of the non-resident ones are copied on a second CUDA stream, which starts after
    layer l's own demand copies and overlaps layer l's expert compute and layer l+1's attention.
    Layer l+1's expert compute waits only for the prefetched experts its request uses.
    """

    m: int = 8
    q: int = 2
    variant: str = "g1"

    def __post_init__(self) -> None:
        if self.variant not in ("g1", "g1n") or self.m < 1 or self.q < 0:
            raise ValueError(f"bad prefetch configuration {self}")


class DriverCopier:
    """Host-to-device expert copies through the CUDA driver API (``cuMemcpyHtoDAsync``) on a torch stream:
    the same bytes as two ``Tensor.copy_(non_blocking=True)`` calls, at a fraction of their CPU cost."""

    def __init__(self) -> None:
        import ctypes

        self.ctypes = ctypes
        lib = ctypes.CDLL("libcuda.so.1")
        self.memcpy = lib.cuMemcpyHtoDAsync_v2
        self.memcpy.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        self.memcpy.restype = ctypes.c_int

    def copy(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        rc = self.memcpy(dst, self.ctypes.c_void_p(src), nbytes, self.ctypes.c_void_p(stream))
        if rc:
            raise RuntimeError(f"cuMemcpyHtoDAsync failed with CUresult {rc}")


class OffloadedMoeLayer:
    """Replaces one sparse-MoE block's forward with the slot-based offloaded version."""

    def __init__(self, block, ordinal: int, runtime: "OffloadRuntime") -> None:
        import torch

        self.torch = torch
        self.block = block
        self.ordinal = ordinal
        self.runtime = runtime
        experts = block.experts
        self.resident_gate_up = experts.gate_up_proj  # (E, 2I, H) on device
        self.resident_down = experts.down_proj  # (E, H, I) on device
        self.E = int(self.resident_gate_up.shape[0])
        self.original_forward = block.forward
        self.host_gate_up = None
        self.host_down = None
        self.slot_gate_up = None
        self.slot_down = None
        self.slot_of = np.full(self.E, -1, dtype=np.int64)
        self.free_slots: list[int] = []
        self.slot_experts = None
        self.device = None
        self.on_host = False  # expert weights live (pinned) in host memory, not on the device
        # prefetching, reset every token: slot -> event of a prefetch copy into it (guards later
        # overwrites), expert -> (issue order, event) for experts prefetched for this layer's next event
        self.pending: dict[int, Any] = {}
        self.inflight: dict[int, tuple[int, Any]] = {}
        block.forward = self.forward

    def allocate(self, capacity: int, device, experts_on_host: bool = False) -> None:
        """Host copies, device slots, and an experts module that computes from the slots.

        The slot module is the same transformers class as the resident one (so it runs
        the same experts implementation and kernels), with its parameters pointing at
        the slot buffers and its expert count set to the capacity; the request is passed
        as slot indices in top-k order.
        """

        import copy

        torch = self.torch
        self.device = device
        pin = device.type == "cuda"
        if experts_on_host:
            self._pin_in_place(pin)
        else:
            self.host_gate_up = self.resident_gate_up.detach().to("cpu")
            self.host_down = self.resident_down.detach().to("cpu")
            if pin:
                self.host_gate_up = self.host_gate_up.pin_memory()
                self.host_down = self.host_down.pin_memory()
        shape_gu = (capacity,) + tuple(self.resident_gate_up.shape[1:])
        shape_d = (capacity,) + tuple(self.resident_down.shape[1:])
        self.slot_gate_up = torch.empty(shape_gu, dtype=self.resident_gate_up.dtype, device=device)
        self.slot_down = torch.empty(shape_d, dtype=self.resident_down.dtype, device=device)
        self.slot_of.fill(-1)
        self.free_slots = list(range(capacity - 1, -1, -1))
        experts = self.block.experts
        config = copy.deepcopy(experts.config)
        for name in ("num_local_experts", "num_experts"):
            if hasattr(config, name):
                setattr(config, name, int(capacity))
        with torch.device("meta"):
            slot_experts = type(experts)(config)
        slot_experts.gate_up_proj = torch.nn.Parameter(self.slot_gate_up, requires_grad=False)
        slot_experts.down_proj = torch.nn.Parameter(self.slot_down, requires_grad=False)
        slot_experts.num_experts = int(capacity)
        self.slot_experts = slot_experts.eval()
        self._pointers()
        # staging for the slot indices: pinned, copied without a host sync, so the CPU is
        # not made to wait for the expert copies queued ahead of it
        top_k = int(self.runtime.spec["architecture"]["top_k"])
        self.pinned_index = torch.empty((1, top_k), dtype=torch.long)
        if pin:
            self.pinned_index = self.pinned_index.pin_memory()

    def _pointers(self) -> None:
        """Byte addresses of every expert's host copy and every slot (for the driver-API copier)."""

        def strides(t):
            return t.data_ptr(), t[0].numel() * t.element_size()

        ok = all(t.is_contiguous() for t in (self.host_gate_up, self.host_down, self.slot_gate_up, self.slot_down))
        self.copy_pointers = None
        if ok:
            hgu, gu = strides(self.host_gate_up)
            hd, d = strides(self.host_down)
            sgu, _ = strides(self.slot_gate_up)
            sd, _ = strides(self.slot_down)
            self.copy_pointers = (hgu, hd, sgu, sd, gu, d)

    def copy_expert(self, expert: int, slot: int, stream_handle: int | None) -> None:
        """Copy one expert's weights from host memory into a slot on the given stream (current if None)."""

        runtime = self.runtime
        if runtime.copier is not None and self.copy_pointers is not None:
            hgu, hd, sgu, sd, gu, d = self.copy_pointers
            handle = runtime.torch.cuda.current_stream().cuda_stream if stream_handle is None else stream_handle
            runtime.copier.copy(sgu + slot * gu, hgu + expert * gu, gu, handle)
            runtime.copier.copy(sd + slot * d, hd + expert * d, d, handle)
            return
        self.slot_gate_up[slot].copy_(self.host_gate_up[expert], non_blocking=True)
        self.slot_down[slot].copy_(self.host_down[expert], non_blocking=True)

    def _pin_in_place(self, pin: bool) -> None:
        """Experts already on the host: pin them and drop the unpinned copy, so host memory
        holds one copy of every expert (61 GB for Qwen3-30B-A3B, 93 GB for Mixtral-8x7B)."""

        torch = self.torch
        experts = self.block.experts
        if experts.gate_up_proj.device.type != "cpu" or experts.down_proj.device.type != "cpu":
            raise RuntimeError("experts_on_host needs the expert weights on the CPU (see place_non_experts)")
        for name in ("gate_up_proj", "down_proj"):
            data = getattr(experts, name).detach()
            if pin and not data.is_pinned():
                data = data.pin_memory()
            setattr(experts, name, torch.nn.Parameter(data, requires_grad=False))
        self.resident_gate_up = experts.gate_up_proj
        self.resident_down = experts.down_proj
        self.host_gate_up = self.resident_gate_up.data
        self.host_down = self.resident_down.data
        self.on_host = True

    def release(self) -> None:
        self.block.forward = self.original_forward

    def _staged_prefill(self, hidden_states):
        """Prefill with the experts on the host: copy this layer's full expert set into the
        runtime's staging buffer and run the experts module of the model's own class on it,
        exactly the block's forward (gate, then experts) with the weights read from the stage."""

        batch, length, dim = hidden_states.shape
        flat = hidden_states.view(-1, dim)
        _logits, weights, indices = self.block.gate(flat)
        staged = self.runtime.stage(self)
        return staged(flat, indices, weights).reshape(batch, length, dim)

    # ------------------------------------------------------------------ forward
    def forward(self, hidden_states):
        runtime = self.runtime
        torch = self.torch
        batch, length, dim = hidden_states.shape
        # "auto" (used by the trace generator's own decode loop): several tokens means prefill
        if runtime.mode == "prefill" or (runtime.mode == "auto" and length > 1):
            if self.on_host:
                return self._staged_prefill(hidden_states)
            return self.original_forward(hidden_states)
        if batch != 1 or length != 1:
            raise RuntimeError("offloaded decode expects one token at a time")
        entered = time.perf_counter_ns()
        flat = hidden_states.view(-1, dim)
        _logits, weights, indices = self.block.gate(flat)
        pf = runtime.prefetch if runtime.policy is not None else None
        nxt = runtime.layers[self.ordinal + 1] if (pf is not None and pf.q > 0
                                                   and self.ordinal + 1 < len(runtime.layers)) else None
        sync_start = time.perf_counter_ns()
        request = indices[0].to("cpu", non_blocking=False).numpy().astype(np.int64)
        gates = weights[0].float().to("cpu").numpy().astype(np.float64)
        runtime.token_sync_ns += time.perf_counter_ns() - sync_start  # the CPU waiting for the GPU
        runtime.observe_routing(self.ordinal, request, gates)
        guess = None
        if nxt is not None:
            # queued after the routing sync: the GPU computes the guess while the CPU makes this layer's
            # demand decision; it is read (already complete) once the expert compute is queued
            guess = self._queue_guess(flat, nxt, pf)

        if runtime.policy is None:  # resident reference
            if self.on_host:
                raise RuntimeError("the resident reference needs every expert on the device")
            return self.block.experts(flat, indices, weights).view(batch, length, dim)

        t0 = time.perf_counter_ns()
        result = runtime.policy.step(self.ordinal, runtime.current_token, request, gates)
        t1 = time.perf_counter_ns()
        runtime.token_policy_ns += t1 - t0
        for expert in result.evicted:
            slot = int(self.slot_of[expert])
            if slot < 0:
                raise RuntimeError("evicting an expert that has no slot")
            self.slot_of[expert] = -1
            self.free_slots.append(slot)
        if result.loaded.size:
            start_event = runtime.event(timing=True) if runtime.cuda else None
            end_event = runtime.event(timing=True) if runtime.cuda else None
            if start_event is not None:
                start_event.record()
            for expert in result.loaded:
                if not self.free_slots:
                    raise RuntimeError("no free slot for an admitted expert")
                slot = self.free_slots.pop()
                pending = self.pending.pop(slot, None)
                if pending is not None:  # an evicted, unused prefetch may still be copying into this slot
                    if not pending.query():
                        runtime.token_slot_stalls += 1  # it was: this demand copy really waits
                    torch.cuda.current_stream().wait_event(pending)
                    runtime.token_slot_waits += 1
                self.slot_of[expert] = slot
                self.copy_expert(int(expert), slot, None)
            if end_event is not None:
                end_event.record()
                runtime.token_copy_events.append((start_event, end_event))
            runtime.token_transfers += int(result.loaded.size)
            runtime.token_bytes += int(result.loaded.size) * runtime.expert_bytes
        if self.inflight:
            # the experts compute only after the prefetched experts the request uses have arrived;
            # the copy stream is FIFO, so waiting on the last-issued one covers the rest
            needed = [self.inflight[int(e)] for e in request if int(e) in self.inflight]
            if needed and runtime.cuda:
                _order, event = max(needed, key=lambda item: item[0])
                if not event.query():
                    runtime.token_needed_stalls += 1  # the experts compute will wait for a prefetch copy
                if runtime.detailed_timing:
                    before = runtime.event(timing=True)
                    after = runtime.event(timing=True)
                    before.record()
                    torch.cuda.current_stream().wait_event(event)
                    after.record()
                    runtime.token_stall_events.append((before, after))
                else:
                    torch.cuda.current_stream().wait_event(event)
            self.inflight.clear()
        runtime.token_issue_ns += time.perf_counter_ns() - t1  # slot bookkeeping + copy issue
        after_demand = None
        slots = self.slot_of[request]
        if (slots < 0).any():
            raise RuntimeError("a requested expert is not resident after admission")
        self.pinned_index[0].copy_(torch.from_numpy(slots))
        slot_index = self.pinned_index.to(indices.device, non_blocking=True)
        if nxt is not None and runtime.cuda:
            # the prefetch copies may start once this layer's demand copies and its (tiny) slot-index copy
            # are done: neither then queues behind a 12.6 MB prefetch copy on the copy engine
            after_demand = runtime.event()
            after_demand.record()
        out = self.slot_experts(flat, slot_index, weights).view(batch, length, dim)
        if nxt is not None:
            # the expert compute is queued: the prefetch decision runs on the CPU while the GPU computes
            self._prefetch_next(nxt, guess, pf, after_demand)
        runtime.token_moe_ns += time.perf_counter_ns() - entered
        return out

    def _queue_guess(self, flat, nxt: "OffloadedMoeLayer", pf: PrefetchConfig):
        """Queue the guess: one matrix-vector kernel (layer l+1's router weights, pre-scaled for ``g1n``) and an
        asynchronous copy of its 64 bf16 logits to pinned host memory; ``_read_guess`` ranks them on the CPU."""

        torch = self.torch
        logits = torch.nn.functional.linear(flat, self.runtime.guess_weight(self.ordinal, pf.variant))[0]
        if not self.runtime.cuda:
            return logits.detach().float().numpy()
        host = getattr(self, "_guess_host", None)
        if host is None or host.numel() != logits.numel():
            host = self._guess_host = torch.empty(logits.numel(), dtype=torch.int16).pin_memory()
        host.copy_(logits.view(torch.int16), non_blocking=True)
        event = self.runtime.event()
        event.record()
        return event

    def _read_guess(self, handle, m: int) -> np.ndarray:
        """The top m experts by logit (stable on ties: the smaller expert id first)."""

        if isinstance(handle, np.ndarray):
            logits = handle
        else:
            handle.synchronize()  # already complete: queued before this layer's demand copies and expert compute
            bits = self._guess_host.numpy().view(np.uint16).astype(np.uint32) << 16  # bf16 -> float32, exact
            logits = bits.view(np.float32)
        return np.argsort(-logits, kind="stable")[:m].astype(np.int64)

    def _guess(self, flat, nxt: "OffloadedMoeLayer", pf: PrefetchConfig):
        """The same ranking computed on the device (tests and the recorder)."""

        torch = self.torch
        logits = torch.nn.functional.linear(flat, self.runtime.guess_weight(self.ordinal, pf.variant))[0].detach().float()
        return torch.from_numpy(np.argsort(-logits.cpu().numpy(), kind="stable")[: pf.m].astype(np.int64))

    def _prefetch_next(self, nxt: "OffloadedMoeLayer", guess: np.ndarray, pf: PrefetchConfig, after_demand) -> None:
        """Admit up to q guessed experts of layer l+1 (the policy picks the victims) and copy them on the
        copy stream, after this layer's demand copies, overlapping this layer's expert compute."""

        runtime = self.runtime
        torch = self.torch
        t0 = time.perf_counter_ns()
        guess = self._read_guess(guess, pf.m)
        runtime.observe_guess(nxt.ordinal, guess)
        evicted, loaded = runtime.policy.prefetch(nxt.ordinal, guess, pf.q)
        t1 = time.perf_counter_ns()
        runtime.token_prefetch_policy_ns += t1 - t0
        for expert in evicted:
            slot = int(nxt.slot_of[expert])
            if slot < 0:
                raise RuntimeError("prefetch evicting an expert that has no slot")
            nxt.slot_of[expert] = -1
            nxt.free_slots.append(slot)
        if loaded.size:
            stream = runtime.copy_stream
            if stream is None:  # CPU (tests): plain copies
                for expert in loaded:
                    slot = nxt.free_slots.pop()
                    nxt.slot_of[expert] = slot
                    runtime.prefetch_issued += 1
                    nxt.slot_gate_up[slot].copy_(nxt.host_gate_up[expert])
                    nxt.slot_down[slot].copy_(nxt.host_down[expert])
                    nxt.inflight[int(expert)] = (runtime.prefetch_issued, None)
            else:
                stream.wait_event(after_demand)
                detailed = runtime.detailed_timing
                if detailed:
                    start = runtime.event(timing=True)
                    end = runtime.event(timing=True)
                with torch.cuda.stream(stream):
                    if detailed:
                        start.record(stream)
                    for expert in loaded:
                        if not nxt.free_slots:
                            raise RuntimeError("no free slot for a prefetched expert")
                        slot = nxt.free_slots.pop()
                        nxt.slot_of[expert] = slot
                        runtime.prefetch_issued += 1
                        nxt.copy_expert(int(expert), slot, stream.cuda_stream)
                        done = runtime.event()
                        done.record(stream)
                        nxt.pending[slot] = done
                        nxt.inflight[int(expert)] = (runtime.prefetch_issued, done)
                    if detailed:
                        end.record(stream)
                if detailed:
                    runtime.token_prefetch_events.append((start, end))
            runtime.token_prefetch_transfers += int(loaded.size)
            runtime.token_prefetch_bytes += int(loaded.size) * runtime.expert_bytes
        runtime.token_prefetch_issue_ns += time.perf_counter_ns() - t1


def place_non_experts(model, device, layer_indices) -> dict[str, int]:
    """Move every parameter and buffer to ``device`` except the expert weights of the given
    MoE layers, which stay where they are (the host). Tied parameters stay tied."""

    import torch

    expert_params = set()
    for index in layer_indices:
        experts = model.model.layers[int(index)].mlp.experts
        expert_params.update(id(p) for p in experts.parameters(recurse=False))
    moved: dict[int, torch.nn.Parameter] = {}
    counts = {"moved_bytes": 0, "host_bytes": 0}
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None:
                continue
            if id(param) in expert_params:
                counts["host_bytes"] += param.numel() * param.element_size()
                continue
            if id(param) not in moved:
                moved[id(param)] = torch.nn.Parameter(param.data.to(device), requires_grad=False)
                counts["moved_bytes"] += param.numel() * param.element_size()
            module._parameters[name] = moved[id(param)]
        for name, buffer in list(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = buffer.to(device)
    return counts


class OffloadRuntime:
    def __init__(self, model, spec: dict[str, Any], device, policy: CachePolicy | None, capacity: int,
                 compute: str = "module", experts_on_host: bool = False,
                 prefetch: PrefetchConfig | None = None, copy_api: str = "torch") -> None:
        import torch

        self.torch = torch
        if compute != "module":
            raise ValueError("only compute='module' (the transformers experts module on the slots) is implemented")
        self.compute = compute
        self.model = model
        self.spec = spec
        self.device = device
        self.cuda = device.type == "cuda"
        self.policy = policy
        self.capacity = capacity
        self.prefetch = prefetch
        self.copy_stream = torch.cuda.Stream(device=device) if self.cuda else None
        if copy_api not in ("torch", "driver"):
            raise ValueError("copy_api is 'torch' or 'driver'")
        self.copy_api = copy_api
        self.copier = DriverCopier() if (copy_api == "driver" and self.cuda) else None
        self.detailed_timing = True  # prefetch-copy and stall events (diagnostics); the runners' --lean sets False
        self._guess_weights: dict[tuple[int, str], Any] = {}
        self.prefetch_issued = 0
        self.guesses: list[tuple[int, np.ndarray]] = []
        self.guess_reference: dict[int, np.ndarray] | None = None  # recorded guesses per sequence (exactness)
        self.guess_log: dict[int, np.ndarray] = {}  # live guesses per sequence
        self.host_monitor: HostMonitor | None = None  # timing-noise diagnostics (interleaved_prefetch --monitor)
        self._norm_ratio: dict[int, Any] = {}
        self.mode = "prefill"
        self.experts_on_host = bool(experts_on_host)
        if self.experts_on_host and policy is None:
            raise RuntimeError("experts_on_host needs an eviction policy (the resident reference is impossible)")
        self.expert_bytes = int(spec["architecture"]["expert_bytes"])
        self.layers: list[OffloadedMoeLayer] = []
        layer_indices = [int(v) for v in spec["architecture"]["layer_indices"]]
        decoder_layers = model.model.layers
        for ordinal, index in enumerate(layer_indices):
            block = decoder_layers[index].mlp
            if not (hasattr(block, "gate") and hasattr(block, "experts")):
                raise RuntimeError(f"layer {index} has no sparse MoE block")
            self.layers.append(OffloadedMoeLayer(block, ordinal, self))
        if policy is not None:
            for layer in self.layers:
                layer.allocate(capacity, device, experts_on_host=self.experts_on_host)
        self._stage = None
        if self.experts_on_host:
            self._stage = self._make_stage(self.layers[0])
        self.reset_token_counters()
        self.current_token = -1
        self.routing: list[tuple[int, np.ndarray, np.ndarray]] = []

    def _make_stage(self, layer: "OffloadedMoeLayer"):
        """One device buffer for a full expert set, and an experts module of the model's own
        class (all E experts) whose parameters are that buffer."""

        import copy

        torch = self.torch
        experts = layer.block.experts
        gate_up = torch.empty(tuple(layer.host_gate_up.shape), dtype=layer.host_gate_up.dtype, device=self.device)
        down = torch.empty(tuple(layer.host_down.shape), dtype=layer.host_down.dtype, device=self.device)
        with torch.device("meta"):
            module = type(experts)(copy.deepcopy(experts.config))
        module.gate_up_proj = torch.nn.Parameter(gate_up, requires_grad=False)
        module.down_proj = torch.nn.Parameter(down, requires_grad=False)
        return module.eval()

    def stage(self, layer: "OffloadedMoeLayer"):
        """Copy one layer's experts into the staging buffer (queued on the current stream,
        so the previous layer's prefill finishes reading it first) and return the module."""

        stage = self._stage
        if tuple(stage.gate_up_proj.shape) != tuple(layer.host_gate_up.shape):
            raise RuntimeError("MoE layers with different expert shapes need one stage each")
        stage.gate_up_proj.data.copy_(layer.host_gate_up, non_blocking=True)
        stage.down_proj.data.copy_(layer.host_down, non_blocking=True)
        return stage

    def start_path(self, policy: CachePolicy | None) -> None:
        """Begin a workload path cold: fresh policy state and empty slots."""

        self.policy = policy
        if policy is not None:
            for layer in self.layers:
                layer.slot_of.fill(-1)
                layer.free_slots = list(range(self.capacity - 1, -1, -1))

    def event(self, timing: bool = False):
        """A CUDA event from a pool reused every token (creating events costs microseconds of CPU each);
        every event of a token is recorded and read before the next token starts."""

        pool, used = (self._timing_events, self._timing_used) if timing else (self._plain_events, self._plain_used)
        if used >= len(pool):
            pool.append(self.torch.cuda.Event(enable_timing=timing))
        if timing:
            self._timing_used += 1
        else:
            self._plain_used += 1
        return pool[used]

    def reset_token_counters(self) -> None:
        """Per-token counters. Called before every decode step, after the previous step's device sync,
        so every prefetch copy of the previous token has completed and every event can be reused."""

        if not hasattr(self, "_timing_events"):
            self._timing_events, self._plain_events = [], []
        self._timing_used = self._plain_used = 0

        self.token_policy_ns = 0
        self.token_issue_ns = 0
        self.token_transfers = 0
        self.token_bytes = 0
        self.token_copy_events: list[tuple[Any, Any]] = []
        self.token_prefetch_policy_ns = 0
        self.token_prefetch_issue_ns = 0
        self.token_prefetch_transfers = 0
        self.token_prefetch_bytes = 0
        self.token_prefetch_events: list[tuple[Any, Any]] = []
        self.token_stall_events: list[tuple[Any, Any]] = []
        self.token_slot_waits = 0
        self.token_slot_stalls = 0
        self.token_needed_stalls = 0
        self.token_sync_ns = 0  # in the routing sync (CPU waiting for the GPU)
        self.token_moe_ns = 0  # inside the offloaded MoE forwards (decode)
        for layer in getattr(self, "layers", []):
            layer.pending.clear()
            layer.inflight.clear()

    def guess_weight(self, ordinal: int, variant: str):
        """Layer l+1's router weight for the guess made at layer l; for ``g1n`` each column scaled by
        w_{l+1} / w_l (the post-attention RMSNorm gains), so the guess is a single matrix-vector product."""

        key = (ordinal, variant)
        weight = self._guess_weights.get(key)
        if weight is None:
            indices = [int(v) for v in self.spec["architecture"]["layer_indices"]]
            gate = self.model.model.layers[indices[ordinal + 1]].mlp.gate.weight
            if variant == "g1":
                weight = gate
            else:
                weight = (gate.float() * self.norm_ratio(ordinal).float()[None, :]).to(gate.dtype)
            self._guess_weights[key] = weight
        return weight

    def norm_ratio(self, ordinal: int):
        """g1n: layer l+1's post-attention RMSNorm gain over layer l's (as record_guesses computes it)."""

        ratio = self._norm_ratio.get(ordinal)
        if ratio is None:
            indices = [int(v) for v in self.spec["architecture"]["layer_indices"]]
            here = self.model.model.layers[indices[ordinal]].post_attention_layernorm.weight
            there = self.model.model.layers[indices[ordinal + 1]].post_attention_layernorm.weight
            ratio = self._norm_ratio[ordinal] = (there.float() / here.float()).to(here.dtype)
        return ratio

    def observe_routing(self, ordinal: int, request: np.ndarray, gates: np.ndarray) -> None:
        self.routing.append((ordinal, request.copy(), gates.copy()))

    def observe_guess(self, ordinal: int, guess: np.ndarray) -> None:
        self.guesses.append((ordinal, guess.copy()))

    def ms(self, events: list[tuple[Any, Any]]) -> float:
        return sum(start.elapsed_time(end) for start, end in events) if self.cuda else 0.0

    def release(self) -> None:
        for layer in self.layers:
            layer.release()

    def copy_ms(self) -> float:
        if not self.cuda:
            return 0.0
        total = 0.0
        for start, end in self.token_copy_events:
            total += start.elapsed_time(end)
        return total


class HostMonitor:
    """Linux diagnostics for timing noise: the CPU the decode thread runs on (per token), and every host
    CPU's busy time over a sequence (/proc/stat is not namespaced, so other tenants' load is visible)."""

    def __init__(self) -> None:
        self.ok = Path("/proc/thread-self/stat").exists() and Path("/proc/stat").exists()

    def cpu(self) -> int:
        if not self.ok:
            return -1
        fields = Path("/proc/thread-self/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[36])  # field 39 of stat(5), counted after the command name

    def mhz(self, cpu: int) -> int:
        """The CPU's current clock (sysfs cpufreq, kHz -> MHz), or -1 if the host hides it."""

        try:
            return int(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read_text()) // 1000
        except (OSError, ValueError):
            return -1

    def busy(self) -> list[int]:
        if not self.ok:
            return []
        out = []
        for line in Path("/proc/stat").read_text().splitlines()[1:]:
            if not line.startswith("cpu"):
                break
            v = [int(x) for x in line.split()[1:]]
            out.append(sum(v) - v[3] - v[4])  # all jiffies but idle and iowait
        return out


def thread_cpu(pids: list[int]) -> dict[int, tuple[int, int]]:
    """tid -> (user + system jiffies, CPU last run on) for every thread of the given processes (Linux)."""

    out: dict[int, tuple[int, int]] = {}
    for pid in pids:
        root = Path(f"/proc/{pid}/task")
        if not root.exists():
            continue
        for task in root.iterdir():
            try:
                fields = (task / "stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            out[int(task.name)] = (int(fields[11]) + int(fields[12]), int(fields[36]))
    return out


def bandwidth_probe(layer: OffloadedMoeLayer, device, reps: int = 100) -> dict[str, float]:
    """Host-to-device copy of one expert (pinned), averaged; the price of one transfer."""

    import torch

    if device.type != "cuda":
        return {}
    slot = 0
    expert = 0
    for _ in range(5):
        layer.slot_gate_up[slot].copy_(layer.host_gate_up[expert], non_blocking=True)
        layer.slot_down[slot].copy_(layer.host_down[expert], non_blocking=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        layer.slot_gate_up[slot].copy_(layer.host_gate_up[expert], non_blocking=True)
        layer.slot_down[slot].copy_(layer.host_down[expert], non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / reps
    nbytes = layer.host_gate_up[expert].numel() * layer.host_gate_up.element_size() + layer.host_down[expert].numel() * layer.host_down.element_size()
    return {"ms_per_expert": ms, "expert_bytes": int(nbytes), "gb_per_s": nbytes / ms / 1e6}


def run_path(runtime: OffloadRuntime, tokenizer, spec: dict[str, Any], workload: dict[str, Any], *,
             teacher_force: bool, limit_sequences: int | None, log) -> dict[str, Any]:
    import torch

    model = runtime.model
    device = runtime.device
    sequences_by_id = {int(s["sequence_id"]): s for s in spec["sequences"]}
    generation = spec["trace_generation"]
    layers = len(spec["architecture"]["layer_indices"])
    eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    rows = []
    for count, item in enumerate(workload["sequences"]):
        if limit_sequences is not None and count >= limit_sequences:
            break
        record = sequences_by_id[int(item["source_sequence_id"])]
        if runtime.policy is not None:
            runtime.policy.begin_sequence()
        encoded = tokenizer(record["prompt"], truncation=True, max_length=int(generation["max_prompt_tokens"]),
                            add_special_tokens=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
        if int(attention_mask.sum().item()) != int(record["prompt_length"]):
            raise RuntimeError(f"prompt length differs from the trace for sequence {record['sequence_id']}")
        recorded = [int(v) for v in record["generated_token_ids"]]
        trace_requests = np.asarray(record["trace_requested_expert_ids"], dtype=np.int64).reshape(len(recorded), layers, -1)

        runtime.mode = "prefill"
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            past = outputs.past_key_values
            logits = outputs.logits
            if runtime.cuda:
                torch.cuda.synchronize()
            predicted_first = int(torch.argmax(logits[:, -1, :].float(), dim=-1).item())

            runtime.mode = "decode"
            wall_ns = 0
            policy_ns = 0
            issue_ns = 0
            copy_ms = 0.0
            transfers = 0
            nbytes = 0
            routing_events = 0
            routing_matches = 0
            token_agreements = 0
            tokens = 0
            pf = {"transfers": 0, "copy_ms": 0.0, "stall_ms": 0.0, "policy_ns": 0, "issue_ns": 0, "slot_waits": 0,
                  "slot_stalls": 0, "needed_stalls": 0, "guess_events": 0, "guess_matches": 0, "sync_ns": 0, "moe_ns": 0}
            token_wall: list[float] = []
            token_cpu: list[int] = []
            token_mhz: list[int] = []
            monitor = runtime.host_monitor
            busy_start = monitor.busy() if monitor is not None else []
            pf_width = runtime.prefetch.m if runtime.prefetch is not None else 0
            live_guesses = np.full((len(recorded), layers, max(pf_width, 1)), -1, dtype=np.int16)
            reference = None if runtime.guess_reference is None else runtime.guess_reference.get(int(record["sequence_id"]))
            predictions = [predicted_first]
            next_token = recorded[0] if teacher_force else predicted_first
            transfers_before = runtime.policy.transfers if runtime.policy is not None else 0
            pf_before = ({key: getattr(runtime.policy, attr, 0) for key, attr in PREFETCH_COUNTERS.items()}
                         if runtime.policy is not None else {})
            for step in range(len(recorded)):
                token_id = int(next_token)
                runtime.current_token = token_id
                runtime.reset_token_counters()
                runtime.routing = []
                runtime.guesses = []
                attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)), dim=1)
                step_input = torch.tensor([[token_id]], dtype=torch.long, device=device)
                t0 = time.perf_counter_ns()
                if runtime.policy is not None and hasattr(runtime.policy, "begin_token"):
                    # split-phase: the token is known now; workers start the P2 scores
                    runtime.policy.begin_token(token_id)
                    runtime.token_policy_ns += time.perf_counter_ns() - t0
                outputs = model(input_ids=step_input, attention_mask=attention_mask, past_key_values=past,
                                use_cache=True, return_dict=True)
                argmax = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1)
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
                nbytes += runtime.token_bytes + runtime.token_prefetch_bytes
                pf["transfers"] += runtime.token_prefetch_transfers
                pf["copy_ms"] += runtime.ms(runtime.token_prefetch_events)
                pf["stall_ms"] += runtime.ms(runtime.token_stall_events)
                pf["policy_ns"] += runtime.token_prefetch_policy_ns
                pf["issue_ns"] += runtime.token_prefetch_issue_ns
                pf["slot_waits"] += runtime.token_slot_waits
                pf["sync_ns"] += runtime.token_sync_ns
                pf["moe_ns"] += runtime.token_moe_ns
                pf["slot_stalls"] += runtime.token_slot_stalls
                pf["needed_stalls"] += runtime.token_needed_stalls
                for ordinal, guess in runtime.guesses:
                    live_guesses[step, ordinal, : guess.size] = guess
                    if reference is not None:
                        pf["guess_events"] += 1
                        pf["guess_matches"] += int(np.array_equal(guess, reference[step, ordinal, : guess.size]))
                past = outputs.past_key_values
                predicted = int(argmax.item())
                predictions.append(predicted)
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
                    next_token = recorded[step + 1] if teacher_force else predicted
                    if not teacher_force and predicted in eos_ids and bool(generation["stop_on_eos"]):
                        break
        rows.append({
            "sequence_position": int(item["position"]),
            "source_sequence_id": int(item["source_sequence_id"]),
            "domain": str(item["domain"]),
            "prompt_length": int(record["prompt_length"]),
            "tokens": tokens,
            "expert_transfers": int(runtime.policy.transfers - transfers_before) if runtime.policy is not None else 0,
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
            "prefetch_transfers": pf["transfers"],
            "prefetch_copy_ms": pf["copy_ms"],
            "prefetch_stall_ms": pf["stall_ms"],
            "prefetch_policy_ms": pf["policy_ns"] / 1e6,
            "prefetch_issue_ms": pf["issue_ns"] / 1e6,
            "sync_ms": pf["sync_ns"] / 1e6,
            "moe_ms": pf["moe_ns"] / 1e6,
            "prefetch_slot_waits": pf["slot_waits"],
            "prefetch_slot_stalls": pf["slot_stalls"],
            "prefetch_needed_stalls": pf["needed_stalls"],
            "guess_events": pf["guess_events"],
            "guess_matches": pf["guess_matches"],
            "token_wall_ms": token_wall,
        })
        if monitor is not None:
            busy_end = monitor.busy()
            rows[-1]["token_cpu"] = token_cpu
            rows[-1]["token_mhz"] = token_mhz
            rows[-1]["host_busy_jiffies"] = [b - a for a, b in zip(busy_start, busy_end)]
        if runtime.policy is not None:
            for key, attr in PREFETCH_COUNTERS.items():
                rows[-1][key] = int(getattr(runtime.policy, attr, 0) - pf_before.get(key, 0))
        if runtime.prefetch is not None:
            runtime.guess_log[int(record["sequence_id"])] = live_guesses
        if runtime.policy is not None and rows[-1]["expert_transfers"] != transfers:
            raise RuntimeError("policy transfer count and layer-level transfer count disagree")
        if runtime.policy is not None and rows[-1]["prefetch_transfers_policy"] != pf["transfers"]:
            raise RuntimeError("policy prefetch count and layer-level prefetch count disagree")
        if (count + 1) % 10 == 0 or count == 0:
            r = rows[-1]
            log(f"    seq {count + 1}/{len(workload['sequences'])}: {r['tokens']} tok, "
                f"{r['wall_ms'] / max(r['tokens'], 1):.1f} ms/tok wall, {r['policy_ms'] / max(r['tokens'], 1):.2f} policy, "
                f"{r['issue_ms'] / max(r['tokens'], 1):.2f} issue, "
                f"{r['copy_ms'] / max(r['tokens'], 1):.2f} copy, {r['expert_transfers']} transfers, "
                f"routing {r['routing_matches']}/{r['routing_events']}")
    totals = {
        "tokens": sum(r["tokens"] for r in rows),
        "expert_transfers": sum(r["expert_transfers"] for r in rows),
        "bytes_transferred": sum(r["bytes_transferred"] for r in rows),
        "wall_ms": sum(r["wall_ms"] for r in rows),
        "policy_ms": sum(r["policy_ms"] for r in rows),
        "issue_ms": sum(r["issue_ms"] for r in rows),
        "copy_ms": sum(r["copy_ms"] for r in rows),
        "routing_events": sum(r["routing_events"] for r in rows),
        "routing_matches": sum(r["routing_matches"] for r in rows),
        "token_agreements": sum(r["token_agreements"] for r in rows),
        "token_comparisons": sum(r["token_comparisons"] for r in rows),
    }
    for key in PREFETCH_ROW_KEYS:
        totals[key] = sum(r.get(key, 0) for r in rows)
    tokens = max(totals["tokens"], 1)
    totals["ms_per_token_wall"] = totals["wall_ms"] / tokens
    totals["ms_per_token_policy"] = totals["policy_ms"] / tokens
    totals["ms_per_token_issue"] = totals["issue_ms"] / tokens
    totals["ms_per_token_copy"] = totals["copy_ms"] / tokens
    totals["transfers_per_token"] = totals["expert_transfers"] / tokens
    totals["tokens_per_second"] = 1e3 * tokens / max(totals["wall_ms"], 1e-9)
    totals["routing_match_fraction"] = totals["routing_matches"] / max(totals["routing_events"], 1)
    totals["token_agreement_fraction"] = totals["token_agreements"] / max(totals["token_comparisons"], 1)
    return {"workload": workload["name"], "regime": workload["regime"], "sequences": rows, "totals": totals}


def load_model(spec: dict[str, Any], device, dtype, attn: str, cache_dir: str | None = None,
               experts_on_host: bool = False, log=print):
    """The trace's model at its pinned revision. Resident: straight onto the device.
    Experts on host: on the CPU first, then everything but the expert weights to the device."""

    from transformers import AutoModelForCausalLM

    where = "host experts" if experts_on_host else str(device)
    log(f"loading {spec['model']} @ {spec['model_revision']} ({dtype}, attn={attn}) on {where}")
    kwargs: dict[str, Any] = {"revision": spec["model_revision"], "dtype": dtype, "low_cpu_mem_usage": True,
                              "attn_implementation": attn}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if device.type != "cpu" and not experts_on_host:
        kwargs["device_map"] = {"": str(device)}
    model = AutoModelForCausalLM.from_pretrained(spec["model"], **kwargs).eval()
    if experts_on_host:
        counts = place_non_experts(model, device, spec["architecture"]["layer_indices"])
        log(f"  {counts['moved_bytes'] / 1e9:.2f} GB to {device}, {counts['host_bytes'] / 1e9:.2f} GB of experts on the host")
    elif device.type == "cpu":
        model.to(device)
    return model


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None, help="defaults to the trace's (eager)")
    parser.add_argument("--backend", default="numba")
    parser.add_argument("--score-scope", default="candidates")
    parser.add_argument("--free-running", action="store_true", help="feed the model's own argmax instead of the recorded tokens")
    parser.add_argument("--limit-sequences", type=int)
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--p2-bundle", type=Path, help="split-phase: request-stale ranker (P2) for layers 1..L-1")
    parser.add_argument("--p1-bundle", type=Path, help="split-phase: token-stale ranker (P1) for layer 0")
    parser.add_argument("--workers", type=int, default=1, help="split-phase background workers")
    parser.add_argument("--allow-foreign-bundle", action="store_true",
                        help="unseen-trace tests: the policy was fitted on a different trace by design")
    parser.add_argument("--experts-on-host", action="store_true",
                        help="load on the CPU, keep the expert weights pinned on the host (models larger than the GPU)")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = json.loads(args.spec.read_text())
    arch = spec["architecture"]
    if not arch["top_k"] <= args.capacity <= arch["num_experts"]:
        raise SystemExit("capacity must satisfy top_k <= B <= E")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    bundle = PolicyBundle.load(args.policy_bundle) if args.policy_bundle else None
    if args.policy in LEARNED_VARIANTS and bundle is None:
        raise SystemExit("learned policies need --policy-bundle")
    if bundle is not None and str(bundle.payload["trace_hash"]) != str(spec["trace_hash"]) and not args.allow_foreign_bundle:
        raise SystemExit("policy bundle and replay spec come from different traces")
    if args.policy == "splitphase" and (args.p2_bundle is None or args.p1_bundle is None):
        raise SystemExit("splitphase needs --p2-bundle and --p1-bundle")
    if args.experts_on_host and args.policy == "resident":
        raise SystemExit("--policy resident needs every expert on the device")

    def log(message: str) -> None:
        print(message, flush=True)

    workloads = spec["workloads"]
    if args.workloads:
        by_name = {w["name"]: w for w in workloads}
        workloads = [by_name[n] for n in args.workloads]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for workload in workloads:
        out = args.output_dir / f"{args.policy}_B{args.capacity}_{workload['name']}.json"
        if args.skip_existing and out.exists():
            log(f"skip {out.name} (exists)")
            continue
        pending.append((workload, out))
    if not pending:
        log("nothing to run")
        return

    torch.manual_seed(int(spec["trace_generation"]["seed"]))
    attn = args.attn_implementation or spec["trace_generation"].get("attn_implementation", "eager")
    model = load_model(spec, device, dtype, attn, args.model_cache_dir, args.experts_on_host, log)
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["model_revision"],
                                              **({"cache_dir": args.model_cache_dir} if args.model_cache_dir else {}))
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved and spec["model_revision"] and resolved != spec["model_revision"]:
        raise SystemExit(f"resolved revision {resolved} differs from the trace's {spec['model_revision']}")

    services = []

    def new_policy():
        if args.policy == "resident":
            return None
        if args.policy == "splitphase":
            from splitphase_service import SplitPhasePolicy, SplitPhaseService

            for old in services:
                old.close()
            services.clear()
            service = SplitPhaseService(str(args.p2_bundle), str(args.p1_bundle), int(args.capacity),
                                        workers=int(args.workers), num_layers=int(arch["num_moe_layers"]),
                                        num_experts=int(arch["num_experts"]), top_k=int(arch["top_k"]))
            services.append(service)
            return SplitPhasePolicy(service, int(arch["num_moe_layers"]), int(arch["num_experts"]),
                                    int(args.capacity), int(arch["top_k"]))
        return make_policy(args.policy, num_layers=int(arch["num_moe_layers"]), num_experts=int(arch["num_experts"]),
                           capacity=int(args.capacity), top_k=int(arch["top_k"]), bundle=bundle,
                           backend=args.backend, score_scope=args.score_scope)

    policy = new_policy()
    runtime = OffloadRuntime(model, spec, device, policy, int(args.capacity), compute="module",
                             experts_on_host=args.experts_on_host)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine()
    probe = bandwidth_probe(runtime.layers[0], device) if policy is not None else {}
    if probe:
        log(f"H2D probe: {probe['ms_per_expert']:.3f} ms per expert ({probe['gb_per_s']:.1f} GB/s)")
    host = {
        "hostname": socket.gethostname(), "platform": platform.platform(), "device": str(device), "gpu": gpu,
        "torch": torch.__version__, "dtype": args.dtype, "attn_implementation": attn,
        "python": platform.python_version(), "experts_on_host": bool(args.experts_on_host),
    }
    try:
        import transformers

        host["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        pass

    for workload, out in pending:
        log(f"== {args.policy} B={args.capacity} {workload['name']} ({len(workload['sequences'])} sequences)")
        runtime.start_path(new_policy())
        start = time.perf_counter()
        result = run_path(runtime, tokenizer, spec, workload, teacher_force=not args.free_running,
                          limit_sequences=args.limit_sequences, log=log)
        result.update({
            "schema_version": "hardware_run_v1",
            "label": args.label,
            "model": spec["model"], "model_revision": spec["model_revision"], "trace_hash": spec["trace_hash"],
            "policy": args.policy, "capacity": int(args.capacity), "backend": args.backend if policy is not None else None,
            "score_scope": args.score_scope, "compute": "module", "teacher_forced": not args.free_running,
            "limit_sequences": args.limit_sequences, "host": host, "bandwidth_probe": probe,
            "policy_bundle": None if bundle is None else {"path": str(args.policy_bundle), "configuration": bundle.configuration},
            "elapsed_seconds": time.perf_counter() - start,
        })
        if args.policy == "splitphase":
            sp = runtime.policy
            tokens_run = max(result["totals"]["tokens"], 1)
            busy = services[0].close() if services else []
            profile = getattr(services[0], "profile", None) if services else None
            services.clear()
            result["splitphase"] = {
                "p2_bundle": str(args.p2_bundle), "p1_bundle": str(args.p1_bundle), "workers": int(args.workers),
                "p2_bundle_sha256": _sha256(args.p2_bundle), "p1_bundle_sha256": _sha256(args.p1_bundle),
                "wait_ms_per_token": sp.wait_ns / 1e6 / tokens_run,
                "decide_ms_per_token": sp.decide_ns / 1e6 / tokens_run,
                "post_ms_per_token": sp.post_ns / 1e6 / tokens_run,
                "worker_busy_ms_per_token": [b / 1e6 / tokens_run for b in busy],
                "worker_profile": profile,
            }
        out.write_text(json.dumps(result, indent=1) + "\n")
        t = result["totals"]
        log(f"   {t['tokens']} tokens, {t['ms_per_token_wall']:.2f} ms/token wall ({t['tokens_per_second']:.1f} tok/s), "
            f"policy {t['ms_per_token_policy']:.2f} ms/token, issue {t['ms_per_token_issue']:.2f}, copy {t['ms_per_token_copy']:.2f} ms/token, "
            f"{t['transfers_per_token']:.2f} transfers/token, routing match {100 * t['routing_match_fraction']:.2f}%, "
            f"token agreement {100 * t['token_agreement_fraction']:.2f}% -> {out}")
    runtime.release()


if __name__ == "__main__":
    main()
