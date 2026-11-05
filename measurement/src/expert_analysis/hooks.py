from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .metrics import DomainStatistics
from .modeling import MoeLayerSpec


@dataclass
class RoutingRecord:
    indices: torch.Tensor
    weights: torch.Tensor
    valid_mask: torch.Tensor
    local_example_for_row: torch.Tensor
    weights_from_router_output: bool


@dataclass
class BatchState:
    example_indices: np.ndarray
    attention_mask: torch.Tensor
    batch_size: int
    sequence_length: int
    routing: dict[int, RoutingRecord] = field(default_factory=dict)
    block_inputs: dict[int, torch.Tensor] = field(default_factory=dict)
    contribution_layers: set[int] = field(default_factory=set)
    gradient_layers: set[int] = field(default_factory=set)


class ExpertInstrumentation:
    """Collect routing, gate mass, and routed-output magnitudes with removable hooks."""

    def __init__(
        self,
        layer_specs: list[MoeLayerSpec],
        statistics: DomainStatistics,
        compute_gradient_attribution: bool = False,
        validation_tolerance: float = 7e-3,
    ) -> None:
        if not layer_specs:
            raise ValueError("At least one MoE layer is required")
        if statistics.num_layers != len(layer_specs):
            raise ValueError("Statistics and discovered layer counts do not match")
        if statistics.num_experts != layer_specs[0].num_experts:
            raise ValueError("Statistics and discovered expert counts do not match")
        if compute_gradient_attribution and statistics.gradient_sums is None:
            raise ValueError("Gradient storage was not allocated")
        self.layer_specs = layer_specs
        self.statistics = statistics
        self.compute_gradient_attribution = compute_gradient_attribution
        self.validation_tolerance = validation_tolerance
        self.diagnostics: dict[int, dict[str, Any]] = {}
        self._handles: list[Any] = []
        self._batch: BatchState | None = None
        self._entered = False

    def __enter__(self) -> "ExpertInstrumentation":
        if self._entered:
            raise RuntimeError("Instrumentation cannot be entered twice")
        self._entered = True
        for spec in self.layer_specs:
            self._handles.append(
                spec.router.register_forward_hook(
                    self._make_router_hook(spec), with_kwargs=True
                )
            )
            if spec.capture_point == "experts_pre":
                self._handles.append(
                    spec.experts.register_forward_pre_hook(
                        self._make_experts_pre_hook(spec), with_kwargs=True
                    )
                )
            elif spec.capture_point == "block_post":
                self._handles.append(
                    spec.block.register_forward_pre_hook(
                        self._make_block_pre_hook(spec), with_kwargs=True
                    )
                )
                self._handles.append(
                    spec.block.register_forward_hook(
                        self._make_block_post_hook(spec), with_kwargs=True
                    )
                )
            else:
                raise RuntimeError(f"Unknown capture point {spec.capture_point!r}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def registered_hook_count(self) -> int:
        return len(self._handles)

    @contextmanager
    def batch(
        self, example_indices: list[int] | np.ndarray, attention_mask: torch.Tensor
    ) -> Iterator[None]:
        if not self._entered or not self._handles:
            raise RuntimeError("Enter ExpertInstrumentation before starting a batch")
        if self._batch is not None:
            raise RuntimeError("Nested instrumentation batches are not supported")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        indices = np.asarray(example_indices, dtype=np.int64)
        if len(indices) != attention_mask.shape[0]:
            raise ValueError("One global example index is required for each batch row")
        if np.any(indices < 0) or np.any(indices >= self.statistics.num_examples):
            raise IndexError("Example index outside the allocated statistics array")
        existing = self.statistics.token_counts[indices]
        if np.any(existing != 0):
            raise RuntimeError("An example is being collected more than once")
        counts = attention_mask.detach().sum(dim=1).to("cpu", dtype=torch.int64).numpy()
        if np.any(counts <= 0):
            raise RuntimeError("A tokenized example contains no valid tokens")
        self.statistics.token_counts[indices] = counts.astype(np.uint32)
        self._batch = BatchState(
            example_indices=indices,
            attention_mask=attention_mask.detach(),
            batch_size=int(attention_mask.shape[0]),
            sequence_length=int(attention_mask.shape[1]),
        )
        try:
            yield
            self._validate_completed_batch()
        finally:
            self._batch = None

    def finalize_gradients(self) -> None:
        if not self.compute_gradient_attribution:
            raise RuntimeError("Gradient attribution was not enabled")
        batch = self._require_batch()
        for spec in self.layer_specs:
            record = batch.routing.get(spec.ordinal)
            if record is None:
                raise RuntimeError(f"No routing record for layer {spec.block_name}")
            if not record.weights_from_router_output:
                raise RuntimeError(
                    f"Layer {spec.block_name} did not return the selected gate weights. "
                    "Gradient × gate attribution cannot use weights reconstructed only in a hook."
                )
            gradient = record.weights.grad
            if gradient is None:
                raise RuntimeError(
                    f"No gate-weight gradient was retained for layer {spec.block_name}; "
                    "ensure the forward pass used differentiable input embeddings"
                )
            attribution = (record.weights.detach().float() * gradient.detach().float()).abs()
            valid_indices = record.indices[record.valid_mask]
            valid_values = attribution[record.valid_mask]
            local_examples = record.local_example_for_row[record.valid_mask]
            values = self._scatter_routes(
                valid_indices,
                valid_values,
                local_examples,
                batch.batch_size,
                spec.num_experts,
            )
            assert self.statistics.gradient_sums is not None
            self.statistics.gradient_sums[
                batch.example_indices, spec.ordinal, :
            ] += values.astype(np.float32)
            batch.gradient_layers.add(spec.ordinal)

    def close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._batch = None
        self._entered = False

    def diagnostic_report(self) -> list[dict[str, Any]]:
        report = []
        for spec in self.layer_specs:
            values = dict(self.diagnostics.get(spec.ordinal, {}))
            values.update(
                {
                    "layer": spec.model_layer_index,
                    "layer_name": spec.block_name,
                    "num_experts": spec.num_experts,
                    "top_k": spec.top_k,
                    "contribution_backend": spec.contribution_backend,
                }
            )
            report.append(values)
        return report

    def _make_router_hook(self, spec: MoeLayerSpec):
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            del args, kwargs
            batch = self._require_batch()
            if spec.ordinal in batch.routing:
                raise RuntimeError(f"Router for {spec.block_name} ran more than once in one batch")
            indices, weights, full_probabilities, from_output = _extract_routing(
                output, module, spec
            )
            if indices.shape != weights.shape or indices.ndim != 2:
                raise RuntimeError(
                    f"Invalid routing shapes in {spec.block_name}: "
                    f"indices={tuple(indices.shape)}, weights={tuple(weights.shape)}"
                )
            if indices.shape[1] != spec.top_k:
                raise RuntimeError(
                    f"Expected top-{spec.top_k} routing in {spec.block_name}, got "
                    f"shape {tuple(indices.shape)}"
                )
            if indices.numel() == 0 or int(indices.min()) < 0 or int(indices.max()) >= spec.num_experts:
                raise RuntimeError(f"Out-of-range or empty expert indices in {spec.block_name}")
            if not torch.isfinite(weights).all() or bool((weights < 0).any()):
                raise RuntimeError(f"Invalid selected router weights in {spec.block_name}")

            valid_mask, local_examples = self._token_mapping(indices.shape[0], indices.device)
            if int(valid_mask.sum()) == 0:
                raise RuntimeError("The router received no valid tokens")
            record = RoutingRecord(
                indices=indices,
                weights=weights,
                valid_mask=valid_mask,
                local_example_for_row=local_examples,
                weights_from_router_output=from_output,
            )
            batch.routing[spec.ordinal] = record

            if self.compute_gradient_attribution:
                if not from_output or not weights.requires_grad:
                    raise RuntimeError(
                        f"Selected weights in {spec.block_name} are not a differentiable router "
                        "output; optional gradient attribution is unavailable for this implementation"
                    )
                weights.retain_grad()

            valid_indices = indices.detach()[valid_mask]
            valid_weights = weights.detach().float()[valid_mask]
            valid_local_examples = local_examples[valid_mask]
            counts = self._scatter_routes(
                valid_indices,
                torch.ones_like(valid_weights),
                valid_local_examples,
                batch.batch_size,
                spec.num_experts,
            )
            gate = self._scatter_routes(
                valid_indices,
                valid_weights,
                valid_local_examples,
                batch.batch_size,
                spec.num_experts,
            )
            self.statistics.routing_counts[
                batch.example_indices, spec.ordinal, :
            ] += np.rint(counts).astype(np.uint32)
            self.statistics.gate_sums[
                batch.example_indices, spec.ordinal, :
            ] += gate.astype(np.float32)
            self._record_router_diagnostics(
                spec, indices, weights, full_probabilities, valid_mask
            )

        return hook

    def _make_experts_pre_hook(self, spec: MoeLayerSpec):
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            del module
            batch = self._require_batch()
            record = batch.routing.get(spec.ordinal)
            if record is None:
                raise RuntimeError(f"Experts in {spec.block_name} ran before its router hook")
            hidden = _find_hidden_tensor(args, kwargs, spec.top_k)
            self._collect_contribution(spec, hidden, record)

        return hook

    def _make_block_pre_hook(self, spec: MoeLayerSpec):
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            del module
            batch = self._require_batch()
            batch.block_inputs[spec.ordinal] = _find_hidden_tensor(args, kwargs, spec.top_k)

        return hook

    def _make_block_post_hook(self, spec: MoeLayerSpec):
        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del module, args, kwargs, output
            batch = self._require_batch()
            record = batch.routing.get(spec.ordinal)
            hidden = batch.block_inputs.pop(spec.ordinal, None)
            if record is None or hidden is None:
                raise RuntimeError(f"Incomplete block capture for {spec.block_name}")
            self._collect_contribution(spec, hidden, record)

        return hook

    def _collect_contribution(
        self, spec: MoeLayerSpec, hidden_states: torch.Tensor, record: RoutingRecord
    ) -> None:
        batch = self._require_batch()
        if spec.ordinal in batch.contribution_layers:
            raise RuntimeError(f"Contribution for {spec.block_name} was collected twice")
        hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if hidden.shape[0] == record.indices.shape[0]:
            valid_hidden = hidden[record.valid_mask]
        elif hidden.shape[0] == int(record.valid_mask.sum()):
            valid_hidden = hidden
        else:
            raise RuntimeError(
                f"Hidden/routing token mismatch in {spec.block_name}: "
                f"hidden={hidden.shape[0]}, routed={record.indices.shape[0]}"
            )
        indices = record.indices.detach()[record.valid_mask]
        weights = record.weights.detach()[record.valid_mask]
        local_examples = record.local_example_for_row[record.valid_mask]
        route_norms = torch.zeros_like(weights, dtype=torch.float32)
        last_output_shape: list[int] | None = None
        with torch.no_grad():
            for expert_id in torch.unique(indices).tolist():
                route_rows, topk_positions = torch.where(indices == int(expert_id))
                expert_input = valid_hidden[route_rows]
                expert_output = _run_expert(spec, int(expert_id), expert_input)
                if expert_output.ndim != 2 or expert_output.shape != expert_input.shape:
                    raise RuntimeError(
                        f"Expert {expert_id} in {spec.block_name} returned shape "
                        f"{tuple(expert_output.shape)} for input {tuple(expert_input.shape)}"
                    )
                weighted = expert_output.float() * weights[
                    route_rows, topk_positions, None
                ].float()
                route_norms[route_rows, topk_positions] = torch.linalg.vector_norm(
                    weighted, ord=2, dim=-1
                )
                last_output_shape = list(expert_output.shape)
        if not torch.isfinite(route_norms).all() or bool((route_norms < 0).any()):
            raise RuntimeError(f"Invalid functional contributions in {spec.block_name}")
        values = self._scatter_routes(
            indices,
            route_norms,
            local_examples,
            batch.batch_size,
            spec.num_experts,
        )
        self.statistics.contribution_sums[
            batch.example_indices, spec.ordinal, :
        ] += values.astype(np.float32)
        batch.contribution_layers.add(spec.ordinal)

        diagnostic = self.diagnostics.setdefault(spec.ordinal, {})
        diagnostic.setdefault("expert_output_shape", last_output_shape)
        diagnostic.setdefault("nonzero_contributions", int((route_norms > 0).sum().item()))

    def _record_router_diagnostics(
        self,
        spec: MoeLayerSpec,
        indices: torch.Tensor,
        weights: torch.Tensor,
        full_probabilities: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if spec.ordinal in self.diagnostics:
            return
        probabilities = full_probabilities.detach().float()[valid_mask]
        selected = weights.detach().float()[valid_mask]
        selected_indices = indices.detach()[valid_mask]
        gathered = probabilities.gather(-1, selected_indices)
        renormalized = gathered / gathered.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        raw_error = float((selected - gathered).abs().max().item())
        normalized_error = float((selected - renormalized).abs().max().item())
        best_error = min(raw_error, normalized_error)
        if best_error > self.validation_tolerance:
            raise RuntimeError(
                f"Selected router weights in {spec.block_name} do not match either full-softmax "
                f"top-k values or their top-k renormalization (max error {best_error:.4g})"
            )
        sums = probabilities.sum(dim=-1)
        if float((sums - 1.0).abs().max().item()) > self.validation_tolerance:
            raise RuntimeError(f"Router probabilities in {spec.block_name} do not sum to one")
        self.diagnostics[spec.ordinal] = {
            "router_probability_sum_mean": float(sums.mean().item()),
            "router_probability_sum_min": float(sums.min().item()),
            "router_probability_sum_max": float(sums.max().item()),
            "selected_weight_sum_mean": float(selected.sum(dim=-1).mean().item()),
            "selected_weight_reference": (
                "full_softmax" if raw_error <= normalized_error else "renormalized_topk"
            ),
            "selected_weight_max_error": best_error,
            "router_token_rows": int(indices.shape[0]),
            "valid_router_token_rows": int(valid_mask.sum().item()),
            "router_output_weights_available": True,
        }

    def _token_mapping(
        self, router_rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self._require_batch()
        mask = batch.attention_mask.to(device=device, dtype=torch.bool).reshape(-1)
        local = torch.arange(batch.batch_size, device=device).repeat_interleave(
            batch.sequence_length
        )
        if router_rows == mask.numel():
            return mask, local
        valid_count = int(mask.sum().item())
        if router_rows == valid_count:
            return torch.ones(router_rows, dtype=torch.bool, device=device), local[mask]
        raise RuntimeError(
            f"Router produced {router_rows} rows for a padded batch with {mask.numel()} total "
            f"and {valid_count} valid tokens"
        )

    @staticmethod
    def _scatter_routes(
        expert_indices: torch.Tensor,
        values: torch.Tensor,
        local_example_for_row: torch.Tensor,
        batch_size: int,
        num_experts: int,
    ) -> np.ndarray:
        if expert_indices.shape != values.shape:
            raise ValueError("Expert indices and route values must have the same shape")
        top_k = expert_indices.shape[1]
        local = local_example_for_row[:, None].expand(-1, top_k)
        flat_locations = (local * num_experts + expert_indices).reshape(-1).long()
        output = torch.zeros(
            batch_size * num_experts, dtype=torch.float32, device=expert_indices.device
        )
        output.scatter_add_(0, flat_locations, values.reshape(-1).float())
        return output.reshape(batch_size, num_experts).cpu().numpy()

    def _validate_completed_batch(self) -> None:
        batch = self._require_batch()
        expected = set(range(len(self.layer_specs)))
        if set(batch.routing) != expected:
            missing = expected - set(batch.routing)
            raise RuntimeError(f"Missing router hooks for MoE layer ordinals {sorted(missing)}")
        if batch.contribution_layers != expected:
            missing = expected - batch.contribution_layers
            raise RuntimeError(f"Missing contribution hooks for MoE layer ordinals {sorted(missing)}")
        if self.compute_gradient_attribution and batch.gradient_layers != expected:
            missing = expected - batch.gradient_layers
            raise RuntimeError(
                f"Gradient attribution was not finalized for layer ordinals {sorted(missing)}"
            )
        for spec in self.layer_specs:
            expected_assignments = (
                self.statistics.token_counts[batch.example_indices].astype(np.uint64) * spec.top_k
            )
            actual = self.statistics.routing_counts[
                batch.example_indices, spec.ordinal, :
            ].sum(axis=-1, dtype=np.uint64)
            if not np.array_equal(actual, expected_assignments):
                raise RuntimeError(
                    f"Routing assignment count mismatch in {spec.block_name}: "
                    f"expected {expected_assignments.tolist()}, got {actual.tolist()}"
                )

    def _require_batch(self) -> BatchState:
        if self._batch is None:
            raise RuntimeError("A model hook fired outside an active instrumentation batch")
        return self._batch


def _extract_routing(
    output: Any, router: nn.Module, spec: MoeLayerSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    tensors = _flatten_tensors(output)
    integer_tensors = [
        tensor
        for tensor in tensors
        if tensor.dtype
        in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        and tensor.ndim >= 1
    ]
    indices = next(
        (
            tensor
            for tensor in integer_tensors
            if tensor.shape[-1] == spec.top_k
        ),
        None,
    )
    float_tensors = [tensor for tensor in tensors if tensor.is_floating_point()]
    full = next(
        (tensor for tensor in float_tensors if tensor.ndim >= 2 and tensor.shape[-1] == spec.num_experts),
        None,
    )
    weights = None
    if indices is not None:
        weights = next(
            (
                tensor
                for tensor in float_tensors
                if tuple(tensor.shape) == tuple(indices.shape) and tensor is not full
            ),
            None,
        )
    if full is None:
        raise RuntimeError(
            f"Router {spec.router_name} did not expose logits/probabilities with "
            f"{spec.num_experts} experts"
        )
    full = full.reshape(-1, spec.num_experts)
    full_float = full.float()
    looks_like_probabilities = bool((full_float >= 0).all()) and float(
        (full_float.sum(dim=-1) - 1.0).abs().max().item()
    ) < 2e-3
    probabilities = full_float if looks_like_probabilities else F.softmax(full_float, dim=-1)
    if indices is None:
        derived_weights, indices = torch.topk(probabilities, spec.top_k, dim=-1)
        if _normalizes_topk(router, spec.block):
            derived_weights = derived_weights / derived_weights.sum(dim=-1, keepdim=True)
        return indices, derived_weights.to(full.dtype), probabilities, False
    indices = (
        indices.long()
        if indices.ndim == 2
        else indices.reshape(-1, spec.top_k).long()
    )
    if weights is None:
        derived_weights = probabilities.gather(-1, indices)
        if _normalizes_topk(router, spec.block):
            derived_weights = derived_weights / derived_weights.sum(dim=-1, keepdim=True)
        return indices, derived_weights.to(full.dtype), probabilities, False
    routed_weights = (
        weights
        if weights.ndim == 2
        else weights.reshape(-1, spec.top_k)
    )
    return indices, routed_weights, probabilities, True


def _normalizes_topk(router: nn.Module, block: nn.Module) -> bool:
    for owner in (router, block, getattr(block, "config", None)):
        if owner is not None and hasattr(owner, "norm_topk_prob"):
            return bool(getattr(owner, "norm_topk_prob"))
    return False


def _run_expert(spec: MoeLayerSpec, expert_id: int, hidden: torch.Tensor) -> torch.Tensor:
    experts = spec.experts
    if spec.contribution_backend == "tensorized_gate_up":
        gate_up = F.linear(hidden, experts.gate_up_proj[expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        activated = _activation(experts, gate) * up
        return F.linear(activated, experts.down_proj[expert_id])
    if spec.contribution_backend == "tensorized_separate":
        gate = F.linear(hidden, experts.gate_proj[expert_id])
        up = F.linear(hidden, experts.up_proj[expert_id])
        return F.linear(_activation(experts, gate) * up, experts.down_proj[expert_id])
    if spec.contribution_backend == "module_list":
        module = _module_at(experts, expert_id)
        return _tensor_output(module(hidden))
    if spec.contribution_backend == "nested_module_list":
        module = _module_at(experts.experts, expert_id)
        return _tensor_output(module(hidden))
    raise RuntimeError(f"Unsupported contribution backend {spec.contribution_backend}")


def _activation(experts: nn.Module, value: torch.Tensor) -> torch.Tensor:
    activation = getattr(experts, "act_fn", None)
    if activation is None:
        activation = getattr(experts, "activation_fn", None)
    if not callable(activation):
        raise RuntimeError(
            f"Tensorized expert container {experts.__class__.__name__} exposes no activation"
        )
    return activation(value)


def _module_at(container: nn.Module, expert_id: int) -> nn.Module:
    if isinstance(container, nn.ModuleDict):
        key = str(expert_id)
        if key in container:
            return container[key]
        return list(container.values())[expert_id]
    return container[expert_id]  # type: ignore[index]


def _tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    tensors = _flatten_tensors(output)
    if not tensors:
        raise RuntimeError("Expert module returned no tensor")
    return tensors[0]


def _find_hidden_tensor(
    args: tuple[Any, ...], kwargs: dict[str, Any], top_k: int
) -> torch.Tensor:
    tensors = _flatten_tensors((args, kwargs))
    candidates = [
        tensor
        for tensor in tensors
        if tensor.is_floating_point() and tensor.ndim >= 2 and tensor.shape[-1] != top_k
    ]
    if not candidates:
        raise RuntimeError("Could not identify hidden states in MoE hook inputs")
    return max(candidates, key=lambda tensor: (tensor.shape[-1], tensor.numel()))


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result: list[torch.Tensor] = []
        for item in value.values():
            result.extend(_flatten_tensors(item))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_flatten_tensors(item))
        return result
    return []


# Public replay helpers.  The collection hook and the Stage-2 activation replay
# must use one mathematical implementation; these narrow wrappers keep that
# contract explicit without exposing the hook internals themselves.
def extract_routing(
    output: Any, router: nn.Module, spec: MoeLayerSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    return _extract_routing(output, router, spec)


def run_expert(spec: MoeLayerSpec, expert_id: int, hidden: torch.Tensor) -> torch.Tensor:
    return _run_expert(spec, expert_id, hidden)


def find_hidden_tensor(
    args: tuple[Any, ...], kwargs: dict[str, Any], top_k: int
) -> torch.Tensor:
    return _find_hidden_tensor(args, kwargs, top_k)


def tensor_output(output: Any) -> torch.Tensor:
    return _tensor_output(output)
