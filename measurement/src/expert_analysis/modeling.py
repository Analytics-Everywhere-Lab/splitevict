from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .hardware import RuntimeDevice


@dataclass
class MoeLayerSpec:
    ordinal: int
    model_layer_index: int
    block_name: str
    router_name: str
    experts_name: str
    num_experts: int
    top_k: int
    contribution_backend: str
    capture_point: str
    block: nn.Module
    router: nn.Module
    experts: nn.Module

    def metadata(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "model_layer_index": self.model_layer_index,
            "block_name": self.block_name,
            "router_name": self.router_name,
            "experts_name": self.experts_name,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "contribution_backend": self.contribution_backend,
            "capture_point": self.capture_point,
        }


@dataclass
class ModelBundle:
    model: nn.Module
    backbone: nn.Module
    tokenizer: Any
    checkpoint: str
    requested_revision: str | None
    resolved_revision: str | None
    runtime: RuntimeDevice


def load_model_and_tokenizer(
    checkpoint: str,
    runtime: RuntimeDevice,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    attn_implementation: str | None = None,
) -> ModelBundle:
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required; install requirements.txt") from exc

    common: dict[str, Any] = {
        "revision": revision,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    common = {key: value for key, value in common.items() if value is not None}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **common)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = dict(common)
    transformers_major = int(transformers.__version__.split(".", 1)[0])
    if transformers_major >= 5:
        model_kwargs["dtype"] = runtime.dtype
    else:
        model_kwargs["torch_dtype"] = runtime.dtype
    # `low_cpu_mem_usage` avoids holding two copies of a ~7B-parameter checkpoint.
    model_kwargs["low_cpu_mem_usage"] = True
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    if runtime.device.type != "cpu":
        # A single-device map streams checkpoint shards onto CUDA/MPS from a meta
        # initialization, avoiding a second full CPU copy during model.to(device).
        model_kwargs["device_map"] = {"": str(runtime.device)}
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)

    model.eval()
    if runtime.device.type == "cpu":
        model.to(runtime.device)
    else:
        first_device = next(model.parameters()).device
        if first_device.type != runtime.device.type:
            raise RuntimeError(
                f"Direct checkpoint placement requested {runtime.device}, but model "
                f"parameters are on {first_device}"
            )
    backbone = find_causal_lm_backbone(model)
    config = getattr(model, "config", None)
    resolved_revision = getattr(config, "_commit_hash", None)
    return ModelBundle(
        model=model,
        backbone=backbone,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        runtime=runtime,
    )


def find_causal_lm_backbone(model: nn.Module) -> nn.Module:
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        candidate = getattr(model, prefix)
        if isinstance(candidate, nn.Module) and candidate is not model:
            return candidate
    for attribute in ("model", "transformer", "decoder"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, nn.Module) and candidate is not model:
            return candidate
    raise RuntimeError(
        f"Could not identify the causal-LM backbone for {model.__class__.__name__}"
    )


def discover_moe_layers(model: nn.Module) -> list[MoeLayerSpec]:
    """Discover MoE blocks structurally instead of assuming OLMoE module paths."""
    module_names = {id(module): name for name, module in model.named_modules()}
    candidates: list[tuple[str, nn.Module, nn.Module, nn.Module]] = []
    for block_name, block in model.named_modules():
        router = _find_router(block)
        experts = _find_expert_container(block)
        if router is None or experts is None or router is experts:
            continue
        num_experts = _infer_num_experts(block, router, experts)
        if num_experts is None or num_experts < 2:
            continue
        candidates.append((block_name, block, router, experts))

    # Keep the smallest structural blocks: an outer module can otherwise rediscover
    # a nested MoE through permissive attribute names in custom implementations.
    block_ids = {id(item[1]) for item in candidates}
    filtered = []
    for item in candidates:
        _, block, _, _ = item
        nested_candidate = any(id(child) in block_ids for child in block.children())
        if not nested_candidate:
            filtered.append(item)
    candidates = filtered
    if not candidates:
        raise RuntimeError(
            "No MoE layers were discovered. Expected a block containing a router/gate "
            "and an expert container. Refusing to collect invalid statistics."
        )

    specs: list[MoeLayerSpec] = []
    for ordinal, (block_name, block, router, experts) in enumerate(candidates):
        num_experts = _infer_num_experts(block, router, experts)
        assert num_experts is not None
        top_k = _infer_top_k(model, block, router)
        backend = _infer_contribution_backend(experts)
        if backend == "unsupported":
            raise RuntimeError(
                f"MoE layer {block_name!r} has unsupported expert container "
                f"{experts.__class__.__name__}. Exact functional contribution cannot be "
                "reconstructed; refusing to silently substitute routing utilization."
            )
        capture_point = (
            "experts_pre"
            if backend.startswith("tensorized") and _has_custom_forward(experts)
            else "block_post"
        )
        specs.append(
            MoeLayerSpec(
                ordinal=ordinal,
                model_layer_index=_parse_layer_index(block_name, ordinal),
                block_name=block_name,
                router_name=module_names.get(id(router), f"{block_name}.<router>"),
                experts_name=module_names.get(id(experts), f"{block_name}.<experts>"),
                num_experts=num_experts,
                top_k=top_k,
                contribution_backend=backend,
                capture_point=capture_point,
                block=block,
                router=router,
                experts=experts,
            )
        )

    specs.sort(key=lambda spec: (spec.model_layer_index, spec.block_name))
    for ordinal, spec in enumerate(specs):
        spec.ordinal = ordinal
    expert_counts = {spec.num_experts for spec in specs}
    if len(expert_counts) != 1:
        raise RuntimeError(
            "This experiment currently requires the same expert count in every MoE layer; "
            f"discovered {sorted(expert_counts)}"
        )
    return specs


def architecture_metadata(model: nn.Module, specs: list[MoeLayerSpec]) -> dict[str, Any]:
    config = getattr(model, "config", None)
    config_dict = config.to_dict() if config is not None and hasattr(config, "to_dict") else {}
    return {
        "model_class": model.__class__.__name__,
        "config_model_type": config_dict.get("model_type"),
        "num_moe_layers": len(specs),
        "num_experts": specs[0].num_experts,
        "top_k": sorted({spec.top_k for spec in specs}),
        "layers": [spec.metadata() for spec in specs],
        "config": {
            key: config_dict.get(key)
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_experts",
                "num_local_experts",
                "num_experts_per_tok",
                "norm_topk_prob",
                "vocab_size",
                "max_position_embeddings",
            )
            if key in config_dict
        },
    }


def _find_router(block: nn.Module) -> nn.Module | None:
    for attribute in ("gate", "router", "routing", "router_layer", "gate_module"):
        candidate = getattr(block, attribute, None)
        if isinstance(candidate, nn.Module):
            return candidate
    return None


def _find_expert_container(block: nn.Module) -> nn.Module | None:
    for attribute in ("experts", "expert", "expert_layer", "expert_modules"):
        candidate = getattr(block, attribute, None)
        if isinstance(candidate, nn.Module):
            return candidate
    return None


def _infer_num_experts(block: nn.Module, router: nn.Module, experts: nn.Module) -> int | None:
    for owner in (router, experts, block, getattr(block, "config", None)):
        if owner is None:
            continue
        for attribute in ("num_experts", "num_local_experts", "n_experts"):
            value = getattr(owner, attribute, None)
            if isinstance(value, int) and value > 1:
                return value
    if isinstance(experts, (nn.ModuleList, nn.ModuleDict)):
        return len(experts)
    for attribute in ("gate_up_proj", "gate_proj", "down_proj"):
        value = getattr(experts, attribute, None)
        if isinstance(value, torch.Tensor) and value.ndim == 3:
            return int(value.shape[0])
    if isinstance(router, nn.Linear):
        return int(router.out_features)
    weight = getattr(router, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return None


def _infer_top_k(model: nn.Module, block: nn.Module, router: nn.Module) -> int:
    for owner in (router, block, getattr(block, "config", None), getattr(model, "config", None)):
        if owner is None:
            continue
        for attribute in ("top_k", "num_experts_per_tok", "topk"):
            value = getattr(owner, attribute, None)
            if isinstance(value, int) and value >= 1:
                return value
    raise RuntimeError(
        f"Could not infer top-k routing for MoE block {block.__class__.__name__}"
    )


def _infer_contribution_backend(experts: nn.Module) -> str:
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if isinstance(gate_up, torch.Tensor) and gate_up.ndim == 3 and isinstance(down, torch.Tensor):
        return "tensorized_gate_up"
    gate = getattr(experts, "gate_proj", None)
    up = getattr(experts, "up_proj", None)
    if all(isinstance(item, torch.Tensor) and item.ndim == 3 for item in (gate, up, down)):
        return "tensorized_separate"
    if isinstance(experts, (nn.ModuleList, nn.ModuleDict)):
        return "module_list"
    nested = getattr(experts, "experts", None)
    if isinstance(nested, (nn.ModuleList, nn.ModuleDict)):
        return "nested_module_list"
    return "unsupported"


def _has_custom_forward(module: nn.Module) -> bool:
    return module.__class__.forward is not nn.Module.forward


def _parse_layer_index(name: str, fallback: int) -> int:
    matches = re.findall(r"(?:layers|blocks|h|layer)\.(\d+)", name)
    if matches:
        return int(matches[-1])
    all_numbers = re.findall(r"(?:^|\.)(\d+)(?:\.|$)", name)
    return int(all_numbers[-1]) if all_numbers else fallback


def supports_kwarg(module: nn.Module, name: str) -> bool:
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
