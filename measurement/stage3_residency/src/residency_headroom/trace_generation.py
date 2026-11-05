from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from expert_analysis.datasets import load_domain_examples
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.hooks import extract_routing
from expert_analysis.modeling import (
    MoeLayerSpec,
    architecture_metadata,
    discover_moe_layers,
    load_model_and_tokenizer,
)

from . import TRACE_SCHEMA_VERSION
from .common import (
    atomic_write_json,
    environment_record,
    resolve_git_commit,
    sha256_json,
    sha256_source_bundle,
    utc_now,
)
from .trace import RoutingTrace


@dataclass(frozen=True)
class CapturedLayerRequest:
    layer_index: int
    expert_ids: np.ndarray
    router_weights: np.ndarray


class DecodeRoutingCapture:
    """Removable hooks that record one atomic request per active decode layer."""

    def __init__(self, specs: Sequence[MoeLayerSpec]) -> None:
        self.specs = tuple(specs)
        self._handles: list[Any] = []
        self._active = False
        self._records: dict[int, CapturedLayerRequest] = {}

    def __enter__(self) -> "DecodeRoutingCapture":
        if self._handles:
            raise RuntimeError("Decode capture cannot be entered twice")
        for spec in self.specs:
            self._handles.append(
                spec.router.register_forward_hook(self._hook(spec), with_kwargs=True)
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._active = False
        self._records.clear()

    def begin_token(self) -> None:
        if not self._handles or self._active:
            raise RuntimeError("Decode capture token lifecycle is invalid")
        self._records = {}
        self._active = True

    def finish_token(self) -> tuple[CapturedLayerRequest, ...]:
        if not self._active:
            raise RuntimeError("No active decode token")
        self._active = False
        expected = {spec.ordinal for spec in self.specs}
        if set(self._records) != expected:
            missing = expected - set(self._records)
            extra = set(self._records) - expected
            raise RuntimeError(
                f"Decode token routing hooks incomplete: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        return tuple(self._records[spec.ordinal] for spec in self.specs)

    def _hook(self, spec: MoeLayerSpec):
        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del args, kwargs
            if not self._active:
                return
            if spec.ordinal in self._records:
                raise RuntimeError(f"Router {spec.router_name} ran twice for one decode token")
            indices, weights, _full, _from_output = extract_routing(output, module, spec)
            if indices.shape != (1, spec.top_k) or weights.shape != indices.shape:
                raise RuntimeError(
                    f"Decode hook expected one top-{spec.top_k} row in {spec.block_name}; "
                    f"got IDs {tuple(indices.shape)} and weights {tuple(weights.shape)}"
                )
            ids = indices.detach().to("cpu", dtype=torch.int64).numpy()[0]
            selected_weights = weights.detach().to("cpu", dtype=torch.float32).numpy()[0]
            if len(np.unique(ids)) != spec.top_k:
                raise RuntimeError(f"Router {spec.router_name} selected a duplicate expert")
            self._records[spec.ordinal] = CapturedLayerRequest(
                layer_index=spec.model_layer_index,
                expert_ids=ids,
                router_weights=selected_weights,
            )

        return hook


def generate_trace(
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    cache_dir: str | None = None,
    dataset_cache_dir: str | None = None,
    local_files_only: bool = False,
) -> RoutingTrace:
    """Generate/resume deterministic decode chunks and combine a validated trace."""

    _validate_generation_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "routing_trace.npz"
    config_hash = sha256_json(config)
    repository_root = Path(__file__).resolve().parents[3]
    source_bundle_hash = sha256_source_bundle(repository_root / "stage3_residency")
    runtime_commit = resolve_git_commit(repository_root, fallback=str(config["source_commit"]))
    if trace_path.exists():
        trace = RoutingTrace.load(trace_path)
        if trace.metadata["hash_basis"]["config_hash"] != config_hash:
            raise RuntimeError("Existing routing trace was generated from a different config")
        return trace

    started = time.monotonic()
    set_reproducible_seed(int(config["seed"]), deterministic=True)
    runtime = resolve_runtime(str(config["device"]), str(config["dtype"]))
    bundle = load_model_and_tokenizer(
        checkpoint=str(config["model"]),
        runtime=runtime,
        revision=str(config["model_revision"]),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        attn_implementation=str(config.get("attn_implementation", "eager")),
    )
    if bundle.resolved_revision != config["model_revision"]:
        raise RuntimeError(
            f"Resolved model revision {bundle.resolved_revision!r} differs from pinned "
            f"{config['model_revision']!r}"
        )
    specs = discover_moe_layers(bundle.model)
    architecture = architecture_metadata(bundle.model, specs)
    _validate_architecture(architecture, config["expected_architecture"])
    expert_bytes = _expert_bytes_by_layer(specs)

    dataset_metadata: dict[str, Any] = {}
    expected_chunks: list[Path] = []
    sequence_id = 0
    progress_path = output_dir / "trace_generation_progress.json"
    with DecodeRoutingCapture(specs) as capture:
        for domain_id, domain in enumerate(config["domains"]):
            examples = load_domain_examples(
                domain=domain,
                num_examples=int(config["num_prompts_per_domain"]),
                seed=int(config["seed"]),
                cache_dir=dataset_cache_dir or cache_dir,
                revision=config["dataset_revisions"][domain],
                include_answers=bool(config["include_reference_answers"]),
                allow_substitution=bool(config["allow_dataset_substitution"]),
                format_style="legacy",
            )
            if len(examples.texts) != int(config["num_prompts_per_domain"]):
                raise RuntimeError(
                    f"Dataset {domain} yielded {len(examples.texts)} prompts; "
                    f"expected {config['num_prompts_per_domain']}"
                )
            if examples.metadata.get("substituted"):
                raise RuntimeError(f"Dataset substitution occurred for {domain}")
            resolved_dataset = examples.metadata.get("resolved_revision")
            if resolved_dataset and resolved_dataset != config["dataset_revisions"][domain]:
                raise RuntimeError(f"Resolved dataset revision changed for {domain}")
            dataset_metadata[domain] = examples.metadata
            prompt_ids = examples.metadata["selected_example_ids"]
            for prompt_index, (prompt_id, prompt) in enumerate(zip(prompt_ids, examples.texts)):
                chunk_path = (
                    output_dir
                    / "chunks"
                    / domain
                    / f"sequence_{sequence_id:04d}.npz"
                )
                expected_chunks.append(chunk_path)
                if chunk_path.exists():
                    chunk = RoutingTrace.load(chunk_path)
                    item = chunk.metadata["sequences"][0]
                    if (
                        chunk.metadata["hash_basis"]["config_hash"] != config_hash
                        or chunk.metadata["hash_basis"].get("stage3_source_bundle_hash")
                        != source_bundle_hash
                        or int(item["sequence_id"]) != sequence_id
                        or str(item["prompt_id"]) != str(prompt_id)
                        or item["prompt_sha256"] != _text_hash(prompt)
                    ):
                        raise RuntimeError(f"Resume chunk {chunk_path} has incompatible identity")
                else:
                    chunk = _generate_one_sequence(
                        bundle,
                        specs,
                        capture,
                        config,
                        config_hash,
                        source_bundle_hash,
                        sequence_id=sequence_id,
                        domain=domain,
                        domain_id=domain_id,
                        prompt_index=prompt_index,
                        prompt_id=str(prompt_id),
                        prompt=prompt,
                        expert_bytes=expert_bytes,
                        architecture=architecture,
                    )
                    chunk.save(chunk_path)
                sequence_id += 1
                atomic_write_json(
                    progress_path,
                    {
                        "schema_version": "race_stage0_trace_progress_v1",
                        "config_hash": config_hash,
                        "completed_sequences": sequence_id,
                        "expected_sequences": len(config["domains"])
                        * int(config["num_prompts_per_domain"]),
                        "last_chunk": str(chunk_path.relative_to(output_dir)),
                        "elapsed_seconds": time.monotonic() - started,
                        "updated_at_utc": utc_now(),
                    },
                )

    trace = combine_chunks(
        expected_chunks,
        config=config,
        config_hash=config_hash,
        source_bundle_hash=source_bundle_hash,
        runtime_source_commit=runtime_commit,
        architecture=architecture,
        expert_bytes=expert_bytes,
        dataset_metadata=dataset_metadata,
        runtime_record={
            "device": str(runtime.device),
            "device_description": runtime.description,
            "dtype": str(runtime.dtype).replace("torch.", ""),
        },
        tokenizer_class=bundle.tokenizer.__class__.__name__,
        resolved_model_revision=bundle.resolved_revision,
        elapsed_seconds=time.monotonic() - started,
    )
    trace.save(trace_path)
    validation = trace.validate(verify_hash=True)
    manifest = {
        "schema_version": "race_stage0_trace_manifest_v1",
        "trace_path": trace_path.name,
        "trace_hash": trace.trace_hash,
        "config_hash": config_hash,
        "source_commit": config["source_commit"],
        "runtime_source_commit": runtime_commit,
        "stage3_source_bundle_hash": source_bundle_hash,
        "model": config["model"],
        "model_revision": bundle.resolved_revision,
        "precision": str(runtime.dtype).replace("torch.", ""),
        "tokenizer": bundle.tokenizer.__class__.__name__,
        "generation_settings": {
            "max_new_tokens": config["max_new_tokens"],
            "do_sample": False,
            "stop_on_eos": config["stop_on_eos"],
            "seed": config["seed"],
        },
        "validation": validation,
        "expert_utilization": trace.metadata["expert_utilization"],
        "environment": trace.metadata["environment"],
        "elapsed_seconds": time.monotonic() - started,
        "created_at_utc": utc_now(),
    }
    atomic_write_json(output_dir / "trace_manifest.json", manifest)
    return RoutingTrace.load(trace_path)


def _generate_one_sequence(
    bundle: Any,
    specs: Sequence[MoeLayerSpec],
    capture: DecodeRoutingCapture,
    config: Mapping[str, Any],
    config_hash: str,
    source_bundle_hash: str,
    *,
    sequence_id: int,
    domain: str,
    domain_id: int,
    prompt_index: int,
    prompt_id: str,
    prompt: str,
    expert_bytes: np.ndarray,
    architecture: Mapping[str, Any],
) -> RoutingTrace:
    encoded = bundle.tokenizer(
        prompt,
        truncation=True,
        max_length=int(config["max_prompt_tokens"]),
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(bundle.runtime.device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
        bundle.runtime.device
    )
    if input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise RuntimeError("Decode trace generation requires one non-empty prompt")
    prompt_length = int(attention_mask.sum().item())
    layer_values: list[int] = []
    expert_values: list[np.ndarray] = []
    weight_values: list[np.ndarray] = []
    generated_indices: list[int] = []
    token_values: list[int] = []
    generated_token_ids: list[int] = []
    eos_ids = _eos_ids(bundle.tokenizer)

    with torch.inference_mode():
        outputs = bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past = getattr(outputs, "past_key_values", None)
        if past is None:
            raise RuntimeError("Model returned no KV cache for deterministic decode")
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Causal LM returned no logits")
        for generated_index in range(int(config["max_new_tokens"])):
            next_token = torch.argmax(logits[:, -1, :].float(), dim=-1)
            token_id = int(next_token.item())
            generated_token_ids.append(token_id)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (1, 1), dtype=attention_mask.dtype, device=attention_mask.device
                    ),
                ),
                dim=1,
            )
            capture.begin_token()
            outputs = bundle.model(
                input_ids=next_token[:, None],
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            records = capture.finish_token()
            for record in records:
                layer_values.append(record.layer_index)
                expert_values.append(record.expert_ids.astype(np.int16, copy=False))
                weight_values.append(record.router_weights.astype(np.float32, copy=False))
                generated_indices.append(generated_index)
                token_values.append(token_id)
            past = getattr(outputs, "past_key_values", None)
            logits = getattr(outputs, "logits", None)
            if past is None or not isinstance(logits, torch.Tensor):
                raise RuntimeError("Decode step returned incomplete cache/logits")
            if bool(config["stop_on_eos"]) and token_id in eos_ids:
                break

    generation_length = len(generated_token_ids)
    events = generation_length * len(specs)
    if len(layer_values) != events or generation_length < 1:
        raise RuntimeError("Decode event accounting failed")
    sequence = {
        "sequence_id": sequence_id,
        "domain": domain,
        "domain_id": domain_id,
        "prompt_index": prompt_index,
        "prompt_id": prompt_id,
        "prompt_sha256": _text_hash(prompt),
        "prompt_length": prompt_length,
        "generation_length": generation_length,
        "generated_token_ids": generated_token_ids,
        "stopped_on_eos": generated_token_ids[-1] in eos_ids,
    }
    arrays = {
        "event_index": np.arange(events, dtype=np.int64),
        "sequence_id": np.full(events, sequence_id, dtype=np.int32),
        "domain_id": np.full(events, domain_id, dtype=np.int8),
        "prompt_index": np.full(events, prompt_index, dtype=np.int32),
        "generated_token_index": np.asarray(generated_indices, dtype=np.int32),
        "layer_index": np.asarray(layer_values, dtype=np.int16),
        "requested_expert_ids": np.stack(expert_values).astype(np.int16, copy=False),
        "router_weights": np.stack(weight_values).astype(np.float32, copy=False),
        "token_id": np.asarray(token_values, dtype=np.int32),
        "prompt_length": np.full(events, prompt_length, dtype=np.int32),
        "generation_length": np.full(events, generation_length, dtype=np.int32),
    }
    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "hash_basis": {
            "config_hash": config_hash,
            "stage3_source_bundle_hash": source_bundle_hash,
            "sequence_id": sequence_id,
            "prompt_sha256": sequence["prompt_sha256"],
            "model_revision": config["model_revision"],
        },
        "domains": list(config["domains"]),
        "layer_indices": [spec.model_layer_index for spec in specs],
        "num_experts": int(architecture["num_experts"]),
        "top_k": int(specs[0].top_k),
        "expert_bytes_by_layer": expert_bytes.tolist(),
        "sequences": [sequence],
    }
    return RoutingTrace.from_mapping(arrays, metadata, validate=True)


def combine_chunks(
    chunk_paths: Sequence[Path],
    *,
    config: Mapping[str, Any],
    config_hash: str,
    source_bundle_hash: str,
    runtime_source_commit: str | None,
    architecture: Mapping[str, Any],
    expert_bytes: np.ndarray,
    dataset_metadata: Mapping[str, Any],
    runtime_record: Mapping[str, Any],
    tokenizer_class: str,
    resolved_model_revision: str | None,
    elapsed_seconds: float,
) -> RoutingTrace:
    chunks = [RoutingTrace.load(path) for path in chunk_paths]
    if not chunks:
        raise ValueError("No decode chunks were produced")
    arrays: dict[str, np.ndarray] = {}
    for name in chunks[0].arrays():
        if name == "event_index":
            continue
        arrays[name] = np.concatenate([getattr(chunk, name) for chunk in chunks], axis=0)
    arrays["event_index"] = np.arange(len(arrays["sequence_id"]), dtype=np.int64)
    sequences = [chunk.metadata["sequences"][0] for chunk in chunks]
    utilization = _utilization_summary(
        arrays["layer_index"],
        arrays["requested_expert_ids"],
        [spec["model_layer_index"] for spec in architecture["layers"]],
        int(architecture["num_experts"]),
    )
    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "hash_basis": {
            "config_hash": config_hash,
            "source_commit": config["source_commit"],
            "runtime_source_commit": runtime_source_commit,
            "stage3_source_bundle_hash": source_bundle_hash,
            "model": config["model"],
            "model_revision": resolved_model_revision,
            "dataset_revisions": config["dataset_revisions"],
            "generation": config["generation"],
            "seed": config["seed"],
        },
        "trace_source": "real deterministic OLMoE decode-token routing",
        "decode_token_definition": (
            "A generated token is greedily selected from prior logits, then fed through "
            "the cached model; its one atomic top-k request per MoE layer is recorded."
        ),
        "model": config["model"],
        "source_commit": config["source_commit"],
        "runtime_source_commit": runtime_source_commit,
        "stage3_source_bundle_hash": source_bundle_hash,
        "requested_model_revision": config["model_revision"],
        "resolved_model_revision": resolved_model_revision,
        "precision": runtime_record["dtype"],
        "tokenizer": tokenizer_class,
        "generation_settings": {
            **dict(config["generation"]),
            "max_new_tokens": config["max_new_tokens"],
            "stop_on_eos": config["stop_on_eos"],
            "max_prompt_tokens": config["max_prompt_tokens"],
        },
        "seed": config["seed"],
        "domains": list(config["domains"]),
        "domain_to_id": {domain: index for index, domain in enumerate(config["domains"])},
        "datasets": dict(dataset_metadata),
        "layer_indices": [item["model_layer_index"] for item in architecture["layers"]],
        "num_experts": int(architecture["num_experts"]),
        "top_k": int(architecture["top_k"][0]),
        "architecture": dict(architecture),
        "expert_bytes_by_layer": expert_bytes.tolist(),
        "all_experts_equal_size": bool(np.all(expert_bytes == expert_bytes.flat[0])),
        "sequences": sequences,
        "expert_utilization": utilization,
        "runtime": dict(runtime_record),
        "environment": environment_record(),
        "elapsed_seconds": elapsed_seconds,
        "created_at_utc": utc_now(),
    }
    trace = RoutingTrace.from_mapping(arrays, metadata, validate=True)
    metadata["trace_hash"] = trace.logical_hash()
    return RoutingTrace.from_mapping(arrays, metadata, validate=True)


def _expert_bytes_by_layer(specs: Sequence[MoeLayerSpec]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for spec in specs:
        if spec.contribution_backend in {"tensorized_gate_up", "tensorized_separate"}:
            values = np.zeros(spec.num_experts, dtype=np.int64)
            found = 0
            for _name, parameter in spec.experts.named_parameters(recurse=True):
                if parameter.ndim >= 1 and parameter.shape[0] == spec.num_experts:
                    per_expert = parameter[0].numel() * parameter.element_size()
                    values += int(per_expert)
                    found += 1
            if found == 0:
                raise RuntimeError(f"No expert-indexed parameters found in {spec.experts_name}")
        else:
            container = getattr(spec.experts, "experts", spec.experts)
            modules = list(container.values()) if isinstance(container, nn.ModuleDict) else list(container)
            if len(modules) != spec.num_experts:
                raise RuntimeError(f"Could not enumerate experts in {spec.experts_name}")
            values = np.asarray(
                [
                    sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
                    for module in modules
                ],
                dtype=np.int64,
            )
        if np.any(values <= 0):
            raise RuntimeError(f"Non-positive expert parameter size in {spec.experts_name}")
        rows.append(values)
    return np.stack(rows)


def _utilization_summary(
    layers: np.ndarray,
    requests: np.ndarray,
    layer_indices: Sequence[int],
    num_experts: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for layer in layer_indices:
        selected = requests[layers == layer].reshape(-1)
        counts = np.bincount(selected, minlength=num_experts)
        total = int(counts.sum())
        frequency = counts / total
        positive = frequency[frequency > 0]
        entropy = float(-(positive * np.log2(positive)).sum())
        output[str(layer)] = {
            "events": int(np.count_nonzero(layers == layer)),
            "assignments": total,
            "active_experts": int(np.count_nonzero(counts)),
            "minimum_assignments": int(counts.min()),
            "maximum_assignments": int(counts.max()),
            "entropy_bits": entropy,
            "normalized_entropy": entropy / np.log2(num_experts),
            "top_10_traffic_share": float(np.sort(frequency)[-10:].sum()),
            "top_20_traffic_share": float(np.sort(frequency)[-20:].sum()),
        }
    return output


def _validate_generation_config(config: Mapping[str, Any]) -> None:
    required = (
        "source_commit",
        "model",
        "model_revision",
        "device",
        "dtype",
        "domains",
        "dataset_revisions",
        "num_prompts_per_domain",
        "max_prompt_tokens",
        "max_new_tokens",
        "seed",
        "expected_architecture",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Trace configuration is missing {missing}")
    if config.get("generation", {}).get("do_sample") is not False:
        raise ValueError("Stage 0 trace generation is preregistered as deterministic greedy decode")
    if int(config["num_prompts_per_domain"]) < 1 or int(config["max_new_tokens"]) < 1:
        raise ValueError("Prompt and decode counts must be positive")
    if set(config["domains"]) != set(config["dataset_revisions"]):
        raise ValueError("Every domain must have exactly one pinned dataset revision")


def _validate_architecture(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    checks = {
        "num_moe_layers": int(actual["num_moe_layers"]),
        "num_experts": int(actual["num_experts"]),
        "top_k": int(actual["top_k"][0]) if len(actual["top_k"]) == 1 else actual["top_k"],
    }
    for name, value in checks.items():
        if value != expected[name]:
            raise RuntimeError(f"Architecture {name}={value!r}; expected {expected[name]!r}")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _eos_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, (tuple, list, set)):
        return set(map(int, value))
    return {int(value)}
