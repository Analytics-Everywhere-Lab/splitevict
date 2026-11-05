from __future__ import annotations

import platform
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeDevice:
    device: torch.device
    dtype: torch.dtype
    description: str


def detect_device(requested: str = "auto") -> torch.device:
    """Choose CUDA, then MPS, then CPU, unless explicitly overridden."""
    requested = requested.lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if device.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but it is not available in this PyTorch build")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    choices = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if requested != "auto":
        try:
            return choices[requested.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype {requested!r}") from exc
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        # BF16 is available on recent Apple Silicon/macOS combinations and better
        # matches the released checkpoint. Probe because older MPS stacks reject it.
        try:
            probe = torch.ones(1, device=device, dtype=torch.bfloat16)
            _ = probe + probe
            del probe
            return torch.bfloat16
        except (RuntimeError, TypeError):
            return torch.float16
    # The released OLMoE is roughly 7B total parameters. BF16 keeps a single-device
    # CPU run feasible on machines where FP32 would require about twice the memory.
    return torch.bfloat16


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple Silicon MPS ({platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def resolve_runtime(device: str = "auto", dtype: str = "auto") -> RuntimeDevice:
    resolved_device = detect_device(device)
    resolved_dtype = choose_dtype(resolved_device, dtype)
    return RuntimeDevice(resolved_device, resolved_dtype, describe_device(resolved_device))


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # warn_only avoids turning a diagnostic into a platform-specific failure when
        # a sparse/index_add kernel has no deterministic implementation.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
