"""RACE Stage 2 — adaptive multi-horizon future-reuse ranking.

Stage 2 reuses the frozen Stage 0 trace/oracle and the frozen Stage 1 cache and
eviction semantics without modification. It only changes how eligible eviction
candidates are *ranked*.
"""

from __future__ import annotations

STAGE2_SCHEMA_VERSION = "race_stage2_v1"
H_MAX = 32
"""Capped future-reuse horizon, in same-layer (same-cache) events."""

NOT_REUSED_WITHIN_HORIZON = H_MAX + 1
"""Capped distance value assigned to experts unused within the next H_MAX events."""
