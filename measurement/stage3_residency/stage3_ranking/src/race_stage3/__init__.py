"""RACE Stage 3 — learned causal future-reuse ranking.

Stage 3 keeps the frozen Stage 1 eviction mechanism and the frozen Stage 0 cache
semantics unchanged. It replaces the retention score with one calibration-fitted
linear ranking function per cache capacity over raw-scale causal features, acting on
the Stage 2 diagnostic that percentile rank normalization destroys the magnitude
information the Stage 1 hybrid exploits.
"""

from __future__ import annotations

STAGE3_SCHEMA_VERSION = "race_stage3_v1"
CAP = 33
