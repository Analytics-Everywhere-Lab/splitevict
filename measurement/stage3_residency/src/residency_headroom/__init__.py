"""RACE Stage 0 expert-residency oracle-headroom study.

This package contains only trace collection, cache simulation, exact/offline
comparators, and reporting.  It intentionally contains no online optimizer.
"""

TRACE_SCHEMA_VERSION = "olmoe_decode_atomic_routing_v1"
CONFIG_SCHEMA_VERSION = "race_stage0_config_v1"
RESULT_SCHEMA_VERSION = "race_stage0_results_v1"

__all__ = ["TRACE_SCHEMA_VERSION", "CONFIG_SCHEMA_VERSION", "RESULT_SCHEMA_VERSION"]
