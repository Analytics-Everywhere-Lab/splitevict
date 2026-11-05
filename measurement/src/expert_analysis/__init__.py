"""Domain-conditioned expert-importance analysis for sparse MoE language models."""

DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924"
DOMAINS = ("general", "math", "coding", "reasoning")
METRICS = ("routing_frequency", "gate_mass", "functional_contribution")

__all__ = ["DEFAULT_MODEL", "DOMAINS", "METRICS"]
