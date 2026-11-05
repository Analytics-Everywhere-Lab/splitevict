from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class DatasetCandidate:
    repository: str
    config: str | None
    split: str
    formatter: str


@dataclass
class DomainExamples:
    domain: str
    texts: list[str]
    metadata: dict[str, Any]


DATASET_CANDIDATES: dict[str, tuple[DatasetCandidate, ...]] = {
    "general": (
        DatasetCandidate("Salesforce/wikitext", "wikitext-103-raw-v1", "test", "wikitext"),
        DatasetCandidate("EleutherAI/lambada_openai", "plain_text", "test", "lambada"),
    ),
    "math": (
        DatasetCandidate("openai/gsm8k", "main", "test", "gsm8k"),
        DatasetCandidate("EleutherAI/hendrycks_math", "algebra", "test", "hendrycks_math"),
    ),
    "coding": (
        DatasetCandidate("google-research-datasets/mbpp", "full", "test", "mbpp"),
        DatasetCandidate("openai/openai_humaneval", None, "test", "humaneval"),
    ),
    "reasoning": (
        DatasetCandidate("allenai/ai2_arc", "ARC-Challenge", "test", "arc"),
        DatasetCandidate("allenai/openbookqa", "main", "test", "arc"),
    ),
}

DOMAIN_SEED_OFFSETS = {"general": 11, "math": 23, "coding": 37, "reasoning": 53}


def load_domain_examples(
    domain: str,
    num_examples: int,
    seed: int,
    cache_dir: str | None = None,
    revision: str | None = None,
    include_answers: bool = True,
    allow_substitution: bool = True,
    format_style: str = "legacy",
) -> DomainExamples:
    if domain not in DATASET_CANDIDATES:
        raise ValueError(f"Unknown domain {domain!r}; expected {sorted(DATASET_CANDIDATES)}")
    if num_examples < 1:
        raise ValueError("num_examples must be positive")
    if format_style not in {"legacy", "neutral_content"}:
        raise ValueError("format_style must be 'legacy' or 'neutral_content'")
    if format_style == "neutral_content" and include_answers:
        raise ValueError("neutral_content formatting does not include reference answers")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face `datasets` package is required for collection. "
            "Install requirements.txt before running a real experiment."
        ) from exc

    candidates = DATASET_CANDIDATES[domain]
    if not allow_substitution:
        candidates = candidates[:1]
    failures: list[dict[str, str]] = []
    for candidate_index, candidate in enumerate(candidates):
        load_kwargs: dict[str, Any] = {
            "path": candidate.repository,
            "name": candidate.config,
            "split": candidate.split,
            "cache_dir": cache_dir,
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
        try:
            dataset = load_dataset(**load_kwargs)
            shuffled = dataset.shuffle(seed=seed + DOMAIN_SEED_OFFSETS[domain])
            formatter = (
                FORMATTERS[candidate.formatter]
                if format_style == "legacy"
                else NEUTRAL_CONTENT_FORMATTERS[candidate.formatter]
            )
            texts: list[str] = []
            example_ids: list[str] = []
            for row_index, row in enumerate(shuffled):
                text = formatter(row, include_answers)
                if not text or len(text.strip()) < 8:
                    continue
                texts.append(text.strip())
                example_ids.append(_example_id(row, row_index))
                if len(texts) >= num_examples:
                    break
            if not texts:
                raise RuntimeError("dataset yielded no non-empty formatted examples")
            metadata = _dataset_metadata(
                dataset,
                candidate,
                requested_revision=revision,
                requested_examples=num_examples,
                actual_examples=len(texts),
                include_answers=include_answers,
                substituted=candidate_index > 0,
                failures=failures,
                example_ids=example_ids,
                format_style=format_style,
            )
            return DomainExamples(domain=domain, texts=texts, metadata=metadata)
        except Exception as exc:  # dataset/network errors vary by datasets version
            failures.append(
                {
                    "repository": candidate.repository,
                    "config": str(candidate.config),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    details = "\n".join(
        f"  - {item['repository']} ({item['config']}): {item['error']}" for item in failures
    )
    raise RuntimeError(f"Could not load a dataset for domain {domain!r}:\n{details}")


def validation_prompts() -> list[str]:
    """Four short, fixed examples used only to validate instrumentation."""
    return [
        "A coastal storm moved inland overnight, bringing rain and strong wind to the city.",
        "Solve step by step: if 3x + 7 = 22, what is x? Answer: x = 5.",
        "Write a Python function that returns the square of each number.\n"
        "def squares(xs):\n    return [x * x for x in xs]",
        "All robins are birds. Some birds migrate. Does it follow that all robins migrate? "
        "No; the conclusion does not follow from the premises.",
    ]


def _format_wikitext(row: Mapping[str, Any], include_answers: bool) -> str:
    del include_answers
    return str(row.get("text", ""))


def _format_lambada(row: Mapping[str, Any], include_answers: bool) -> str:
    del include_answers
    return str(row.get("text", row.get("sentence", "")))


def _format_gsm8k(row: Mapping[str, Any], include_answers: bool) -> str:
    question = str(row.get("question", ""))
    answer = str(row.get("answer", ""))
    if include_answers and answer:
        return f"Math problem:\n{question}\n\nWorked answer:\n{answer}"
    return f"Math problem:\n{question}\n\nAnswer:"


def _format_hendrycks_math(row: Mapping[str, Any], include_answers: bool) -> str:
    problem = str(row.get("problem", ""))
    solution = str(row.get("solution", ""))
    if include_answers and solution:
        return f"Math problem:\n{problem}\n\nWorked solution:\n{solution}"
    return f"Math problem:\n{problem}\n\nSolution:"


def _format_mbpp(row: Mapping[str, Any], include_answers: bool) -> str:
    task = str(row.get("text", row.get("prompt", "")))
    tests = row.get("test_list", [])
    test_text = "\n".join(str(item) for item in tests[:3]) if isinstance(tests, list) else ""
    prefix = f"Python programming task:\n{task}"
    if test_text:
        prefix += f"\n\nTests:\n{test_text}"
    code = str(row.get("code", ""))
    if include_answers and code:
        return f"{prefix}\n\nReference solution:\n```python\n{code}\n```"
    return f"{prefix}\n\nSolution:\n```python"


def _format_humaneval(row: Mapping[str, Any], include_answers: bool) -> str:
    prompt = str(row.get("prompt", ""))
    solution = str(row.get("canonical_solution", ""))
    if include_answers:
        return f"Python programming task and reference solution:\n```python\n{prompt}{solution}\n```"
    return f"Python programming task:\n```python\n{prompt}"


def _format_arc(row: Mapping[str, Any], include_answers: bool) -> str:
    question = str(row.get("question", row.get("question_stem", "")))
    choices = row.get("choices", {})
    labels: list[Any] = []
    texts: list[Any] = []
    if isinstance(choices, Mapping):
        labels = list(choices.get("label", []))
        texts = list(choices.get("text", []))
    elif isinstance(choices, list):
        labels = [choice.get("label", index + 1) for index, choice in enumerate(choices)]
        texts = [choice.get("text", "") for choice in choices]
    rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    prompt = f"Reasoning question:\n{question}\n\nChoices:\n{rendered}\n\nAnswer:"
    answer = str(row.get("answerKey", row.get("answer_key", "")))
    return f"{prompt} {answer}" if include_answers and answer else prompt


def _neutral_wikitext(row: Mapping[str, Any], include_answers: bool) -> str:
    del include_answers
    return str(row.get("text", ""))


def _neutral_lambada(row: Mapping[str, Any], include_answers: bool) -> str:
    del include_answers
    return str(row.get("text", row.get("sentence", "")))


def _neutral_gsm8k(row: Mapping[str, Any], include_answers: bool) -> str:
    del include_answers
    return str(row.get("question", ""))


def _neutral_hendrycks_math(row: Mapping[str, Any], include_answers: bool) -> str:
    del include_answers
    return str(row.get("problem", ""))


def _neutral_mbpp(row: Mapping[str, Any], include_answers: bool) -> str:
    del include_answers
    task = str(row.get("text", row.get("prompt", "")))
    tests = row.get("test_list", [])
    test_text = "\n".join(str(item) for item in tests[:3]) if isinstance(tests, list) else ""
    return f"{task}\n\n{test_text}" if test_text else task


def _neutral_humaneval(row: Mapping[str, Any], include_answers: bool) -> str:
    del include_answers
    return str(row.get("prompt", ""))


def _neutral_arc(row: Mapping[str, Any], include_answers: bool) -> str:
    del include_answers
    question = str(row.get("question", row.get("question_stem", "")))
    choices = row.get("choices", {})
    labels: list[Any] = []
    texts: list[Any] = []
    if isinstance(choices, Mapping):
        labels = list(choices.get("label", []))
        texts = list(choices.get("text", []))
    elif isinstance(choices, list):
        labels = [choice.get("label", index + 1) for index, choice in enumerate(choices)]
        texts = [choice.get("text", "") for choice in choices]
    rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return f"{question}\n\n{rendered}" if rendered else question


FORMATTERS: dict[str, Callable[[Mapping[str, Any], bool], str]] = {
    "wikitext": _format_wikitext,
    "lambada": _format_lambada,
    "gsm8k": _format_gsm8k,
    "hendrycks_math": _format_hendrycks_math,
    "mbpp": _format_mbpp,
    "humaneval": _format_humaneval,
    "arc": _format_arc,
}


NEUTRAL_CONTENT_FORMATTERS: dict[str, Callable[[Mapping[str, Any], bool], str]] = {
    "wikitext": _neutral_wikitext,
    "lambada": _neutral_lambada,
    "gsm8k": _neutral_gsm8k,
    "hendrycks_math": _neutral_hendrycks_math,
    "mbpp": _neutral_mbpp,
    "humaneval": _neutral_humaneval,
    "arc": _neutral_arc,
}


def _example_id(row: Mapping[str, Any], fallback: int) -> str:
    for key in ("id", "task_id", "idx", "problem_id"):
        if key in row:
            return str(row[key])
    return str(fallback)


def _dataset_metadata(
    dataset: Any,
    candidate: DatasetCandidate,
    requested_revision: str | None,
    requested_examples: int,
    actual_examples: int,
    include_answers: bool,
    substituted: bool,
    failures: list[dict[str, str]],
    example_ids: list[str],
    format_style: str,
) -> dict[str, Any]:
    info = getattr(dataset, "info", None)
    resolved_revision = None
    try:
        from huggingface_hub import HfApi

        resolved_revision = HfApi().dataset_info(
            candidate.repository, revision=requested_revision
        ).sha
    except Exception:
        # Dataset fingerprint and packaged version still make the local artifact auditable.
        pass
    return {
        "repository": candidate.repository,
        "config": candidate.config,
        "split": candidate.split,
        "formatter": candidate.formatter,
        "format_style": format_style,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_version": str(getattr(info, "version", "")) if info is not None else None,
        "dataset_rows": len(dataset),
        "requested_examples": requested_examples,
        "actual_examples": actual_examples,
        "include_reference_answers": include_answers,
        "substituted": substituted,
        "failed_primary_candidates": failures,
        "selected_example_ids": example_ids,
    }
