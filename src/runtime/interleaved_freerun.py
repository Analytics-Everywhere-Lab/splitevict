"""Runner for T7: ``interleaved_h100.py`` with free-running decoding.

Every arm feeds the model its own greedy token (``run_path(..., teacher_force=False)``) instead of the trace's
recorded token. The recorded tokens and routing stay the reference that every step is checked against
(``token_agreements``, ``routing_matches``): the new trace was generated greedily by this runtime, so an arm that
leaves the model's arithmetic alone reproduces it token for token, and a mismatch would mean that a policy changed
the model's output.

Each sequence decodes as many steps as the trace recorded. ``run_path`` would stop free-running decoding as soon as
it predicts EOS, before feeding it; the trace generator feeds the EOS token through the model, records its routing
and only then stops, and so do teacher forcing and the simulator. The wrapper therefore hands ``run_path`` the spec
with ``stop_on_eos`` off: an exact run then feeds exactly the recorded tokens, EOS included, and a run that predicts
EOS early simply shows a token mismatch.

Everything else is ``interleaved_h100.py`` as it is; only its ``run_path`` is swapped for the run's duration.
The runner writes to ``<output>.partial``; the wrapper adds ``"decoding": "free_running"`` and
``"runner": "interleaved_freerun.py"`` and moves the result into place atomically, so ``<output>`` exists only complete.

    python src/runtime/interleaved_freerun.py <interleaved_h100.py arguments>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import interleaved_h100  # noqa: E402

_teacher_forced_run_path = interleaved_h100.run_path


def free_running_run_path(runtime, tokenizer, spec, workload, *, teacher_force, **kwargs):
    """``run_path`` with the model's own greedy tokens fed back for the recorded number of steps, whatever the caller
    asked for."""

    spec = dict(spec, trace_generation=dict(spec["trace_generation"], stop_on_eos=False))
    return _teacher_forced_run_path(runtime, tokenizer, spec, workload, teacher_force=False, **kwargs)


def output_path(argv: list[str]) -> tuple[Path, list[str]]:
    """The requested output, and the arguments with the runner's output redirected to ``<output>.partial``."""

    argv = list(argv)
    for i, value in enumerate(argv):
        if value == "--output":
            output = Path(argv[i + 1])
            argv[i + 1] = str(output) + ".partial"
            return output, argv
        if value.startswith("--output="):
            output = Path(value.split("=", 1)[1])
            argv[i] = f"--output={output}.partial"
            return output, argv
    raise SystemExit("--output is required")


def main(argv=None) -> None:
    output, argv = output_path(list(sys.argv[1:] if argv is None else argv))
    partial = Path(str(output) + ".partial")
    previous, interleaved_h100.run_path = interleaved_h100.run_path, free_running_run_path
    try:
        interleaved_h100.main(argv)
    finally:
        interleaved_h100.run_path = previous
    out = json.loads(partial.read_text())
    out["decoding"] = "free_running"
    out["runner"] = "interleaved_freerun.py"
    staging = Path(str(output) + ".tmp")
    staging.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(staging, output)
    partial.unlink()


if __name__ == "__main__":
    main()
