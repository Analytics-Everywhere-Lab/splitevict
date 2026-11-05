"""Runner for T10: ``interleaved_graphs.py`` with split-phase's scores computed by several worker processes.

In T9, at the H100 node's own copy cost, a graphed split-phase token took 11.1 ms while its one worker needed 10.0 ms
per token for the scores, so the decode thread waited 1.15 ms per token for them. Here the split-phase service runs
``--splitphase-workers`` processes; worker j owns the layers l with l % W == j (``splitphase_service``), so the
decisions are those of one worker. Each worker sits alone on its own physical core of the GPU's NUMA node, and the
decode process keeps the node's remaining cores. Everything else is ``interleaved_graphs.py``.

    python src/runtime/interleaved_graphs_workers.py --splitphase-workers 2 <interleaved_graphs.py arguments>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import interleaved_graphs  # noqa: E402
import interleaved_freerun  # noqa: E402
import interleaved_h100  # noqa: E402
import splitphase_service  # noqa: E402

GROUPS: dict[int, list[int]] = {}  # first CPU of an arm's group -> the group


def placement_for(per_arm: int):
    """``interleaved_h100.plan_placement`` with ``per_arm`` worker cores for every split-phase arm."""

    plan = interleaved_h100.plan_placement

    def plan_placement(arms: int, gpu_index: int = 0) -> dict:
        placement = plan(arms * per_arm, gpu_index)
        groups = [placement["workers"][i * per_arm:(i + 1) * per_arm] for i in range(arms)]
        GROUPS.update({group[0]: group for group in groups})
        placement["workers"] = [group[0] for group in groups]  # the runner hands each arm the first core of its group
        placement["worker_groups"] = groups
        return placement

    return plan_placement


def service_for(per_arm: int):
    """The split-phase service with ``per_arm`` workers, pinned to the arm's group of cores."""

    class Service(splitphase_service.SplitPhaseService):
        def __init__(self, *args, workers: int = 1, cpus: list[int] | None = None, **kwargs) -> None:
            super().__init__(*args, workers=per_arm, cpus=None if cpus is None else GROUPS[cpus[0]], **kwargs)

    return Service


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--splitphase-workers" not in argv:
        raise SystemExit("T10 needs --splitphase-workers")
    i = argv.index("--splitphase-workers")
    workers = int(argv[i + 1])
    del argv[i:i + 2]
    if workers < 1:
        raise SystemExit("--splitphase-workers must be at least 1")
    output, _ = interleaved_freerun.output_path(argv)
    saved = (interleaved_h100.plan_placement, splitphase_service.SplitPhaseService)
    interleaved_h100.plan_placement = placement_for(workers)
    splitphase_service.SplitPhaseService = service_for(workers)
    try:
        interleaved_graphs.main(argv)
    finally:
        interleaved_h100.plan_placement, splitphase_service.SplitPhaseService = saved
    out = json.loads(output.read_text())
    out["runner"] = "interleaved_graphs_workers.py"
    out["splitphase_workers"] = workers
    for stats in (out.get("splitphase") or {}).values():
        for entry in stats if isinstance(stats, list) else [stats]:
            entry["workers"] = workers
    staging = Path(str(output) + ".tmp")
    staging.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(staging, output)
    for arm, stats in (out.get("splitphase") or {}).items():
        for entry in stats if isinstance(stats, list) else [stats]:
            busy = ", ".join(f"{b:.2f}" for b in entry["worker_busy_ms_per_token"])
            print(f"{arm}: {workers} workers on CPUs {entry['cpus']}, busy [{busy}] ms/token, "
                  f"wait {entry['wait_ms_per_token']:.2f} ms/token", flush=True)


if __name__ == "__main__":
    main()
