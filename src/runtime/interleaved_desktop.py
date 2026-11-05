"""Runner for T11: P-core placement on the desktop (RTX 4090 on PCIe gen4 x16, Intel i9-13900K).

The i9-13900K has eight P-cores with two hardware threads each (CPUs 0-15; siblings 0/1, 2/3, ..., 14/15) and sixteen
E-cores without SMT (CPUs 16-31, at most 4.3 GHz). ``interleaved_h100.plan_placement`` groups CPUs by thread siblings,
so it counts every E-core as a physical core of its own and would put decode threads on E-cores. This module replaces
it, for the run's duration, by a placement on P-cores only (the Linux hybrid PMU lists them in
``/sys/devices/cpu_core/cpus``):

* each split-phase worker gets one P-core: its first hardware thread; the sibling stays unused;
* the decode process gets the first hardware thread of every other P-core; their siblings stay unused, so no two
  threads of the run share a core (in an exploratory run, a worker on the decode thread's sibling slowed decoding by
  about half);
* no thread of the run is placed on an E-core.

Workers take the first P-cores in CPU order (CPU 0 for one worker), so the two favoured 5.8 GHz cores of this CPU
(CPUs 8 and 10) stay in the decode set. The plan is printed and stored in the run's ``placement``.

Everything else is ``interleaved_freerun.py`` (free-running decoding, experts copied from pinned host memory) for the
native setting and ``interleaved_nvme.py`` (every demand copy read from the drive with direct I/O, then free-running
decoding) when ``--nvme-dir`` is given.

    python src/runtime/interleaved_desktop.py [--nvme-dir DIR] <interleaved_h100.py arguments> --placement physical
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import interleaved_h100  # noqa: E402

P_CORE_CPUS = Path("/sys/devices/cpu_core/cpus")


def plan_placement(workers: int, gpu_index: int = 0) -> dict:
    """Workers and the decode process on P-cores only, one hardware thread per core (``gpu_index``: unused, one
    NUMA node)."""

    p_cpus = interleaved_h100.parse_cpulist(P_CORE_CPUS.read_text())
    cores = [core for core in interleaved_h100.physical_cores() if set(core) <= p_cpus]
    if len(cores) < workers + 1:
        raise SystemExit(f"placement needs {workers + 1} P-cores, {len(cores)} are allowed")
    plan = {"rule": "t11: P-cores only; one hardware thread per core; siblings and E-cores unused",
            "cores": cores, "workers": [core[0] for core in cores[:workers]],
            "main": [core[0] for core in cores[workers:]],
            "excluded_siblings": sorted(c for core in cores for c in core[1:]),
            "unused_e_cores": sorted(set(os.sched_getaffinity(0)) - p_cpus)}
    print(f"placement: workers on CPUs {plan['workers']}, decode process on CPUs {plan['main']}, "
          f"siblings unused {plan['excluded_siblings']}, E-cores unused {plan['unused_e_cores']}", flush=True)
    return plan


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    interleaved_h100.plan_placement = plan_placement
    if any(a == "--nvme-dir" or a.startswith("--nvme-dir=") for a in argv):
        import interleaved_nvme

        interleaved_nvme.main(argv)
    else:
        import interleaved_freerun

        interleaved_freerun.main(argv)


if __name__ == "__main__":
    main()
