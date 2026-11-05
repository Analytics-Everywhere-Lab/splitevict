"""Interleaved runner for T5 and T6 on an H100 node: ``interleaved_prefetch.py`` with more arms and a copy-cost sweep.

The interleaved design, the arms with and without prefetching and the CPU placement are those of
``interleaved_prefetch.py``. Additions:

* the synchronous learned ranker as an arm (``global_token_gbt``, no prefetching; ``--sync-bundle``);
* the eviction rules of HOBBIT (``hobbit``, ``--hobbit-w``) and MoE-Infinity (``moe_infinity``,
  ``--moe-infinity-store``) as arms, with or without prefetching (``policy_baselines.py``);
* ``--copy-repeat K``: every expert copy is issued K times into its slot. The bytes stay the same; the copy
  time and issue cost grow K times. This is the controlled copy-cost sweep for the break-even model;
* ``--gpu-index``: the physical GPU this process uses, for its NUMA placement and for a GPU-busy check
  limited to that GPU (when one job runs one process per CPU socket).

    python src/runtime/interleaved_h100.py --spec data/specs/olmoe_t5_replay_spec.json --capacity 32 \
        --sequences calibration --passes 2 --placement physical --monitor --lean --copy-api driver --warmup 1 \
        --arms lfu_decay hobbit splitphase global_token_gbt --hobbit-w W \
        --p2-bundle data/rankers/olmoe_frozen_lag_req_trees200.pkl \
        --p1-bundle data/rankers/olmoe_frozen_lag_tok_trees200.pkl \
        --sync-bundle data/rankers/olmoe_setdev_policy.pkl --output data/runs/t5/olmoe/B32.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from interleaved_multi import SharedSlotContext, rotation  # noqa: E402
from offload_runtime import (PREFETCH_ROW_KEYS, OffloadRuntime, PrefetchConfig, _sha256,  # noqa: E402
                             bandwidth_probe, load_model, run_path)
from policy import LFU_DECAY_ALPHA, make_policy  # noqa: E402

ARMS = ("lfu_decay", "splitphase", "lfu_decay+pf", "splitphase+pf")
BASES = ("lfu_decay", "splitphase", "global_token_gbt", "hobbit", "moe_infinity")


def parse_arm(arm: str, default: PrefetchConfig) -> tuple[str, PrefetchConfig | None]:
    """``base``, ``base+pf`` (the default prefetcher) or ``base+pf:m=12,q=3`` (its own m/q)."""

    head, _, options = arm.partition(":")
    base, _, suffix = head.partition("+")
    if base not in BASES or suffix not in ("", "pf") or (options and not suffix):
        raise SystemExit(f"bad arm {arm!r}")
    if base == "global_token_gbt" and suffix:
        raise SystemExit("the synchronous ranker runs without prefetching")
    if not suffix:
        return base, None
    values = {"m": default.m, "q": default.q}
    for item in filter(None, options.split(",")):
        key, _, value = item.partition("=")
        values[key] = int(value)
    return base, PrefetchConfig(m=values["m"], q=values["q"], variant=default.variant)


def physical_cores() -> list[list[int]]:
    """The allowed CPUs grouped by physical core (Linux sysfs), cores in order of their first CPU."""

    allowed = sorted(os.sched_getaffinity(0))
    groups: dict[str, list[int]] = {}
    for cpu in allowed:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        key = path.read_text().strip() if path.exists() else str(cpu)
        groups.setdefault(key, []).append(cpu)
    return sorted(groups.values(), key=lambda g: g[0])


def confine_process(cpus: set[int]) -> None:
    """Every existing thread of this process, and every thread it starts later, on ``cpus``."""

    for task in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpus)
        except (ProcessLookupError, PermissionError):
            pass
    os.sched_setaffinity(0, cpus)


def parse_cpulist(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def gpu_local_cpus(index: int = 0) -> set[int] | None:
    """The CPUs of the GPU's NUMA node (sysfs ``local_cpulist`` of its PCI device), if the host says."""

    import subprocess

    try:
        bus = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
        domain, rest = bus.split(":", 1)
        path = Path(f"/sys/bus/pci/devices/{int(domain, 16):04x}:{rest.lower()}/local_cpulist")
        return parse_cpulist(path.read_text()) if path.exists() else None
    except Exception:
        return None


def plan_placement(workers: int, gpu_index: int = 0) -> dict:
    """Workers and the decode process on the GPU's NUMA node; each worker alone on a physical core whose
    other hyperthread nobody of ours uses; the decode process on the node's remaining cores."""

    cores = physical_cores()
    local = gpu_local_cpus(gpu_index)
    if local:
        node = [core for core in cores if set(core) <= local]
        if len(node) >= workers + 2:
            cores = node
    if len(cores) < workers + 1:
        raise SystemExit(f"placement needs {workers + 1} physical cores, the container allows {len(cores)}")
    worker_cpus = [cores[j][0] for j in range(workers)]
    main = sorted(core[0] for core in cores[workers:])  # one hyperthread per core: our threads never share a core
    return {"cores": cores, "workers": worker_cpus, "main": main, "gpu_local_cpus": sorted(local) if local else None,
            "excluded_siblings": sorted(c for core in cores for c in core[1:])}


def totals(rows: list[dict]) -> dict:
    keys = ("tokens", "expert_transfers", "bytes_transferred", "wall_ms", "policy_ms", "issue_ms", "copy_ms",
            "routing_events", "routing_matches", "token_agreements", "token_comparisons") + PREFETCH_ROW_KEYS
    out = {k: sum(r.get(k, 0) for r in rows) for k in keys}
    tok = max(out["tokens"], 1)
    out.update({"ms_per_token_wall": out["wall_ms"] / tok, "ms_per_token_policy": out["policy_ms"] / tok,
                "ms_per_token_issue": out["issue_ms"] / tok, "ms_per_token_copy": out["copy_ms"] / tok,
                "transfers_per_token": out["expert_transfers"] / tok,
                "prefetches_per_token": out["prefetch_transfers"] / tok,
                "ms_per_token_prefetch_copy": out["prefetch_copy_ms"] / tok,
                "ms_per_token_prefetch_stall": out["prefetch_stall_ms"] / tok,
                "ms_per_token_prefetch_policy": out["prefetch_policy_ms"] / tok,
                "ms_per_token_prefetch_issue": out["prefetch_issue_ms"] / tok,
                "ms_per_copy": out["copy_ms"] / max(out["expert_transfers"], 1),
                "ms_per_prefetch_copy": out["prefetch_copy_ms"] / max(out["prefetch_transfers"], 1),
                "routing_match_fraction": out["routing_matches"] / max(out["routing_events"], 1),
                "token_agreement_fraction": out["token_agreements"] / max(out["token_comparisons"], 1),
                "guess_match_fraction": (out["guess_matches"] / out["guess_events"]) if out["guess_events"] else None})
    return out


def load_guess_reference(path: Path, variant: str) -> dict[int, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {int(key.split("/")[1]): data[key].astype(np.int64) for key in data.files if key.startswith(f"{variant}/")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS),
                        help="lfu_decay | splitphase | hobbit | moe_infinity, optionally +pf or +pf:m=12,q=3; "
                             "global_token_gbt (no prefetching)")
    parser.add_argument("--sequences", default="calibration",
                        help="'calibration', 'all', a workload path name, or comma-separated sequence ids")
    parser.add_argument("--sync-bundle", type=Path, help="global_token_gbt: the synchronous learned ranker")
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="physical GPU index (CUDA_VISIBLE_DEVICES is set to it by the job script)")
    parser.add_argument("--p2-bundle", type=Path)
    parser.add_argument("--p1-bundle", type=Path)
    parser.add_argument("--alpha", type=float, default=LFU_DECAY_ALPHA)
    parser.add_argument("--hobbit-w", type=float, help="hobbit: the recency weight, fitted on calibration data")
    parser.add_argument("--moe-infinity-store", type=Path,
                        help="moe_infinity: .npy of calibration request EAMs, shaped [requests, layers, experts]")
    parser.add_argument("--moe-infinity-static", action="store_true",
                        help="moe_infinity: do not add finished sequences to the store")
    parser.add_argument("--prefetch-m", type=int, default=8)
    parser.add_argument("--prefetch-q", type=int, default=2)
    parser.add_argument("--prefetch-variant", default="g1", choices=("g1", "g1n"))
    parser.add_argument("--lfu-wrong-rule", default="last", choices=("last", "policy"))
    parser.add_argument("--guesses", type=Path, help="recorded guesses (record_guesses.py): live guesses are checked")
    parser.add_argument("--placement", default="none", choices=("none", "physical"))
    parser.add_argument("--monitor", action="store_true", help="record the decode thread's CPU and host CPU load")
    parser.add_argument("--profile", type=Path, help="torch.profiler every decoded sequence into this directory (not timed)")
    parser.add_argument("--lean", action="store_true",
                        help="no prefetch-copy or stall timing events (the needed-stall counter stays)")
    parser.add_argument("--copy-api", default="torch", choices=("torch", "driver"),
                        help="issue expert copies with Tensor.copy_ or the CUDA driver's cuMemcpyHtoDAsync")
    parser.add_argument("--sync-audit", action="store_true",
                        help="count CPU-GPU synchronizations per token in every arm (torch sync debug mode; not timed)")
    parser.add_argument("--warmup", type=int, default=0,
                        help="decode this many sequences first with a throwaway LFU+prefetch context (untimed, discarded)")
    parser.add_argument("--passes", type=int, default=1,
                        help="replay the whole schedule this many times, each pass from cold caches (timing repeats)")
    parser.add_argument("--experts-on-host", action="store_true")
    parser.add_argument("--copy-repeat", type=int, default=1,
                        help="issue every expert copy this many times into its slot (same bytes; copy-cost sweep)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    if len(set(args.arms)) != len(args.arms):
        raise SystemExit("arm names must be unique")
    split_arms = [a for a in args.arms if a.startswith("splitphase")]
    if args.device.startswith("cuda"):
        import subprocess

        others = subprocess.run(["nvidia-smi", f"--id={args.gpu_index}", "--query-compute-apps=pid",
                                 "--format=csv,noheader"], capture_output=True, text=True).stdout.split()
        if others:  # a second process on the GPU would contaminate every timing
            raise SystemExit(f"the GPU is already used by process(es) {others}")
    placement = plan_placement(len(split_arms), args.gpu_index) if args.placement == "physical" else None
    if placement is not None:
        confine_process(set(placement["main"]))

    import torch
    from transformers import AutoTokenizer

    spec = json.loads(args.spec.read_text())
    arch = spec["architecture"]
    L, E, k = int(arch["num_moe_layers"]), int(arch["num_experts"]), int(arch["top_k"])
    if not k < args.capacity <= E:
        raise SystemExit("capacity must satisfy top_k < B <= E")
    by_id = {int(s["sequence_id"]): s for s in spec["sequences"]}
    if args.sequences == "calibration":
        ids = [int(v) for v in spec["calibration_sequence_ids"]]
    elif args.sequences == "all":
        ids = [int(s["sequence_id"]) for s in spec["sequences"]]
    elif args.sequences in {w["name"] for w in spec["workloads"]}:
        ids = [int(s["source_sequence_id"]) for s in {w["name"]: w for w in spec["workloads"]}[args.sequences]["sequences"]]
    else:
        ids = [int(v) for v in args.sequences.split(",")]
    if args.limit:
        ids = ids[: args.limit]
    items = [{"source_sequence_id": sid, "position": i, "domain": by_id[sid]["domain"]} for i, sid in enumerate(ids)]

    device = torch.device(args.device)
    torch.manual_seed(int(spec["trace_generation"]["seed"]))
    attn = spec["trace_generation"].get("attn_implementation", "eager")
    model = load_model(spec, device, torch.bfloat16, attn, None, args.experts_on_host, log=lambda m: print(m, flush=True))
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["model_revision"])
    pf = PrefetchConfig(m=args.prefetch_m, q=args.prefetch_q, variant=args.prefetch_variant)

    configs = {arm: parse_arm(arm, pf) for arm in args.arms}
    bases = {c[0] for c in configs.values()}
    if "hobbit" in bases and args.hobbit_w is None:
        raise SystemExit("hobbit needs --hobbit-w")
    eam_store = None
    if "moe_infinity" in bases:
        if args.moe_infinity_store is None:
            raise SystemExit("moe_infinity needs --moe-infinity-store")
        eam_store = np.load(args.moe_infinity_store)
        if eam_store.ndim != 3 or eam_store.shape[1:] != (L, E):
            raise SystemExit(f"the EAM store is shaped {eam_store.shape}, not [requests, {L}, {E}]")

    def make_policies():
        """Fresh policies (and split-phase services) for one pass: every pass starts cold."""

        services, policies = {}, {}
        for arm in args.arms:
            base = configs[arm][0]
            if base == "splitphase":
                from splitphase_service import SplitPhasePolicy, SplitPhaseService

                cpu = None if placement is None else [placement["workers"][split_arms.index(arm)]]
                services[arm] = SplitPhaseService(str(args.p2_bundle), str(args.p1_bundle), int(args.capacity),
                                                  workers=1, num_layers=L, num_experts=E, top_k=k, cpus=cpu)
                policies[arm] = SplitPhasePolicy(services[arm], L, E, int(args.capacity), k)
            elif base == "hobbit":
                from policy_baselines import HobbitPolicy

                policies[arm] = HobbitPolicy(L, E, int(args.capacity), k, w=float(args.hobbit_w))
                if configs[arm][1] is not None:
                    policies[arm].wrong_guess_rule = args.lfu_wrong_rule
            elif base == "moe_infinity":
                from policy_baselines import MoEInfinityPolicy

                policies[arm] = MoEInfinityPolicy(L, E, int(args.capacity), k, store=eam_store,
                                                  online=not args.moe_infinity_static)
                if configs[arm][1] is not None:
                    policies[arm].wrong_guess_rule = args.lfu_wrong_rule
            elif base == "global_token_gbt":
                from policy import PolicyBundle

                policies[arm] = make_policy(base, num_layers=L, num_experts=E, capacity=int(args.capacity), top_k=k,
                                            bundle=PolicyBundle.load(args.sync_bundle), backend="numba",
                                            score_scope="candidates")
            else:
                policies[arm] = make_policy(base, num_layers=L, num_experts=E, capacity=int(args.capacity), top_k=k,
                                            alpha=args.alpha)
                if configs[arm][1] is not None:
                    policies[arm].wrong_guess_rule = args.lfu_wrong_rule
        return services, policies

    services, policies = make_policies()
    if args.copy_repeat < 1:
        raise SystemExit("--copy-repeat must be at least 1")
    if args.copy_repeat > 1:
        from offload_runtime import OffloadedMoeLayer

        single = OffloadedMoeLayer.copy_expert

        def repeated(self, expert, slot, stream_handle, _k=int(args.copy_repeat)):
            for _ in range(_k):
                single(self, expert, slot, stream_handle)

        OffloadedMoeLayer.copy_expert = repeated
    runtime = OffloadRuntime(model, spec, device, policies[args.arms[0]], int(args.capacity), compute="module",
                             experts_on_host=args.experts_on_host, copy_api=args.copy_api)
    runtime.detailed_timing = not args.lean
    if args.guesses is not None:
        runtime.guess_reference = load_guess_reference(args.guesses, args.prefetch_variant)
    if args.monitor:
        from offload_runtime import HostMonitor

        runtime.host_monitor = HostMonitor()
    probes = [bandwidth_probe(runtime.layers[0], device)]

    from offload_runtime import thread_cpu

    class SyncAudit:
        """Counts torch's 'synchronizing CUDA operation' warnings, by the source line that raised them."""

        def __init__(self):
            import warnings

            torch.cuda.set_sync_debug_mode(1)
            self.counts = {}
            original = warnings.showwarning

            def show(message, category, filename, lineno, file=None, line=None):
                if "synchroniz" in str(message):
                    key = f"{Path(filename).name}:{lineno}"
                    self.counts[key] = self.counts.get(key, 0) + 1
                else:
                    original(message, category, filename, lineno, file, line)

            warnings.simplefilter("always")
            warnings.showwarning = show

        def pop(self):
            out, self.counts = self.counts, {}
            return out

    audit = SyncAudit() if args.sync_audit else None

    def decode(ctx, item, worker_pids=()):
        runtime.prefetch = ctx.prefetch
        runtime.guess_log = ctx.guess_log
        pids = [os.getpid(), *worker_pids]
        before = thread_cpu(pids) if args.monitor else {}
        workload = {"name": f"interleaved_{args.sequences}", "regime": "interleaved", "sequences": [item]}
        if args.profile is not None and ctx.name != "warmup":
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                result = run_path(runtime, tokenizer, spec, workload, teacher_force=True, limit_sequences=None,
                                  log=lambda message: None)
            args.profile.mkdir(parents=True, exist_ok=True)
            stem = f"{ctx.name.replace(':', '_').replace(',', '_').replace('=', '')}_{item['source_sequence_id']}"
            (args.profile / f"{stem}_cpu.txt").write_text(
                prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
            (args.profile / f"{stem}_cuda.txt").write_text(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
            prof.export_chrome_trace(str(args.profile / f"{stem}_trace.json"))
        else:
            result = run_path(runtime, tokenizer, spec, workload, teacher_force=True, limit_sequences=None,
                              log=lambda message: None)
        row = result["sequences"][0]
        if audit is not None:
            row["sync_warnings"] = audit.pop()
        if args.monitor:
            after = thread_cpu(pids)
            row["threads"] = [[tid, ticks - before.get(tid, (0, 0))[0], cpu] for tid, (ticks, cpu) in after.items()
                              if ticks - before.get(tid, (0, 0))[0] > 0]
        return row

    if args.warmup:
        # warm the CUDA kernels, the copy stream and the prefetch path on a context that is then discarded
        warm = SharedSlotContext("warmup", make_policy("lfu_decay", num_layers=L, num_experts=E,
                                                       capacity=int(args.capacity), top_k=k, alpha=args.alpha), runtime)
        # the warm-up prefetches only if some arm does (Mixtral runs no prefetch arm)
        warm.prefetch = pf if any(configs[a][1] is not None for a in args.arms) else None
        warm.guess_log = {}
        for item in items[: int(args.warmup)]:
            warm.use(runtime)
            decode(warm, item)
        print(f"warm-up: {int(args.warmup)} sequence(s) decoded and discarded", flush=True)

    names = list(args.arms)
    order_log = []
    rows_by_arm = {name: [] for name in names}
    refills = {name: [0, 0.0] for name in names}
    guess_logs: dict[str, dict] = {name: {} for name in names}
    service_stats = {name: [] for name in services}
    start = time.perf_counter()
    for pass_index in range(int(args.passes)):
        if pass_index:
            services, policies = make_policies()
        contexts = []
        for arm in names:
            ctx = SharedSlotContext(arm, policies[arm], runtime)
            ctx.prefetch = configs[arm][1]
            ctx.guess_log = guess_logs[arm] if pass_index == 0 else {}
            contexts.append(ctx)
        by_name = {c.name: c for c in contexts}
        for i, item in enumerate(items):
            order = rotation(names, i + pass_index)
            for slot, name in enumerate(order):
                ctx = by_name[name]
                ctx.use(runtime)
                pids = [p.pid for p in services[name].procs] if name in services else []
                row = dict(decode(ctx, item, pids))
                row["order_slot"] = slot
                row["pass"] = pass_index
                row["clock_s"] = time.perf_counter() - start
                ctx.rows.append(row)
            order_log.append(order)
            if (i + 1) % 10 == 0 or i == 0:
                parts = ", ".join(f"{c.name} {c.rows[-1]['wall_ms'] / max(c.rows[-1]['tokens'], 1):.1f} ms/tok "
                                  f"({c.rows[-1]['expert_transfers']}+{c.rows[-1]['prefetch_transfers']} xfer)"
                                  for c in contexts)
                print(f"pass {pass_index} seq {i + 1}/{len(items)}: {parts}", flush=True)
        for c in contexts:
            rows_by_arm[c.name].extend(c.rows)
            refills[c.name][0] += c.refills
            refills[c.name][1] += c.refill_seconds
        for arm, service in services.items():
            sp = policies[arm]
            busy = service.close()
            tokens = max(sum(r["tokens"] for r in by_name[arm].rows), 1)
            service_stats[arm].append({"pass": pass_index, "workers": 1, "cpus": service.cpus,
                                       "wait_ms_per_token": sp.wait_ns / 1e6 / tokens,
                                       "prefetch_wait_ms_per_token": sp.prefetch_wait_ns / 1e6 / tokens,
                                       "decide_ms_per_token": sp.decide_ns / 1e6 / tokens,
                                       "post_ms_per_token": sp.post_ns / 1e6 / tokens,
                                       "worker_busy_ms_per_token": [b / 1e6 / tokens for b in busy],
                                       "worker_profile": getattr(service, "profile", None)})
    elapsed = time.perf_counter() - start
    probes.append(bandwidth_probe(runtime.layers[0], device))

    host = {"hostname": socket.gethostname(), "platform": platform.platform(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "experts_on_host": bool(args.experts_on_host), "cpu_count": os.cpu_count(),
            "allowed_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    try:
        import transformers

        host["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        pass
    out = {
        "schema_version": "hardware_interleaved_h100_v1", "label": args.label, "model": spec["model"],
        "trace_hash": spec["trace_hash"], "capacity": int(args.capacity), "sequences": args.sequences,
        "design": "every arm decodes every sequence back to back in one process, order rotating per sequence; one "
                  "shared slot set refilled from host memory when the arm changes (untimed)",
        "prefetch": {"m": pf.m, "q": pf.q, "variant": pf.variant, "lfu_wrong_rule": args.lfu_wrong_rule},
        "copy_api": args.copy_api, "lean": bool(args.lean), "copy_repeat": int(args.copy_repeat),
        "placement": placement, "host": host, "bandwidth_probes": probes, "order": order_log, "alpha": args.alpha,
        "guesses": None if args.guesses is None else {"path": str(args.guesses), "sha256": _sha256(args.guesses)},
        "gpu_index": args.gpu_index,
        "baselines": {"hobbit_w": args.hobbit_w,
                      "moe_infinity_store": None if args.moe_infinity_store is None else {
                          "path": str(args.moe_infinity_store), "sha256": _sha256(args.moe_infinity_store),
                          "online": not args.moe_infinity_static}},
        "bundles": {"sync": None if args.sync_bundle is None else {"path": str(args.sync_bundle), "sha256": _sha256(args.sync_bundle)},
                    "p2": None if args.p2_bundle is None else {"path": str(args.p2_bundle), "sha256": _sha256(args.p2_bundle)},
                    "p1": None if args.p1_bundle is None else {"path": str(args.p1_bundle), "sha256": _sha256(args.p1_bundle)}},
        "passes": int(args.passes),
        "arms": {name: {"totals": totals([r for r in rows_by_arm[name] if r["pass"] == 0]),
                        "totals_by_pass": [totals([r for r in rows_by_arm[name] if r["pass"] == p])
                                           for p in range(int(args.passes))],
                        "sequences": rows_by_arm[name], "refills": refills[name][0],
                        "refill_seconds": refills[name][1],
                        "prefetch": None if configs[name][1] is None else {"m": configs[name][1].m, "q": configs[name][1].q,
                                                                           "variant": configs[name][1].variant}}
                 for name in names},
        "splitphase": {arm: stats[0] if len(stats) == 1 else stats for arm, stats in service_stats.items()},
        "elapsed_seconds": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    logs = {f"{name}/{sid}": arr for name, log in guess_logs.items() for sid, arr in log.items()}
    if logs:
        np.savez_compressed(args.output.with_suffix(".guesses.npz"), **logs)
    for name, entry in out["arms"].items():
        t = entry["totals"]
        gm = t["guess_match_fraction"]
        print(f"{name}: {t['ms_per_token_wall']:.2f} ms/token wall, policy {t['ms_per_token_policy']:.2f}"
              f"+{t['ms_per_token_prefetch_policy']:.2f}, copy {t['ms_per_token_copy']:.2f} ({t['ms_per_copy']:.3f} ms/copy), "
              f"{t['transfers_per_token']:.2f}+{t['prefetches_per_token']:.2f} xfer/tok, stall {t['ms_per_token_prefetch_stall']:.2f}, "
              f"routing {100 * t['routing_match_fraction']:.2f}%"
              + ("" if gm is None else f", guesses {100 * gm:.2f}%"), flush=True)
    runtime.release()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
