"""Runner for T8: ``interleaved_freerun.py`` with every demand copy read from the node's NVMe drive (direct I/O).

Every demand copy reads the expert's bytes from a file on the node-local NVMe drive into a pinned host buffer with
``O_DIRECT``, so the page cache is bypassed and every read comes from the drive, and then copies the buffer into the
expert's GPU slot. This is the copy path of a machine whose host memory cannot hold the model's experts. Everything else
is ``interleaved_h100.py`` as it is, run through ``interleaved_freerun.py``'s free-running decoding; the slot refills
between arms and the host-to-device bandwidth probes still copy from pinned host memory, untimed.

Staging: at start-up, layer l's experts are written to ``<nvme-dir>/layer<ll>.bin``, expert e at byte offset
``e * stride``. The stride is one expert's gate-up and down weights rounded up to 4,096 bytes, the alignment direct
I/O needs. The files are synced and dropped from the page cache, and one expert per layer is read back with direct I/O
and compared with its host copy.

Accounting: a read blocks the decode thread. Its time is reported per sequence as ``read_ms`` (over ``reads`` reads) and
kept out of ``issue_ms``, which keeps its meaning: slot bookkeeping and copy issue on the CPU. ``copy_ms`` (CUDA events
around a layer's demand copies) includes the reads, since each host-to-device copy is queued only after its read, so
the break-even model's per-copy cost t_copy + t_issue counts every read once. The pinned staging buffers
(``--nvme-buffers``) are used in turn; a buffer is refilled only after the copy that last read it has finished.

The output adds ``"nvme"``: the staging layout and time, the read-back check, and read probes before and after the
arms (single expert reads at random offsets, and one expert read twice in a row: both reads take drive time when the
page cache is bypassed), plus the host's page-cache size before and after.

    python src/runtime/interleaved_nvme.py --nvme-dir $SLURM_TMPDIR/experts <interleaved_h100.py arguments>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import interleaved_freerun  # noqa: E402
import interleaved_h100  # noqa: E402
from offload_runtime import OffloadedMoeLayer, OffloadRuntime  # noqa: E402

ALIGN = 4096


def page_cache_kib() -> int | None:
    """``Cached`` of /proc/meminfo (Linux), in KiB."""

    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("Cached:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


class NvmeExperts:
    """Every MoE layer's experts in one file on the NVMe drive, read with direct I/O into pinned staging buffers."""

    def __init__(self, runtime: OffloadRuntime, directory: Path, buffers: int) -> None:
        import torch

        self.torch = torch
        self.runtime = runtime
        self.directory = Path(directory)
        first = runtime.layers[0]
        self.gate_up_bytes = first.host_gate_up[0].numel() * first.host_gate_up.element_size()
        self.down_bytes = first.host_down[0].numel() * first.host_down.element_size()
        self.expert_bytes = self.gate_up_bytes + self.down_bytes
        self.stride = -(-self.expert_bytes // ALIGN) * ALIGN
        for layer in runtime.layers:
            if (layer.host_gate_up[0].numel() * layer.host_gate_up.element_size() != self.gate_up_bytes
                    or layer.host_down[0].numel() * layer.host_down.element_size() != self.down_bytes):
                raise RuntimeError("MoE layers with different expert shapes need one stride each")
        self.paths = [self.directory / f"layer{layer.ordinal:02d}.bin" for layer in runtime.layers]
        self.cache_before_stage_kib = page_cache_kib()
        self.stage_seconds = self._stage()
        self.direct = hasattr(os, "O_DIRECT")
        self.fds = [os.open(path, os.O_RDONLY | (os.O_DIRECT if self.direct else 0)) for path in self.paths]
        if not self.direct:  # macOS (tests only): no O_DIRECT, F_NOCACHE keeps the reads out of the cache
            import fcntl

            for fd in self.fds:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        self.buffers = []
        for _ in range(int(buffers)):  # direct I/O needs 4,096-byte aligned buffers: slice one from a slightly larger block
            block = torch.empty(self.stride + ALIGN, dtype=torch.uint8)
            block = block.pin_memory() if runtime.cuda else block
            start = -block.data_ptr() % ALIGN
            self.buffers.append(block[start: start + self.stride])
        self.views = [memoryview(b.numpy()) for b in self.buffers]
        self.events: list = [None] * len(self.buffers)
        self.next = 0
        self.read_ns = 0
        self.reads = 0
        self.readback_equal = self._readback()

    def _host_bytes(self, layer: OffloadedMoeLayer):
        torch = self.torch
        gate_up = layer.host_gate_up.view(torch.uint8).reshape(layer.E, -1).numpy()
        down = layer.host_down.view(torch.uint8).reshape(layer.E, -1).numpy()
        return gate_up, down

    def _stage(self) -> float:
        start = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        pad = bytes(self.stride - self.expert_bytes)
        for layer, path in zip(self.runtime.layers, self.paths):
            gate_up, down = self._host_bytes(layer)
            with open(path, "wb") as handle:
                for expert in range(layer.E):
                    handle.write(gate_up[expert].data)
                    handle.write(down[expert].data)
                    handle.write(pad)
                handle.flush()
                os.fsync(handle.fileno())
            if hasattr(os, "posix_fadvise"):
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
        return time.perf_counter() - start

    def _readback(self) -> bool:
        for layer in self.runtime.layers:
            expert = layer.ordinal % layer.E
            index, _ = self.read(layer.ordinal, expert)
            gate_up, down = self._host_bytes(layer)
            got = np.frombuffer(self.views[index], dtype=np.uint8)
            if not (np.array_equal(got[: self.gate_up_bytes], gate_up[expert])
                    and np.array_equal(got[self.gate_up_bytes: self.expert_bytes], down[expert])):
                return False
        return True

    def read(self, ordinal: int, expert: int) -> tuple[int, int]:
        """Read one expert into the next staging buffer: (buffer index, read nanoseconds)."""

        index = self.next
        self.next = (index + 1) % len(self.buffers)
        event = self.events[index]
        if event is not None:
            event.synchronize()  # the copy that last read this buffer has finished
        start = time.perf_counter_ns()
        got = os.preadv(self.fds[ordinal], [self.views[index]], expert * self.stride)
        elapsed = time.perf_counter_ns() - start
        if got != self.stride:
            raise RuntimeError(f"short read of expert {expert}, layer {ordinal}: {got} of {self.stride} bytes")
        self.read_ns += elapsed
        self.reads += 1
        return index, elapsed

    def probe(self, reads: int = 64, seed: int = 0) -> dict:
        """Single expert reads at random offsets, and one expert read twice in a row (milliseconds)."""

        rng = np.random.default_rng(seed)
        layers = self.runtime.layers
        times = []
        for _ in range(reads):
            layer = layers[int(rng.integers(len(layers)))]
            times.append(self.read(layer.ordinal, int(rng.integers(layer.E)))[1] / 1e6)
        first = self.read(layers[-1].ordinal, layers[-1].E - 1)[1] / 1e6
        second = self.read(layers[-1].ordinal, layers[-1].E - 1)[1] / 1e6
        ms = np.asarray(times)
        return {"reads": reads, "ms_mean": float(ms.mean()), "ms_p50": float(np.percentile(ms, 50)),
                "ms_p90": float(np.percentile(ms, 90)), "ms_max": float(ms.max()),
                "gb_per_s": self.stride / (float(ms.mean()) * 1e6), "repeat_first_ms": first, "repeat_second_ms": second,
                "page_cache_kib": page_cache_kib()}

    def layout(self) -> dict:
        return {"directory": str(self.directory), "files": [p.name for p in self.paths], "direct_io": self.direct,
                "expert_bytes": self.expert_bytes, "stride": self.stride, "buffers": len(self.buffers),
                "stage_seconds": self.stage_seconds, "readback_equal": self.readback_equal,
                "page_cache_kib_before_staging": self.cache_before_stage_kib}


def nvme_copy_expert(self: OffloadedMoeLayer, expert: int, slot: int, stream_handle: int | None) -> None:
    """Read one expert from the drive into a staging buffer, then copy the buffer into a slot (current stream if None)."""

    runtime = self.runtime
    torch = runtime.torch
    store: NvmeExperts = runtime.nvme
    index, elapsed = store.read(self.ordinal, int(expert))
    if stream_handle is None:
        runtime.token_issue_ns -= elapsed
    else:
        runtime.token_prefetch_issue_ns -= elapsed
    buffer = store.buffers[index]
    if runtime.copier is not None and self.copy_pointers is not None:
        _hgu, _hd, sgu, sd, gu, d = self.copy_pointers
        handle = torch.cuda.current_stream().cuda_stream if stream_handle is None else stream_handle
        base = buffer.data_ptr()
        runtime.copier.copy(sgu + slot * gu, base, gu, handle)
        runtime.copier.copy(sd + slot * d, base + gu, d, handle)
    else:
        if stream_handle is not None:
            raise RuntimeError("the torch copy path copies demand experts only")
        gate_up = buffer[: store.gate_up_bytes].view(self.slot_gate_up.dtype).view(self.slot_gate_up.shape[1:])
        down = buffer[store.gate_up_bytes: store.expert_bytes].view(self.slot_down.dtype).view(self.slot_down.shape[1:])
        self.slot_gate_up[slot].copy_(gate_up, non_blocking=True)
        self.slot_down[slot].copy_(down, non_blocking=True)
    if runtime.cuda:
        event = store.events[index]
        if event is None:
            event = store.events[index] = torch.cuda.Event()
        event.record(torch.cuda.current_stream() if stream_handle is None else runtime.copy_stream)


OPTIONS: dict = {}
RUNTIMES: list = []


class NvmeRuntime(OffloadRuntime):
    """``OffloadRuntime`` with its experts also staged on the drive and every demand copy read from there."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nvme = NvmeExperts(self, OPTIONS["directory"], OPTIONS["buffers"])
        self.nvme_probes = [self.nvme.probe()]
        RUNTIMES.append(self)


def nvme_run_path(runtime, tokenizer, spec, workload, *, teacher_force, **kwargs):
    """``interleaved_freerun``'s free-running ``run_path``, with each sequence's reads and read time added to its row."""

    store = runtime.nvme
    read_ns, reads = store.read_ns, store.reads
    result = interleaved_freerun.free_running_run_path(runtime, tokenizer, spec, workload, teacher_force=teacher_force, **kwargs)
    for row in result["sequences"]:
        row["read_ms"] = (store.read_ns - read_ns) / 1e6
        row["reads"] = store.reads - reads
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nvme-dir", type=Path, required=True)
    parser.add_argument("--nvme-buffers", type=int, default=4)
    own, rest = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    for i, value in enumerate(rest):
        if (value == "--copy-repeat" and rest[i + 1] != "1") or (value.startswith("--copy-repeat=") and value != "--copy-repeat=1"):
            raise SystemExit("T8 reads every expert from the drive once: --copy-repeat must be 1")
    if own.nvme_buffers < 1:
        raise SystemExit("--nvme-buffers must be at least 1")
    OPTIONS.update(directory=own.nvme_dir, buffers=own.nvme_buffers)
    output, rest = interleaved_freerun.output_path(rest)
    partial = Path(str(output) + ".partial")
    saved = (interleaved_h100.run_path, interleaved_h100.OffloadRuntime, OffloadedMoeLayer.copy_expert)
    interleaved_h100.run_path, interleaved_h100.OffloadRuntime = nvme_run_path, NvmeRuntime
    OffloadedMoeLayer.copy_expert = nvme_copy_expert
    try:
        interleaved_h100.main(rest)
    finally:
        interleaved_h100.run_path, interleaved_h100.OffloadRuntime, OffloadedMoeLayer.copy_expert = saved
    runtime = RUNTIMES[-1]
    runtime.nvme_probes.append(runtime.nvme.probe(seed=1))
    out = json.loads(partial.read_text())
    out["decoding"] = "free_running"
    out["runner"] = "interleaved_nvme.py"
    out["nvme"] = dict(runtime.nvme.layout(), probes=runtime.nvme_probes,
                       arm_reads={name: sum(r.get("reads", 0) for r in a["sequences"]) for name, a in out["arms"].items()},
                       arm_read_ms={name: sum(r.get("read_ms", 0.0) for r in a["sequences"]) for name, a in out["arms"].items()})
    staging = Path(str(output) + ".tmp")
    staging.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(staging, output)
    partial.unlink()
    for fd in runtime.nvme.fds:
        os.close(fd)
    p0, p1 = runtime.nvme_probes[0], runtime.nvme_probes[-1]
    print(f"nvme: {p0['ms_mean']:.2f} ms/read before ({p0['gb_per_s']:.2f} GB/s), {p1['ms_mean']:.2f} after; repeated read "
          f"{p1['repeat_first_ms']:.2f} then {p1['repeat_second_ms']:.2f} ms; read-back equal {runtime.nvme.readback_equal}",
          flush=True)


if __name__ == "__main__":
    main()
