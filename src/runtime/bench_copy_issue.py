"""How much decode-thread CPU time does issuing one expert copy cost?

An OLMoE expert is two pinned host tensors (gate_up 2I x H, down H x I, bf16), so one expert copy is
two host-to-device copies. This script measures, on the GPU host, the CPU time to *issue* expert
copies (no wait for completion) with (a) ``Tensor.copy_(non_blocking=True)`` and (b) the CUDA
driver's ``cuMemcpyHtoDAsync`` through ctypes on torch's current stream. It also checks that (b)
copies the same bytes.

    python src/runtime/bench_copy_issue.py
"""

from __future__ import annotations

import ctypes
import time

import torch


def main() -> None:
    hidden, inter, n = 2048, 1024, 64
    dtype = torch.bfloat16
    host_gu = torch.randn(n, 2 * inter, hidden, dtype=dtype).pin_memory()
    host_d = torch.randn(n, hidden, inter, dtype=dtype).pin_memory()
    dev_gu = torch.empty(n, 2 * inter, hidden, dtype=dtype, device="cuda")
    dev_d = torch.empty(n, hidden, inter, dtype=dtype, device="cuda")
    torch.cuda.synchronize()

    def issue_torch(experts):
        for e in experts:
            dev_gu[e].copy_(host_gu[e], non_blocking=True)
            dev_d[e].copy_(host_d[e], non_blocking=True)

    cuda = ctypes.CDLL("libcuda.so.1")
    memcpy = cuda.cuMemcpyHtoDAsync_v2
    memcpy.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    memcpy.restype = ctypes.c_int
    gu_bytes = host_gu[0].numel() * host_gu.element_size()
    d_bytes = host_d[0].numel() * host_d.element_size()
    hgu, hd, dgu, dd = host_gu.data_ptr(), host_d.data_ptr(), dev_gu.data_ptr(), dev_d.data_ptr()

    def issue_driver(experts):
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        for e in experts:
            rc = memcpy(dgu + e * gu_bytes, ctypes.c_void_p(hgu + e * gu_bytes), gu_bytes, stream)
            rc |= memcpy(dd + e * d_bytes, ctypes.c_void_p(hd + e * d_bytes), d_bytes, stream)
            if rc:
                raise RuntimeError(f"cuMemcpyHtoDAsync failed: {rc}")

    for name, fn in (("torch copy_", issue_torch), ("cuMemcpyHtoDAsync", issue_driver)):
        times = []
        for rep in range(30):
            experts = [(rep * 7 + i) % n for i in range(4)]
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            fn(experts)
            t1 = time.perf_counter_ns()
            torch.cuda.synchronize()
            times.append((t1 - t0) / 1e3 / len(experts))
        times.sort()
        print(f"{name:20s} issue cost per expert (2 copies): median {times[len(times) // 2]:.1f} us, "
              f"min {times[0]:.1f} us", flush=True)
    dev_gu.zero_()
    dev_d.zero_()
    issue_driver(list(range(n)))
    torch.cuda.synchronize()
    ok = torch.equal(dev_gu.cpu(), host_gu) and torch.equal(dev_d.cpu(), host_d)
    print(f"driver copies equal the host tensors: {ok}", flush=True)


if __name__ == "__main__":
    main()
