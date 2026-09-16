#!/usr/bin/env python3
"""Phase 3 CUDA device-memory sweep probe (torch fallback for p3_cuda_sweep.cu).

Allocates <MB> MiB of device memory, fills a fixed pattern, holds a device-side
counter, and writes "<counter> <checksum>" to a status file each second. The
pattern is built in small chunks and the cache is emptied so the process's
device footprint is ~<MB> (important: cuda-checkpoint copies all device memory).

Usage: p3_cuda_sweep_torch.py <MB> <status_file> [iters]
"""
import os
import sys
import time

import torch

SEED = 0x9E3779B9
CHUNK = 1 << 24


def main() -> None:
    mb = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    state = sys.argv[2] if len(sys.argv) > 2 else "/tmp/p3_cuda/state"
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    os.makedirs(os.path.dirname(state), exist_ok=True)

    n = mb * 1024 * 1024 // 4
    pattern = torch.empty(n, dtype=torch.float32, device="cuda")
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        idx = torch.arange(s, e, dtype=torch.int64, device="cuda")
        pattern[s:e] = ((idx * SEED) % 1024).to(torch.float32) * 0.001
        del idx
    torch.cuda.empty_cache()

    ctr = torch.zeros(1, dtype=torch.int64, device="cuda")
    checksum = float(pattern.sum().item())

    with open(state, "w") as f:
        f.write(f"{int(ctr.item())} {checksum:.3f}\n")
    print(f"ready pid={os.getpid()} mb={mb} checksum={checksum:.3f}", flush=True)

    for _ in range(iters):
        ctr += 1
        cur = float(pattern.sum().item())
        with open(state, "w") as f:
            f.write(f"{int(ctr.item())} {cur:.3f}\n")
        time.sleep(1)


if __name__ == "__main__":
    main()
