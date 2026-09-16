#!/usr/bin/env python3
"""Torch equivalent of p0_cuda_min.cu for hosts without nvcc.

Allocates a device buffer, fills a known pattern, holds a device-side counter,
and writes "<counter> <checksum>" to a status file every second. Prints "ready"
once initialized so a cuda-checkpoint/CRIU round-trip can be verified.

Usage: p0_cuda_min_torch.py <status_file> [iters]
"""
import os
import sys
import time

import torch

N = 64 * 1024 * 1024  # 256 MiB of float32
SEED = 0x9E3779B9


def main() -> None:
    state = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p0_cuda/state"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    os.makedirs(os.path.dirname(state), exist_ok=True)

    idx = torch.arange(N, dtype=torch.int64, device="cuda")
    pattern = ((idx * SEED) % 1024).to(torch.float32) * 0.001
    ctr = torch.zeros(1, dtype=torch.int64, device="cuda")
    checksum = float(pattern.sum().item())

    with open(state, "w") as f:
        f.write(f"{int(ctr.item())} {checksum:.3f}\n")
    print(f"ready pid={os.getpid()} checksum={checksum:.3f}", flush=True)

    for _ in range(iters):
        ctr += 1
        cur = float(pattern.sum().item())
        with open(state, "w") as f:
            f.write(f"{int(ctr.item())} {cur:.3f}\n")
        time.sleep(1)


if __name__ == "__main__":
    main()
