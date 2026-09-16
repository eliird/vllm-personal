# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree vLLM plugin: reload weights in place after ``sleep(level=2)``.

Background
----------
``Worker.sleep(level=2)`` discards the weights AND the KV cache with no CPU
backup; its ``wake_up`` restores only non-parameter buffers. The weight
*parameters* are left as empty memory (the CUDA virtual addresses are preserved
by the CuMem allocator). That is exactly what we want for a durable snapshot:
the process/context (and thus the compiled artifacts and CUDA graphs) can be
restored by CRIU + ``cuda-checkpoint`` at a tiny image size, then the weights are
reloaded from the model source.

This plugin wraps ``Worker.wake_up`` so that after a level-2 sleep it refills the
model parameters **in place** via the model loader's documented standalone API
``load_weights(model, model_config)``. Reloading in place preserves the tensor
addresses, so captured CUDA graphs stay valid.

It is registered through the standard ``vllm.general_plugins`` entry point and
also exposes a ``snapshot_cumem`` sleep-mode backend name; no vLLM core changes.
"""

from __future__ import annotations

import functools
import time

_LOG = "[vllm-snapshot-plugin]"


def _reload_weights(worker) -> None:
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader import get_model_loader

    runner = worker.model_runner
    cfg = runner.vllm_config
    loader = get_model_loader(cfg.load_config)
    t0 = time.perf_counter()
    # Loading weights needs the same ambient vLLM config that normal model
    # loading sets up (used by _init_ep_weight_filter and custom ops).
    with set_current_vllm_config(cfg):
        loader.load_weights(runner.model, cfg.model_config)
    torch.accelerator.synchronize()
    print(
        f"{_LOG} reloaded weights in place in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )


# Module-level so the backend factory can resolve it lazily by name.
from vllm.device_allocator.sleep_mode_backend import CuMemBackend


class SnapshotCuMemBackend(CuMemBackend):
    """CuMem backend whose level-2 wake reloads weights (via the patch)."""

    @classmethod
    def preserves_compiled_artifacts(cls) -> bool:
        return True

    @classmethod
    def supports_durable_storage(cls) -> bool:
        return True


def register() -> None:
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker.wake_up, "_vllm_snapshot_patched", False):
        return  # idempotent (plugins may be loaded more than once)

    orig_sleep = Worker.sleep
    orig_wake = Worker.wake_up

    @functools.wraps(orig_sleep)
    def sleep(self, level: int = 1):
        self._vllm_snapshot_sleep_level = level
        return orig_sleep(self, level)

    @functools.wraps(orig_wake)
    def wake_up(self, tags=None):
        level = getattr(self, "_vllm_snapshot_sleep_level", 1)
        result = orig_wake(self, tags)
        wake_weights = tags is None or "weights" in tags
        if level >= 2 and wake_weights:
            _reload_weights(self)
        return result

    sleep._vllm_snapshot_patched = True  # type: ignore[attr-defined]
    wake_up._vllm_snapshot_patched = True  # type: ignore[attr-defined]
    Worker.sleep = sleep
    Worker.wake_up = wake_up

    try:
        SleepModeBackendFactory.register_backend(
            "snapshot_cumem", __name__, "SnapshotCuMemBackend"
        )
    except ValueError:
        pass  # already registered

    print(f"{_LOG} registered: level-2 wake reloads weights in place", flush=True)
