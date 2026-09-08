"""CPU, single CUDA device, independent jobs, or torchrun/DDP execution."""
from __future__ import annotations

from datetime import timedelta
import os
import random

import numpy as np
import torch
import torch.distributed as dist


class Runtime:
    def __init__(self, config: dict, *, initialize: bool = True):
        self.deterministic = bool(config["deterministic"])
        self.cpu_threads = int(config["cpu_threads"])
        if self.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.rank = int(os.environ.get("RANK", "0")) if initialize else 0
        self.world_size = int(os.environ.get("WORLD_SIZE", "1")) if initialize else 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0")) if initialize else 0
        self.parallel = config["parallel"]
        if self.parallel not in {"jobs", "ddp"}:
            raise ValueError("runtime.parallel must be 'jobs' or 'ddp'.")
        requested = config["device"]
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported devices are auto, cpu, cuda, and cuda:N.")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested, but this PyTorch installation has no usable CUDA GPU.")
            if self.world_size > 1 and self.device.index is not None:
                raise ValueError("Use device='cuda' with torchrun; select devices with CUDA_VISIBLE_DEVICES.")
            index = self.local_rank if self.world_size > 1 else (self.device.index or 0)
            if index >= torch.cuda.device_count():
                raise ValueError("More local workers than visible CUDA devices.")
            self.device = torch.device("cuda", index)
            torch.cuda.set_device(self.device)
        self.ddp = self.parallel == "ddp" and self.world_size > 1
        self.leader = not self.ddp or self.rank == 0
        torch.set_num_threads(int(config["cpu_threads"]))
        torch.use_deterministic_algorithms(bool(config["deterministic"]))
        self._owns_group = False
        if initialize and self.ddp:
            if dist.is_initialized():
                raise RuntimeError("A process group is already active; do not nest execute() calls.")
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo",
                                    timeout=timedelta(minutes=config["timeout_minutes"]))
            self._owns_group = True

    @property
    def training_signature(self) -> dict:
        # Numerical backends/world sizes can change training, even with identical seeds.
        return {"device_type": self.device.type,
                "world_size": self.world_size if self.ddp else 1,
                "torch": str(torch.__version__), "cpu_threads": self.cpu_threads,
                "deterministic": self.deterministic,
                "cuda": torch.version.cuda if self.device.type == "cuda" else None,
                "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"}

    def broadcast(self, value):
        if self.ddp:
            values = [value]
            dist.broadcast_object_list(values, src=0, device=self.device)
            return values[0]
        return value

    def barrier(self) -> None:
        if self.ddp:
            dist.barrier()

    def close(self) -> None:
        if self._owns_group:
            dist.destroy_process_group()
            self._owns_group = False


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
