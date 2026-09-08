from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from nnpd import execute, restore
from nnpd.results import runs
from gaussian import GaussianExperiment

ROOT = Path(__file__).resolve().parents[1]


def launch(config, tmp_path, cuda=False):
    settings = tmp_path / "worker_config.json"
    settings.write_text(json.dumps(config))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
    if not cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc-per-node=2", str(ROOT / "tests" / "distributed_worker.py"), str(settings)]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    (tmp_path / "torchrun.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.distributed
def test_two_cpu_ddp_matches_single_global_batches_including_empty_last_shard(tiny, tmp_path):
    tiny["training"].update(batch_size=127, epochs=2)  # 128 rows => last global batch has ONE row
    tiny["metrics"] = ["classification", "bias", "coverage", "width"]
    single = restore(execute(tiny, GaussianExperiment())[0], GaussianExperiment())
    distributed = deepcopy(tiny)
    distributed["output"] = str(tmp_path / "ddp")
    distributed["runtime"]["parallel"] = "ddp"
    launch(distributed, tmp_path)
    records = runs(distributed["output"])
    assert len(records) == 1
    ddp = restore(records[0]["path"], GaussianExperiment())
    a, b = single.artifacts["model"].state_dict(), ddp.artifacts["model"].state_dict()
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=2e-4, atol=2e-6)
    history = ddp.artifacts["model"].json("history")
    assert history["world_size"] == 2
    assert history["global_batch_size"] == 127 and history["rows_per_epoch"] == 128
    np.testing.assert_allclose(single.artifacts["model"].json("history")["train_loss"],
                               history["train_loss"], atol=1e-6)
    worker1 = json.loads((Path(distributed["output"]) / "worker-1.json").read_text())
    assert worker1["completed"] == []  # only rank zero analyzes and writes each DDP run


@pytest.mark.distributed
def test_two_cpu_workers_process_distinct_full_jobs_and_share_observations(tiny, tmp_path):
    tiny["cohort"]["priors"] = ["uniform", "normal"]
    tiny["runtime"]["parallel"] = "jobs"
    launch(tiny, tmp_path)
    records = runs(tiny["output"])
    assert len(records) == 2
    owners = [json.loads((Path(tiny["output"]) / f"worker-{rank}.json").read_text())["completed"]
              for rank in range(2)]
    assert len(owners[0]) == len(owners[1]) == 1
    assert set(owners[0]).isdisjoint(owners[1])
    observations = list((Path(tiny["output"]) / "cache" / "observations").glob("*/artifact.json"))
    assert len(observations) == 1
    # A second identical distributed launch exercises the cached-model DDP/jobs path.
    before = {path: path.stat().st_mtime_ns for path in (Path(tiny["output"]) / "cache" / "model").glob("*/weights.pt")}
    launch(tiny, tmp_path)
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in before.items())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available")
@pytest.mark.parametrize("precision", ["off", "float16"])
def test_single_cuda_training_and_inference(tiny, precision):
    tiny["runtime"]["device"] = "cuda"
    tiny["training"]["precision"] = precision
    path = execute(tiny, GaussianExperiment())[0]
    context = restore(path, GaussianExperiment(), device="cuda")
    assert next(context.model.parameters()).is_cuda
    assert np.isfinite(context.require("inference").array("ratio_mode")).all()


@pytest.mark.cuda
@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices are required")
def test_two_cuda_nccl_training_and_inference(tiny, tmp_path):
    tiny["runtime"].update(device="cuda", parallel="ddp")
    launch(tiny, tmp_path, cuda=True)
    record = runs(tiny["output"])[0]
    assert record["backend"]["world_size"] == 2
    assert record["backend"]["device_type"] == "cuda"
