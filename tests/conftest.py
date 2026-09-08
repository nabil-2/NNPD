import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from settings import make_config


@pytest.fixture
def tiny(tmp_path):
    config = make_config("smoke")
    config["output"] = str(tmp_path / "results")
    config["cohort"]["priors"] = ["uniform"]
    config["data"] = {"train_size": 128, "validation_size": 64, "test_size": 64}
    config["model"]["hidden"] = [16, 16]
    config["training"].update(epochs=2, verbose=False, batch_size=30)
    config["inference"].update(truth_points_per_axis=3, candidate_count=64, diagnostic_cases=1)
    config["verification"].update(pairs=3, samples_per_endpoint=8, source_samples=16,
                                  target_samples=16, directions=2, marginal_samples=32,
                                  normalization_thetas=2)
    config["runtime"].update(device="cpu", cpu_threads=1)
    config["plots"] = []
    return config
