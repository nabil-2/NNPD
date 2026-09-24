from dataclasses import dataclass
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from nnpd import execute, plan
from settings_analysis import make_settings as analysis_settings
from settings_training import make_settings as training_settings


def write_settings(folder: Path, training: dict, analysis: dict) -> Path:
    """Both settings files, as a user would write them, with a single profile named 'tiny'."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings_training.py").write_text(
        f"from nnpd import Choice\n\nDEFAULT_PROFILE = 'tiny'\nSETTINGS = {training!r}\n\n\n"
        "def make_settings(profile=DEFAULT_PROFILE):\n    return SETTINGS\n")
    (folder / "settings_analysis.py").write_text(
        f"from nnpd import Choice\n\nSETTINGS = {analysis!r}\n\n\ndef make_settings(profile):\n    return SETTINGS\n")
    return folder


@dataclass
class Setup:
    """Editable training and analysis settings, launched through the real settings files."""
    training: dict
    analysis: dict
    folder: Path

    @property
    def output(self) -> Path:
        return Path(self.training["output"])

    def write(self) -> Path:
        return write_settings(self.folder, self.training, self.analysis)

    def execute(self, stage="run"):
        return execute(stage, settings_dir=self.write())

    def plan(self):
        return plan(settings_dir=self.write())


@pytest.fixture
def tiny(tmp_path):
    training, analysis = training_settings("smoke"), analysis_settings("smoke")
    training["output"] = str(tmp_path / "results")
    training["cohort"]["priors"] = ["uniform"]
    training["data"] = {"train_size": 128, "validation_size": 64, "test_size": 64}
    training["model"]["hidden"] = [16, 16]
    training["training"].update(epochs=2, verbose=False, batch_size=30)
    training["runtime"].update(device="cpu", cpu_threads=1)
    analysis["inference"].update(truth_points_per_axis=3, candidate_count=64, diagnostic_cases=1)
    analysis["verification"].update(pairs=3, samples_per_endpoint=8, source_samples=16,
                                    target_samples=16, directions=2, marginal_samples=32,
                                    normalization_thetas=2)
    analysis["plots"] = []
    return Setup(training, analysis, tmp_path / "settings")
