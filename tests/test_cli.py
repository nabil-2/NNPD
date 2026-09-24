import json
from pathlib import Path

import pytest

from nnpd import load_experiment
from nnpd.cli import main
from nnpd.results import runs

ROOT = Path(__file__).resolve().parents[1]


def write_settings(config, tmp_path):
    path = tmp_path / "settings.py"
    path.write_text(f"DEFAULT_PROFILE = 'tiny'\ndef make_config(profile='tiny'):\n    return {config!r}\n")
    return path


def test_cli_plan_is_safe_and_has_no_output_side_effects(tmp_path, tiny, capsys):
    path = write_settings(tiny, tmp_path)
    main(["--config", str(path)])
    result = json.loads(capsys.readouterr().out)
    assert result["total_models"] == 1
    assert not Path(tiny["output"]).exists()


def test_cli_plan_reports_infeasible_analysis_and_exits_with_error(tmp_path, tiny, capsys):
    tiny["inference"]["max_model_evaluations"] = 1
    path = write_settings(tiny, tmp_path)
    with pytest.raises(SystemExit, match="(?s)not feasible.*baseline.*max_model_evaluations"):
        main(["--config", str(path)])
    assert json.loads(capsys.readouterr().out)["jobs"][0]["workload"]["errors"]
    assert not Path(tiny["output"]).exists()


def test_unknown_truth_design_is_rejected(tmp_path, tiny):
    tiny["inference"]["truth_design"] = "auto"
    with pytest.raises(ValueError, match="sobol, grid, or size_adaptive"):
        main(["--config", str(write_settings(tiny, tmp_path))])


def test_cli_train_analyze_verify_and_saved_settings_copy(tiny, tmp_path, capsys):
    tiny["application"] = "gaussian.extensions:ExtendedGaussian"
    tiny["metrics"] += ["median_absolute_bias", "parameter_count"]
    tiny["plots"] = ["observation_histogram", "inference"]
    tiny["problem"]["dimension"] = 2
    tiny["problem"]["infer"] = ["mean:*", "std:1"]
    tiny["inference"]["truth_points_per_axis"] = 2
    path = write_settings(tiny, tmp_path)
    args = ["--config", str(path)]
    main(["train", *args])
    assert runs(tiny["output"]) == []  # training alone is not a completed analysis
    main(["analyze", *args])
    record = runs(tiny["output"])[0]
    assert record["config"]["application"] == "gaussian.extensions:ExtendedGaussian"
    assert record["metrics"]["parameter_count"]["trainable_parameters"] > 0
    assert record["metrics"]["median_absolute_bias"]["median_absolute"] >= 0
    figures = list((Path(record["path"]) / "plots").rglob("*.png"))
    assert any("projection" in item.name for item in figures)
    main(["verify", *args])
    assert "verified" in capsys.readouterr().out
    models = list((Path(tiny["output"]) / "cache" / "model").glob("*/weights.pt"))
    before = {item: item.stat().st_mtime_ns for item in models}
    main(["run", *args])
    assert before and all(item.stat().st_mtime_ns == stamp for item, stamp in before.items())
    # The settings file is saved byte for byte and runs the same experiment again.
    saved = Path(tiny["output"]) / "settings.py"
    assert saved.read_bytes() == path.read_bytes()
    main(["analyze", "--config", str(saved)])
    assert [item["path"] for item in runs(tiny["output"])] == [record["path"]]
    assert saved.read_bytes() == path.read_bytes()


def test_application_factory_rejects_non_experiments():
    with pytest.raises(TypeError, match="subclass Experiment"):
        load_experiment("builtins:dict")
