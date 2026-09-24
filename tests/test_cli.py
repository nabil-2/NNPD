import json
from pathlib import Path

import pytest

from nnpd import load_analysis, load_experiment
from nnpd.cli import main
from nnpd.results import runs


def test_cli_plan_is_safe_and_has_no_output_side_effects(tiny, capsys):
    main(["--settings-dir", str(tiny.write())])
    result = json.loads(capsys.readouterr().out)
    assert result["total_models"] == 1
    assert not tiny.output.exists()


def test_cli_plan_reports_infeasible_analysis_and_exits_with_error(tiny, capsys):
    tiny.analysis["inference"]["max_model_evaluations"] = 1
    with pytest.raises(SystemExit, match="(?s)not feasible.*baseline.*max_model_evaluations"):
        main(["--settings-dir", str(tiny.write())])
    assert json.loads(capsys.readouterr().out)["jobs"][0]["workload"]["errors"]
    assert not tiny.output.exists()


def test_unknown_truth_design_is_rejected(tiny):
    tiny.analysis["inference"]["truth_design"] = "auto"
    with pytest.raises(ValueError, match="sobol, grid, or size_adaptive"):
        main(["--settings-dir", str(tiny.write())])


def test_cli_train_analyze_verify_and_saved_settings_copies(tiny, capsys):
    tiny.analysis["analysis"] = "gaussian.extensions:ExtendedAnalysis"
    tiny.analysis["metrics"] += ["median_absolute_bias", "parameter_count"]
    tiny.analysis["plots"] = ["observation_histogram", "inference"]
    tiny.training["problem"]["dimension"] = 2
    tiny.training["problem"]["infer"] = ["mean:*", "std:1"]
    tiny.analysis["inference"]["truth_points_per_axis"] = 2
    args = ["--settings-dir", str(tiny.write())]
    main(["train", *args])
    assert runs(tiny.output) == []  # training alone is not a completed analysis
    main(["analyze", *args])
    record = runs(tiny.output)[0]
    assert record["analysis"]["settings"]["analysis"] == "gaussian.extensions:ExtendedAnalysis"
    assert record["metrics"]["parameter_count"]["trainable_parameters"] > 0
    assert record["metrics"]["median_absolute_bias"]["median_absolute"] >= 0
    figures = list((Path(record["path"]) / "plots").rglob("*.png"))
    assert any("projection" in item.name for item in figures)
    main(["verify", *args])
    assert "verified" in capsys.readouterr().out
    models = list((tiny.output / "cache" / "model").glob("*/weights.pt"))
    before = {item: item.stat().st_mtime_ns for item in models}
    main(["run", *args])
    assert before and all(item.stat().st_mtime_ns == stamp for item, stamp in before.items())
    # Both settings files are saved byte for byte and run the same experiment again.
    names = ("settings_training.py", "settings_analysis.py")
    for name in names:
        assert (tiny.output / name).read_bytes() == (tiny.folder / name).read_bytes()
    main(["analyze", "--settings-dir", str(tiny.output)])
    assert [item["path"] for item in runs(tiny.output)] == [record["path"]]
    for name in names:
        assert (tiny.output / name).read_bytes() == (tiny.folder / name).read_bytes()


def test_settings_must_name_the_right_classes():
    with pytest.raises(TypeError, match="subclass of Experiment"):
        load_experiment("builtins:dict")
    with pytest.raises(TypeError, match="subclass of Analysis"):
        load_analysis("gaussian:GaussianExperiment")
