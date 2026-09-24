from pathlib import Path

import pytest
from nnpd import Choice, expand_sweep, plan
from nnpd.core.runner import expand_jobs
from nnpd.core.settings import at, canonical, load_settings
from gaussian import GaussianExperiment
from settings_analysis import make_settings as analysis_settings
from settings_training import make_settings as training_settings

ROOT = Path(__file__).resolve().parents[1]


def test_user_example_is_five_runs_not_nine():
    settings = {"training_size": Choice([50_000, 100_000, 150_000]),
                "hyperparameter": Choice([0.1, 0.01, 0.001])}
    values = [v.settings for v in expand_sweep(settings)]
    assert values == [{"training_size": 50_000, "hyperparameter": 0.1},
                      {"training_size": 100_000, "hyperparameter": 0.1},
                      {"training_size": 150_000, "hyperparameter": 0.1},
                      {"training_size": 50_000, "hyperparameter": 0.01},
                      {"training_size": 50_000, "hyperparameter": 0.001}]


def test_lists_are_literal_and_parameter_sets_are_atomic():
    settings = {"widths": [64, 64, 64, 64],
                "infer": Choice([["mean:*"], ["mean:*", "std:1"]]),
                "optimizer": {"lr": Choice([0.1, 0.01, 0.1, 0.01])}}
    values = expand_sweep(settings)
    assert len(values) == 3
    assert values[1].settings["infer"] == ["mean:*", "std:1"]
    assert values[2].settings["infer"] == ["mean:*"]
    assert all(v.settings["widths"] == [64] * 4 for v in values)
    values[1].settings["widths"].append(128)
    assert len(settings["widths"]) == 4
    assert at(values[2].settings, "optimizer.lr") == 0.01


def test_atomic_dictionary_and_identity_ignore_dictionary_order():
    settings = {"model": Choice([{"kind": "mlp", "width": 64}, {"width": 64, "kind": "mlp"},
                                 {"kind": "mlp", "width": 128}])}
    assert len(expand_sweep(settings)) == 2
    assert canonical({"a": 1, "b": 2}) == canonical({"b": 2, "a": 1})


@pytest.mark.parametrize("value", [[], (), "not-a-list"])
def test_invalid_choice(value):
    with pytest.raises((ValueError, TypeError)):
        Choice(value)


@pytest.mark.parametrize("settings", [{"bad.key": 2}, {"x": [Choice([1, 2])]}, {"x": float("nan")}, []])
def test_invalid_sweep(settings):
    with pytest.raises((ValueError, TypeError)):
        expand_sweep(settings)


def test_all_variants_validated_before_first_training(tiny):
    tiny.training["problem"]["infer"] = Choice([["mean:*"], ["mean:*", "std:1"]])
    with pytest.raises(ValueError, match="in-range"):
        tiny.plan()  # std:1 does not exist in 1D
    assert not tiny.output.exists()


def test_analysis_settings_cannot_sweep(tiny):
    tiny.analysis["inference"]["candidate_count"] = Choice([64, 128])
    with pytest.raises(ValueError, match="cannot contain Choice"):
        tiny.plan()
    with pytest.raises(ValueError, match="cannot contain Choice"):
        tiny.execute("analyze")


def test_default_configuration_is_explicit():
    jobs = plan("default", ROOT)
    assert len(jobs) == 24  # 6 configurations, fixed 4-prior cohort
    assert [jobs[i].settings["problem"]["dimension"] for i in range(0, 24, 4)] == list(range(1, 7))
    base = jobs[0].settings
    assert sum(base["data"].values()) == 100_000
    assert base["model"]["hidden"] == [64] * 4
    assert base["training"]["epochs"] == 5
    assert analysis_settings("default")["inference"]["observations"] == 15
    # Sobol truths by default: 1,024 at one inferred parameter, doubling per additional one.
    assert [jobs[i].workload["inference_cases"] for i in range(0, 24, 4)] == [1024 * 2 ** k for k in range(6)]
    assert {job.workload["truth_design"] for job in jobs} == {"sobol"}
    paper = plan("paper", ROOT)
    # size_adaptive: the exact 25**p grid while it fits max_grid_truths, Sobol above.
    assert [job.workload["truth_design"] for job in paper[::4]] == ["grid"] * 4 + ["sobol"] * 2
    assert paper[12].workload["inference_cases"] == 25 ** 4
    assert paper[-1].workload["inference_cases"] == 1024 * 2 ** 5
    assert not any(job.workload["errors"] for job in [*jobs, *paper])  # every dimension can be analyzed
    assert paper[0].settings["problem"]["parameters"]["mean"]["exponential_rate"] == -0.1
    assert base["problem"]["parameters"]["mean"]["exponential_rate"] == 0.1


def test_network_and_dependency_options_are_ofat():
    settings = training_settings("smoke")
    settings["problem"]["dimension"] = 2
    settings["problem"]["infer"] = Choice([["mean:*"], ["mean:*", "std:1"]])
    settings["model"]["kind"] = Choice(["mlp", "residual"])
    settings["training"]["learning_rate"] = Choice([0.001, 0.01])
    settings["cohort"]["priors"] = ["uniform", "normal"]
    jobs = expand_jobs(settings, GaussianExperiment())
    assert len(jobs) == 8  # 4 scientific configurations, 2-prior cohort
    assert {job.workload["network_inputs"] for job in jobs} == {4, 5}
    for job in jobs:
        if job.settings["model"]["kind"] == "residual":
            assert job.settings["problem"]["infer"] == ["mean:*"]


def test_load_settings_and_unknown_profile(tmp_path):
    path = tmp_path / "settings_training.py"
    path.write_text("DEFAULT_PROFILE = 'a'\ndef make_settings(profile):\n    return {'profile': profile}\n")
    assert load_settings(path) == {"profile": "a"}
    assert load_settings(path, "b") == {"profile": "b"}
    for make_settings in (training_settings, analysis_settings):
        with pytest.raises(ValueError, match="Unknown profile"):
            make_settings("missing")


@pytest.mark.parametrize("kind,section,key,value", [
    ("training", "data", "train_size", 127), ("training", "training", "epochs", 0),
    ("training", "training", "learning_rate", -0.1), ("training", "training", "precision", "bad"),
    ("training", "model", "dropout", 1), ("analysis", "inference", "levels", [68, 95]),
    ("analysis", "inference", "candidate_count", 0), ("analysis", "inference", "truth_margin_fraction", 6),
])
def test_bad_knobs_fail_early(tiny, kind, section, key, value):
    getattr(tiny, kind)[section][key] = value
    with pytest.raises(ValueError):
        tiny.plan()
