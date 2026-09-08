from copy import deepcopy

import pytest
from nnpd import Choice, expand_sweep, plan
from nnpd.core.config import at, canonical, load_settings
from gaussian import GaussianExperiment
from settings import make_config


def test_user_example_is_five_runs_not_nine():
    cfg = {"training_size": Choice([50_000, 100_000, 150_000]),
           "hyperparameter": Choice([0.1, 0.01, 0.001])}
    values = [v.config for v in expand_sweep(cfg)]
    assert values == [{"training_size": 50_000, "hyperparameter": 0.1},
                      {"training_size": 100_000, "hyperparameter": 0.1},
                      {"training_size": 150_000, "hyperparameter": 0.1},
                      {"training_size": 50_000, "hyperparameter": 0.01},
                      {"training_size": 50_000, "hyperparameter": 0.001}]


def test_lists_are_literal_and_parameter_sets_are_atomic():
    cfg = {"widths": [64, 64, 64, 64],
           "infer": Choice([["mean:*"], ["mean:*", "std:1"]]),
           "optimizer": {"lr": Choice([0.1, 0.01, 0.1, 0.01])}}
    values = expand_sweep(cfg)
    assert len(values) == 3
    assert values[1].config["infer"] == ["mean:*", "std:1"]
    assert values[2].config["infer"] == ["mean:*"]
    assert all(v.config["widths"] == [64] * 4 for v in values)
    values[1].config["widths"].append(128)
    assert len(cfg["widths"]) == 4
    assert at(values[2].config, "optimizer.lr") == 0.01


def test_atomic_dictionary_and_identity_ignore_dictionary_order():
    cfg = {"model": Choice([{"kind": "mlp", "width": 64}, {"width": 64, "kind": "mlp"},
                            {"kind": "mlp", "width": 128}])}
    assert len(expand_sweep(cfg)) == 2
    assert canonical({"a": 1, "b": 2}) == canonical({"b": 2, "a": 1})


@pytest.mark.parametrize("value", [[], (), "not-a-list"])
def test_invalid_choice(value):
    with pytest.raises((ValueError, TypeError)):
        Choice(value)


@pytest.mark.parametrize("config", [{"bad.key": 2}, {"x": [Choice([1, 2])]}, {"x": float("nan")}, []])
def test_invalid_sweep(config):
    with pytest.raises((ValueError, TypeError)):
        expand_sweep(config)


def test_all_variants_validated_before_first_training(tiny):
    tiny["problem"]["infer"] = Choice([["mean:*"], ["mean:*", "std:1"]])
    with pytest.raises(ValueError, match="in-range"):
        plan(tiny, GaussianExperiment())  # std:1 does not exist in 1D
    assert not __import__("pathlib").Path(tiny["output"]).exists()


def test_legacy_configuration_is_explicit():
    jobs = plan(make_config("legacy"), GaussianExperiment())
    assert len(jobs) == 24  # 6 configurations, fixed 4-prior cohort
    assert [jobs[i].config["problem"]["dimension"] for i in range(0, 24, 4)] == list(range(1, 7))
    base = jobs[0].config
    assert sum(base["data"].values()) == 100_000
    assert base["model"]["hidden"] == [64] * 4
    assert base["training"]["epochs"] == 5
    assert base["inference"]["observations"] == 15
    assert jobs[-1].workload["truth_design"] == "sobol"
    paper = plan(make_config("paper"), GaussianExperiment())
    assert paper[-1].workload["inference_cases"] == 25 ** 6
    assert paper[-1].workload["errors"]  # refuses the huge literal analysis allocation
    assert paper[0].config["problem"]["parameters"]["mean"]["exponential_rate"] == -0.1
    assert base["problem"]["parameters"]["mean"]["exponential_rate"] == 0.1


def test_network_and_dependency_options_are_ofat():
    jobs = plan(make_config("mixed"), GaussianExperiment())
    assert len(jobs) == 8  # 4 scientific configurations, 2-prior cohort
    assert {job.workload["network_inputs"] for job in jobs} == {4, 5}
    for job in jobs:
        if job.config["model"]["kind"] == "residual":
            assert job.config["problem"]["infer"] == ["mean:*"]


def test_load_settings_and_unknown_profile(tmp_path):
    path = tmp_path / "config.py"
    path.write_text("DEFAULT_PROFILE = 'a'\ndef make_config(profile):\n    return {'profile': profile}\n")
    assert load_settings(path) == {"profile": "a"}
    assert load_settings(path, "b") == {"profile": "b"}
    with pytest.raises(ValueError, match="Unknown profile"):
        make_config("missing")


@pytest.mark.parametrize("section,key,value", [
    ("data", "train_size", 127), ("training", "epochs", 0),
    ("training", "learning_rate", -0.1), ("training", "precision", "bad"),
    ("model", "dropout", 1), ("inference", "levels", [68, 95]),
    ("inference", "candidate_count", 0), ("inference", "truth_margin_fraction", 6),
])
def test_bad_knobs_fail_early(tiny, section, key, value):
    cfg = deepcopy(tiny)
    cfg[section][key] = value
    with pytest.raises(ValueError):
        plan(cfg, GaussianExperiment())
