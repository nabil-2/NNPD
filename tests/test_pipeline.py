from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from nnpd import Choice, Experiment, Metric, Plot, Product, execute, restore
from nnpd.core.runner import analyze, verify_store
from nnpd.nre.inference import normalized_mass
from nnpd.nre.training import predict_logits
from nnpd.results import runs, export_csv, comparison_plot
from gaussian import GaussianExperiment


def saved_references(path):
    record = json.loads((path / "artifacts.json").read_text())
    return {name: (path / relative).resolve() for name, relative in record["items"].items()}


def test_real_pipeline_all_four_priors_shared_artifacts_and_reload(tiny):
    tiny["cohort"]["priors"] = ["uniform", "normal", "exponential", "grid"]
    paths = execute(tiny, GaussianExperiment())
    assert len(paths) == 4
    references = [saved_references(path) for path in paths]
    for name in ("observations", "verification_observations"):
        assert len({ref[name] for ref in references}) == 1
    assert len({ref["candidates"] for ref in references}) == 2  # continuous, native lattice
    for path in paths:
        context = restore(path, GaussianExperiment())
        data = context.artifacts["training"]
        history = context.artifacts["model"].json("history")
        assert len(history["train_loss"]) == tiny["training"]["epochs"]
        assert history["rows_per_epoch"] == tiny["data"]["train_size"]
        assert np.isfinite(history["train_loss"]).all()
        logits = predict_logits(context.model, data.array("test_x"), data.array("test_theta"), 7, "cpu")
        cached = context.require("test_predictions").array("logits")
        np.testing.assert_allclose(logits, cached, atol=3e-7)
        metrics = json.loads((path / "metrics.json").read_text())
        assert set(metrics) == set(tiny["metrics"])
        assert 0 <= metrics["classification"]["auc"] <= 1
        assert metrics["posterior"]["enabled"]
        assert metrics["resolution"]["candidate_ess_min"] >= 1 - 1e-10
    assert verify_store(tiny["output"]) > 20


def model_size(context, dependencies):
    return sum(parameter.numel() for parameter in context.model.parameters())


def median_bias(context, dependencies):
    return float(np.median(np.abs(dependencies["inference"].array("ratio_mode") -
                                  dependencies["observations"].array("truth"))))


class ExtraMetrics(GaussianExperiment):
    def metrics(self):
        return {**super().metrics(), "model_size": Metric(model_size, ("model",)),
                "median_bias": Metric(median_bias, ("inference", "observations"))}


def test_adding_metrics_and_repeating_runs_do_not_retrain(tiny):
    paths = execute(tiny, GaussianExperiment())
    references = saved_references(paths[0])
    stamp = (references["model"] / "weights.pt").stat().st_mtime_ns
    execute(tiny, GaussianExperiment())
    assert (references["model"] / "weights.pt").stat().st_mtime_ns == stamp
    cfg = deepcopy(tiny)
    cfg["metrics"] += ["model_size", "median_bias"]
    new = execute(cfg, ExtraMetrics())[0]
    other = saved_references(new)
    for name in ("model", "training", "inference", "observations"):
        assert other[name] == references[name]
    assert json.loads((new / "metrics.json").read_text())["model_size"] > 0
    assert (references["model"] / "weights.pt").stat().st_mtime_ns == stamp


def test_learning_rate_sweep_reuses_data_and_observations_but_not_model(tiny):
    tiny["training"]["learning_rate"] = Choice([0.001, 0.01])
    paths = execute(tiny, GaussianExperiment())
    a, b = map(saved_references, paths)
    assert a["model"] != b["model"]
    for name in ("training", "observations", "candidates", "verification_observations"):
        assert a[name] == b[name]
    assert a["inference"] != b["inference"]


def test_saved_root_is_relocatable_and_restore_ignores_launcher_environment(tiny, tmp_path, monkeypatch):
    original = execute(tiny, GaussianExperiment())[0]
    destination = tmp_path / "moved"
    shutil.copytree(tiny["output"], destination)
    moved = destination / original.relative_to(tiny["output"])
    shutil.rmtree(tiny["output"])
    monkeypatch.setenv("RANK", "99")
    monkeypatch.setenv("WORLD_SIZE", "100")
    monkeypatch.setenv("LOCAL_RANK", "99")
    context = restore(moved, GaussianExperiment())
    assert context.runtime.world_size == 1
    assert context.require("inference").path.is_relative_to(destination)
    assert analyze(context)["classification"]["bce"] > 0


def test_train_then_analyze_and_no_implicit_training(tiny):
    with pytest.raises(FileNotFoundError, match="training data"):
        execute(tiny, GaussianExperiment(), stage="analyze")
    path = execute(tiny, GaussianExperiment(), stage="train")[0]
    assert not (path / "metrics.json").exists()
    assert json.loads((path / "status.json").read_text())["state"] == "trained"
    weights = saved_references(path)["model"] / "weights.pt"
    stamp = weights.stat().st_mtime_ns
    execute(tiny, GaussianExperiment(), stage="analyze")
    assert weights.stat().st_mtime_ns == stamp
    assert (path / "metrics.json").exists()


class FailsOnce(GaussianExperiment):
    fail = True
    def train(self, context, model):
        if self.fail:
            self.fail = False
            raise RuntimeError("injected training interruption")
        return super().train(context, model)


def test_failed_training_recovers_without_rebuilding_data(tiny):
    experiment = FailsOnce()
    with pytest.raises(RuntimeError, match="interruption"):
        execute(tiny, experiment)
    training = list((Path(tiny["output"]) / "cache" / "training").glob("*/artifact.json"))
    assert len(training) == 1
    path = execute(tiny, experiment)[0]
    assert json.loads((path / "status.json").read_text())["state"] == "complete"
    assert len(list((Path(tiny["output"]) / "cache" / "training").glob("*/artifact.json"))) == 1


class LikelihoodOracle(torch.nn.Module):
    """The x-only evidence term is omitted; it cancels in normalized inference."""
    def forward(self, features):
        x, theta = features[:, 0].double(), features[:, 1].double()
        return -0.5 * ((x - theta) / 2) ** 2 - np.log(2) - 0.5 * np.log(2 * np.pi)


class OracleExperiment(GaussianExperiment):
    def build_model(self, context):
        return LikelihoodOracle()
    def train(self, context, model):
        return {"oracle": True}


def test_end_to_end_analytic_oracle_and_prior_multiplied_once(tiny):
    tiny["cohort"]["priors"] = ["normal"]
    tiny["inference"].update(candidate_design="grid", candidate_points_per_axis=2001,
                              native_grid_prior=False, retain="full", truth_points_per_axis=3)
    tiny["metrics"] = ["bias", "coverage", "width", "posterior", "exact_reference"]
    context = restore(execute(tiny, OracleExperiment())[0], OracleExperiment())
    data, candidates = context.require("inference"), context.require("candidates")
    theta, measure = candidates.array("theta"), candidates.array("log_measure")
    for case in range(3):
        ratio, exact = data.array("ratio_log_score")[case], data.array("exact_log_score")[case]
        np.testing.assert_allclose(normalized_mass(ratio, measure), normalized_mass(exact, measure), atol=1e-13)
        posterior = data.array("posterior_log_score")[case]
        np.testing.assert_allclose(posterior - ratio, context.prior.log_prob(theta), atol=1e-12)
        assert not np.allclose(posterior - ratio, 15 * context.prior.log_prob(theta))
    np.testing.assert_allclose(data.array("ratio_mode"), data.array("exact_mode"), atol=0)


def test_parameter_set_with_std_really_trains_and_evaluates(tiny):
    tiny["problem"].update(dimension=2, infer=["mean:*", "std:1"])
    tiny["inference"].update(truth_points_per_axis=2, candidate_count=128)
    context = restore(execute(tiny, GaussianExperiment())[0], GaussianExperiment())
    assert context.problem.names == ["mean:0", "mean:1", "std:1"]
    assert context.artifacts["training"].array("train_theta").shape == (128, 3)
    assert next(context.model.parameters()).shape[1] == 5
    assert context.require("inference").array("ratio_mode").shape == (8, 3)
    assert (context.require("observations").array("truth")[:, 2] > 0).all()


def arbitrary_plot(context, dependencies):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    x = dependencies["observations"].array("observations")[0, :, 0]
    ax.hist(x)
    ax.set_title(f"{type(context.model).__name__}, prior={context.prior.measure}")
    return {"anything": fig}


class ExtraPlot(GaussianExperiment):
    def plots(self):
        return {**super().plots(), "custom": Plot(arbitrary_plot, ("observations", "model"))}


def test_plots_and_result_readers(tiny, tmp_path):
    tiny["plots"] = ["training", "prior", "inference", "pairwise", "reweighting", "showcase", "custom"]
    tiny["verification"]["showcase"]["samples"] = 64
    path = execute(tiny, ExtraPlot())[0]
    assert len(list((path / "plots").rglob("*.png"))) >= 8
    assert (path / "plots" / "custom" / "anything.png").stat().st_size > 1000
    records = runs(tiny["output"])
    assert len(records) == 1
    table = export_csv(tiny["output"], tmp_path / "summary.csv", {"bce": "metrics.classification.bce"})
    assert "bce" in table.read_text()
    fig = comparison_plot(records, "config.problem.dimension", "metrics.classification.auc")
    assert len(fig.axes) == 1
    from matplotlib import pyplot as plt
    plt.close(fig)


def test_guard_refuses_large_work_without_shrinking_science(tiny):
    tiny["inference"]["max_model_evaluations"] = 1
    with pytest.raises(ValueError, match="guards"):
        execute(tiny, GaussianExperiment())
    assert not Path(tiny["output"]).exists()


class IndependentExample(Experiment):
    """A second, non-Gaussian application proves that core has no Gaussian schema."""
    name = "independent-example"
    def make_problem(self, config):
        return {"n": config["sample_size"]}
    def make_prior(self, config, member, problem):
        return None
    def sample_training(self, context, writer):
        writer.array("x", np.arange(context.problem["n"]))
        return {"arbitrary": True}
    def build_model(self, context):
        return torch.nn.Linear(1, 1)
    def train(self, context, model):
        with torch.no_grad():
            model.weight.fill_(1)
            model.bias.zero_()
        return {"method": "explicit fixed model"}
    def metrics(self):
        return {"parameter_count": Metric(model_size, ("model",))}


def test_framework_has_no_gaussian_configuration_dependency(tiny):
    config = {"seed": 4, "sample_size": 10, "output": tiny["output"], "runtime": tiny["runtime"],
              "metrics": ["parameter_count"], "plots": [], "plotting": {"dpi": 100}}
    path = execute(config, IndependentExample())[0]
    assert json.loads((path / "metrics.json").read_text()) == {"parameter_count": 2}
