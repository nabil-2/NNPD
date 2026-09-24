import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from nnpd import Analysis, Choice, Experiment, Metric, Plot, restore
from nnpd.core.runner import analyze_run, verify_store
from nnpd.nre.inference import normalized_mass
from nnpd.nre.training import predict_logits
from nnpd.results import runs, export_csv, comparison_plot
from gaussian import GaussianAnalysis, GaussianExperiment


def saved_references(path):
    record = json.loads((path / "artifacts.json").read_text())
    return {name: (path / relative).resolve() for name, relative in record["items"].items()}


def test_real_pipeline_all_four_priors_shared_artifacts_and_reload(tiny):
    tiny.training["cohort"]["priors"] = ["uniform", "normal", "exponential", "grid"]
    paths = tiny.execute()
    assert len(paths) == 4
    references = [saved_references(path) for path in paths]
    for name in ("observations", "verification_observations"):
        assert len({ref[name] for ref in references}) == 1
    assert len({ref["candidates"] for ref in references}) == 2  # continuous, native lattice
    for path in paths:
        context = restore(path)
        data = context.artifacts["training"]
        history = context.artifacts["model"].json("history")
        assert len(history["train_loss"]) == tiny.training["training"]["epochs"]
        assert history["rows_per_epoch"] == tiny.training["data"]["train_size"]
        assert np.isfinite(history["train_loss"]).all()
        logits = predict_logits(context.model, data.array("test_x"), data.array("test_theta"), 7, "cpu")
        cached = context.require("test_predictions").array("logits")
        np.testing.assert_allclose(logits, cached, atol=3e-7)
        metrics = json.loads((path / "metrics.json").read_text())
        assert set(metrics) == set(tiny.analysis["metrics"])
        assert 0 <= metrics["classification"]["auc"] <= 1
        assert metrics["posterior"]["enabled"]
        assert metrics["resolution"]["candidate_ess_min"] >= 1 - 1e-10
    assert verify_store(tiny.output) > 20


def model_size(context, dependencies):
    return sum(parameter.numel() for parameter in context.model.parameters())


def median_bias(context, dependencies):
    return float(np.median(np.abs(dependencies["inference"].array("ratio_mode") -
                                  dependencies["observations"].array("truth"))))


class ExtraMetrics(GaussianAnalysis):
    def metrics(self):
        return {**super().metrics(), "model_size": Metric(model_size, ("model",)),
                "median_bias": Metric(median_bias, ("inference", "observations"))}


def test_adding_metrics_and_repeating_runs_do_not_retrain(tiny):
    paths = tiny.execute()
    references = saved_references(paths[0])
    stamp = (references["model"] / "weights.pt").stat().st_mtime_ns
    tiny.execute()
    assert (references["model"] / "weights.pt").stat().st_mtime_ns == stamp
    tiny.analysis["analysis"] = "test_pipeline:ExtraMetrics"
    tiny.analysis["metrics"] += ["model_size", "median_bias"]
    assert tiny.execute("analyze") == paths  # the same run: its analysis is replaced
    other = saved_references(paths[0])
    for name in ("model", "training", "inference", "observations"):
        assert other[name] == references[name]
    assert json.loads((paths[0] / "metrics.json").read_text())["model_size"] > 0
    assert (references["model"] / "weights.pt").stat().st_mtime_ns == stamp


def test_changed_analysis_replaces_results_and_ignores_later_training_edits(tiny):
    paths = tiny.execute()
    stamp = (saved_references(paths[0])["model"] / "weights.pt").stat().st_mtime_ns
    saved_training = (tiny.output / "settings_training.py").read_bytes()
    tiny.analysis["metrics"] = ["classification"]
    tiny.training["training"]["learning_rate"] = 0.5  # not trained: analyze uses the saved models
    assert tiny.execute("analyze") == paths
    assert set(json.loads((paths[0] / "metrics.json").read_text())) == {"classification"}
    assert not (paths[0] / "metrics" / "bias.json").exists()
    assert "inference" not in saved_references(paths[0])
    assert (saved_references(paths[0])["model"] / "weights.pt").stat().st_mtime_ns == stamp
    assert (tiny.output / "settings_training.py").read_bytes() == saved_training
    assert (tiny.output / "settings_analysis.py").read_bytes() == (tiny.folder / "settings_analysis.py").read_bytes()
    assert runs(tiny.output)[0]["analysis"]["settings"]["metrics"] == ["classification"]


def test_learning_rate_sweep_reuses_data_and_observations_but_not_model(tiny):
    tiny.training["training"]["learning_rate"] = Choice([0.001, 0.01])
    paths = tiny.execute()
    a, b = map(saved_references, paths)
    assert a["model"] != b["model"]
    for name in ("training", "observations", "candidates", "verification_observations"):
        assert a[name] == b[name]
    assert a["inference"] != b["inference"]


def test_latest_settings_win_and_the_cache_keeps_the_last_three(tiny):
    root = tiny.output
    models = []
    for rate in (0.001, 0.002, 0.003, 0.004):
        tiny.training["training"]["learning_rate"] = rate
        paths = tiny.execute()
        models.append(saved_references(paths[0])["model"])
        assert [Path(record["path"]) for record in runs(root)] == paths  # only the latest settings' runs
        for name in ("settings_training.py", "settings_analysis.py"):
            assert (root / name).read_bytes() == (tiny.folder / name).read_bytes()
    assert not models[0].exists() and all(model.exists() for model in models[1:])
    assert len(list((root / "cache" / "model").iterdir())) == 3
    assert len(json.loads((root / "history.json").read_text())) == 3
    assert saved_references(paths[0])["training"].exists()  # shared by all settings, so kept


def test_same_training_again_keeps_its_analysis_data(tiny):
    tiny.training["training"]["learning_rate"] = Choice([0.001, 0.01])
    tiny.execute("train")
    paths = tiny.execute("analyze")
    assert tiny.execute("analyze") == paths
    assert len(json.loads((tiny.output / "history.json").read_text())) == 1
    inference = saved_references(paths[0])["inference"]
    assert tiny.execute("train") == paths  # models reused; the runs' analysis starts over
    assert not (paths[0] / "metrics.json").exists() and runs(tiny.output) == []
    assert inference.exists()  # cached analysis data is kept for the next analyze
    stamp = (inference / "artifact.json").stat().st_mtime_ns
    tiny.execute("analyze")
    assert saved_references(paths[0])["inference"] == inference
    assert (inference / "artifact.json").stat().st_mtime_ns == stamp


def failing(context, dependencies):
    raise RuntimeError("deliberate metric failure")


class FailingMetric(GaussianAnalysis):
    def metrics(self):
        return {**super().metrics(), "failing": Metric(failing, ("model",))}


def test_failed_launch_deletes_nothing(tiny):
    paths = tiny.execute()
    model = saved_references(paths[0])["model"]
    tiny.training["training"]["learning_rate"] = 0.01
    tiny.analysis["analysis"] = "test_pipeline:FailingMetric"
    tiny.analysis["metrics"] += ["failing"]
    with pytest.raises(RuntimeError, match="deliberate"):
        tiny.execute()
    assert paths[0].exists() and model.exists()
    assert len(json.loads((tiny.output / "history.json").read_text())) == 1
    failed = [record for record in runs(tiny.output, complete_only=False) if record["analysis"].get("state") == "failed"]
    assert len(failed) == 1 and failed[0]["status"]["state"] == "trained"
    assert "deliberate" in failed[0]["analysis"]["traceback"]


def test_saved_root_is_relocatable_and_restore_ignores_launcher_environment(tiny, tmp_path, monkeypatch):
    original = tiny.execute()[0]
    destination = tmp_path / "moved"
    shutil.copytree(tiny.output, destination)
    moved = destination / original.relative_to(tiny.output)
    shutil.rmtree(tiny.output)
    monkeypatch.setenv("RANK", "99")
    monkeypatch.setenv("WORLD_SIZE", "100")
    monkeypatch.setenv("LOCAL_RANK", "99")
    context = restore(moved)
    assert context.runtime.world_size == 1
    assert context.require("inference").path.is_relative_to(destination)
    assert analyze_run(context)["classification"]["bce"] > 0


def test_train_then_analyze_and_no_implicit_training(tiny):
    with pytest.raises(FileNotFoundError, match="No trained models"):
        tiny.execute("analyze")
    path = tiny.execute("train")[0]
    assert not (path / "metrics.json").exists() and runs(tiny.output) == []
    assert json.loads((path / "status.json").read_text())["state"] == "trained"
    assert (tiny.output / "settings_training.py").exists() and not (tiny.output / "settings_analysis.py").exists()
    weights = saved_references(path)["model"] / "weights.pt"
    stamp = weights.stat().st_mtime_ns
    tiny.execute("analyze")
    assert weights.stat().st_mtime_ns == stamp
    assert (path / "metrics.json").exists()


class FailsOnce(GaussianExperiment):
    failures = 1  # class state: each launch creates a new instance

    def train(self, context, model):
        if FailsOnce.failures:
            FailsOnce.failures -= 1
            raise RuntimeError("injected training interruption")
        return super().train(context, model)


def test_failed_training_recovers_without_rebuilding_data(tiny):
    FailsOnce.failures = 1
    tiny.training["experiment"] = "test_pipeline:FailsOnce"
    with pytest.raises(RuntimeError, match="interruption"):
        tiny.execute()
    training = list((tiny.output / "cache" / "training").glob("*/artifact.json"))
    assert len(training) == 1
    path = tiny.execute()[0]
    assert json.loads((path / "analysis.json").read_text())["state"] == "complete"
    assert len(list((tiny.output / "cache" / "training").glob("*/artifact.json"))) == 1


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
    tiny.training["experiment"] = "test_pipeline:OracleExperiment"
    tiny.training["cohort"]["priors"] = ["normal"]
    tiny.analysis["inference"].update(candidate_design="grid", candidate_points_per_axis=2001,
                                      native_grid_prior=False, retain="full", truth_points_per_axis=3)
    tiny.analysis["metrics"] = ["bias", "coverage", "width", "posterior", "exact_reference"]
    context = restore(tiny.execute()[0])
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
    tiny.training["problem"].update(dimension=2, infer=["mean:*", "std:1"])
    tiny.analysis["inference"].update(truth_points_per_axis=2, candidate_count=128)
    context = restore(tiny.execute()[0])
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


class ExtraPlot(GaussianAnalysis):
    def plots(self):
        return {**super().plots(), "custom": Plot(arbitrary_plot, ("observations", "model"))}


def test_plots_and_result_readers(tiny, tmp_path):
    tiny.analysis["analysis"] = "test_pipeline:ExtraPlot"
    tiny.analysis["plots"] = ["training", "prior", "inference", "pairwise", "reweighting", "showcase", "custom"]
    tiny.analysis["verification"]["showcase"]["samples"] = 64
    path = tiny.execute()[0]
    assert len(list((path / "plots").rglob("*.png"))) >= 8
    assert (path / "plots" / "custom" / "anything.png").stat().st_size > 1000
    records = runs(tiny.output)
    assert len(records) == 1
    table = export_csv(tiny.output, tmp_path / "summary.csv", {"bce": "metrics.classification.bce"})
    assert "bce" in table.read_text()
    fig = comparison_plot(records, "settings.problem.dimension", "metrics.classification.auc")
    assert len(fig.axes) == 1
    from matplotlib import pyplot as plt
    plt.close(fig)


INFEASIBLE = r"not feasible[\s\S]*problem\.dimension=2: .*max_model_evaluations"


def make_infeasible(tiny):
    tiny.analysis["inference"].update(truth_design="grid", truth_points_per_axis=40, max_model_evaluations=1_000_000)


def test_infeasible_analysis_refuses_run_before_computing(tiny):
    tiny.training["problem"]["dimension"] = Choice([1, 2])
    make_infeasible(tiny)
    with pytest.raises(ValueError, match=INFEASIBLE):
        tiny.execute("run")
    assert not tiny.output.exists()  # nothing trained, including dimension 1


def test_infeasible_analysis_refuses_analyze_but_not_train(tiny):
    tiny.training["problem"]["dimension"] = Choice([1, 2])
    make_infeasible(tiny)
    paths = tiny.execute("train")  # training does not depend on the analysis settings
    with pytest.raises(ValueError, match=INFEASIBLE):
        tiny.execute("analyze")
    assert not (tiny.output / "cache" / "observations").exists()
    assert not any((path / "analysis.json").exists() for path in paths)


class IndependentExample(Experiment):
    """A second, non-Gaussian application proves that core has no Gaussian schema."""
    name = "independent-example"
    def make_problem(self, settings):
        return {"n": settings["sample_size"]}
    def make_prior(self, settings, member, problem):
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


class IndependentAnalysis(Analysis):
    def metrics(self):
        return {"parameter_count": Metric(model_size, ("model",))}


def test_framework_has_no_gaussian_settings_dependency(tiny):
    tiny.training = {"experiment": "test_pipeline:IndependentExample", "seed": 4, "sample_size": 10,
                     "output": str(tiny.output), "runtime": tiny.training["runtime"]}
    tiny.analysis = {"analysis": "test_pipeline:IndependentAnalysis", "metrics": ["parameter_count"],
                     "plots": [], "plotting": {"dpi": 100}}
    path = tiny.execute()[0]
    assert json.loads((path / "metrics.json").read_text()) == {"parameter_count": 2}
