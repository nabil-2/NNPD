from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from nnpd import Choice, execute, restore
from nnpd.core.runtime import Runtime
from gaussian import GaussianExperiment


@pytest.mark.parametrize("precision", ["off", "bfloat16"])
def test_adamw_residual_gradient_clipping_and_best_checkpoint(tiny, precision):
    tiny["training"].update(optimizer="adamw", checkpoint="best", precision=precision, gradient_clip=1.0)
    tiny["model"].update(kind="residual", width=8, blocks=1, dropout=0.1, activation="gelu")
    context = restore(execute(tiny, GaussianExperiment())[0], GaussianExperiment())
    history = context.artifacts["model"].json("history")
    assert history["selected_epoch"] == 1 + int(np.argmin([row["bce"] for row in history["validation"]]))
    assert np.isfinite(context.require("inference").array("ratio_mode")).all()


def test_sobol_truths_mixed_prior_support_and_zero_diagnostic_retention(tiny):
    tiny["problem"].update(dimension=2, infer=["mean:*", "std:1"])
    tiny["priors"]["mixed"] = {"family": "uniform", "by_parameter": {"mean:0": "grid"}}
    tiny["cohort"]["priors"] = ["mixed"]
    tiny["inference"].update(truth_design="sobol", sobol_truths=5, repeats=2, diagnostic_cases=0)
    tiny["verification"].update(pair_design="sobol", normalization_design="prior")
    context = restore(execute(tiny, GaussianExperiment())[0], GaussianExperiment())
    observations = context.require("observations")
    assert observations.array("truth").shape == (10, 3)
    np.testing.assert_allclose(observations.array("truth")[::2], observations.array("truth")[1::2])
    assert not np.array_equal(observations.array("observations")[0], observations.array("observations")[1])
    candidates = context.require("candidates")
    assert candidates.metadata["base_measure"] == "mixed"
    assert np.isfinite(context.prior.log_prob(candidates.array("theta"))).all()
    assert context.require("inference").array("ratio_log_score").shape[0] == 0


def test_no_posterior_or_reference_targets_is_explicit(tiny):
    tiny["inference"]["targets"] = ["ratio"]
    tiny["verification"].update(pair_design="uniform", normalization_on_grid=False)
    path = execute(tiny, GaussianExperiment())[0]
    metrics = json.loads((path / "metrics.json").read_text())
    assert metrics["posterior"] == {"enabled": False}
    assert metrics["exact_reference"] == {"enabled": False}


def test_runtime_validations(tiny):
    config = deepcopy(tiny["runtime"])
    with pytest.raises(ValueError, match="parallel"):
        Runtime({**config, "parallel": "unknown"})
    with pytest.raises(ValueError, match="Supported devices"):
        Runtime({**config, "device": "meta"})
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA requested"):
            Runtime({**config, "device": "cuda"})
    config = deepcopy(tiny)
    config["runtime"]["cpu_threads"] = Choice([1, 2])
    with pytest.raises(ValueError, match="separate launches"):
        execute(config, GaussianExperiment())
