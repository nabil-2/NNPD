from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from nnpd import Choice, restore
from nnpd.core.runtime import Runtime


@pytest.mark.parametrize("precision", ["off", "bfloat16"])
def test_adamw_residual_gradient_clipping_and_best_checkpoint(tiny, precision):
    tiny.training["training"].update(optimizer="adamw", checkpoint="best", precision=precision, gradient_clip=1.0)
    tiny.training["model"].update(kind="residual", width=8, blocks=1, dropout=0.1, activation="gelu")
    context = restore(tiny.execute()[0])
    history = context.artifacts["model"].json("history")
    assert history["selected_epoch"] == 1 + int(np.argmin([row["bce"] for row in history["validation"]]))
    assert np.isfinite(context.require("inference").array("ratio_mode")).all()


def test_sobol_truths_mixed_prior_support_and_zero_diagnostic_retention(tiny):
    tiny.training["problem"].update(dimension=2, infer=["mean:*", "std:1"])
    tiny.training["priors"]["mixed"] = {"family": "uniform", "by_parameter": {"mean:0": "grid"}}
    tiny.training["cohort"]["priors"] = ["mixed"]
    tiny.analysis["inference"].update(truth_design="sobol", sobol_truths=5, repeats=2, diagnostic_cases=0)
    tiny.analysis["verification"].update(pair_design="sobol", normalization_design="prior")
    context = restore(tiny.execute()[0])
    observations = context.require("observations")
    assert observations.array("truth").shape == (5 * 2 ** 2 * 2, 3)  # 5 truths at p=1, doubled twice for p=3; 2 repeats
    np.testing.assert_allclose(observations.array("truth")[::2], observations.array("truth")[1::2])
    assert not np.array_equal(observations.array("observations")[0], observations.array("observations")[1])
    candidates = context.require("candidates")
    assert candidates.metadata["base_measure"] == "mixed"
    assert np.isfinite(context.prior.log_prob(candidates.array("theta"))).all()
    assert context.require("inference").array("ratio_log_score").shape[0] == 0


def test_no_posterior_or_reference_targets_is_explicit(tiny):
    tiny.analysis["inference"]["targets"] = ["ratio"]
    tiny.analysis["verification"].update(pair_design="uniform", normalization_on_grid=False)
    path = tiny.execute()[0]
    metrics = json.loads((path / "metrics.json").read_text())
    assert metrics["posterior"] == {"enabled": False}
    assert metrics["exact_reference"] == {"enabled": False}


def test_runtime_validations(tiny):
    settings = deepcopy(tiny.training["runtime"])
    with pytest.raises(ValueError, match="parallel"):
        Runtime({**settings, "parallel": "unknown"})
    with pytest.raises(ValueError, match="Supported devices"):
        Runtime({**settings, "device": "meta"})
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA requested"):
            Runtime({**settings, "device": "cuda"})
    tiny.training["runtime"]["cpu_threads"] = Choice([1, 2])
    with pytest.raises(ValueError, match="separate launches"):
        tiny.execute()
