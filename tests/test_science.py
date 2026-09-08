from copy import deepcopy
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm, truncnorm, wasserstein_distance

from gaussian.problem import GaussianProblem
from gaussian.diagnostics import sliced_wasserstein
from nnpd.nre.distributions import IndependentPrior, grid_axis
from nnpd.nre.data import balanced_pairs
from nnpd.nre.inference import ensemble_log_ratio, normalized_mass, summarize_density
from nnpd.nre.models import build_network
from nnpd.nre.training import classification, predict_logits


def make_prior(tiny, kind, rate=0.1):
    cfg = deepcopy(tiny["problem"])
    cfg["parameters"]["mean"]["exponential_rate"] = rate
    return IndependentPrior(GaussianProblem(cfg).parameters, {"family": kind})


@pytest.mark.parametrize("kind,rate", [("uniform", 0), ("normal", 0), ("exponential", 0.1),
                                       ("exponential", -0.1), ("exponential", 0),
                                       ("exponential", 10), ("exponential", -10)])
def test_continuous_priors_normalized_and_sampler_matches_density(tiny, kind, rate):
    prior = make_prior(tiny, kind, rate)
    density = lambda x: float(np.exp(prior.log_prob(np.array([x]))))
    assert quad(density, 0, 10, epsabs=1e-10)[0] == pytest.approx(1, abs=1e-8)
    expected = quad(lambda x: x * density(x), 0, 10)[0]
    sample = prior.sample(40_000, np.random.default_rng(413))
    assert sample.shape == (40_000, 1)
    assert np.isfinite(sample).all() and sample.min() >= 0 and sample.max() <= 10
    assert sample.mean() == pytest.approx(expected, abs=0.055)
    assert np.isneginf(prior.log_prob(np.array([[-1], [11]]))).all()


def test_normal_uses_standard_deviation_four_not_variance_four(tiny):
    prior = make_prior(tiny, "normal")
    theta = np.array([[1.0], [5.0], [9.0]])
    np.testing.assert_allclose(prior.log_prob(theta), truncnorm.logpdf(theta[:, 0], -1.25, 1.25, loc=5, scale=4))


def test_grid_probability_mass_and_native_support(tiny):
    prior = make_prior(tiny, "grid")
    axis = grid_axis(0, 10, 0.2)
    assert len(axis) == 51
    assert np.exp(prior.log_prob(axis[:, None])).sum() == pytest.approx(1)
    assert np.isneginf(prior.log_prob(np.array([[0.1], [10.2]]))).all()
    values = prior.sample(2000, np.random.default_rng(3))
    assert np.isfinite(prior.log_prob(values)).all()
    assert prior.measure == "discrete"


def test_mixed_support_and_selected_parameter_order(tiny):
    cfg = deepcopy(tiny["problem"])
    cfg.update(dimension=2, infer=["std:1", "mean:0"], fixed={"mean:1": 8})
    problem = GaussianProblem(cfg)
    mean, std = problem.unpack(np.array([[3, 7], [1, 4]]))
    np.testing.assert_allclose(mean, [[7, 8], [4, 8]])
    np.testing.assert_allclose(std, [[2, 3], [2, 1]])
    prior = IndependentPrior(problem.parameters, {"family": "uniform", "by_parameter": {"mean:0": "grid"}})
    assert prior.measure == "mixed"
    assert prior.support_axes[0] is None
    assert prior.sample(100, np.random.default_rng(0)).shape == (100, 2)


@pytest.mark.parametrize("infer", [[], ["mean:0", "mean:0"], ["foo:0"], ["std:1"], ["mean:*", "mean:0"]])
def test_invalid_dependencies(tiny, infer):
    cfg = deepcopy(tiny["problem"])
    cfg["infer"] = infer
    with pytest.raises(ValueError):
        GaussianProblem(cfg)


def test_gaussian_likelihood_sampling_and_positive_std(tiny):
    cfg = deepcopy(tiny["problem"])
    cfg.update(dimension=2, infer=["mean:*", "std:1"])
    problem = GaussianProblem(cfg)
    theta = np.repeat([[3, 6, 1.5]], 50_000, axis=0)
    x = problem.sample(theta, np.random.default_rng(1))
    np.testing.assert_allclose(x.mean(axis=0), [3, 6], atol=0.04)
    np.testing.assert_allclose(x.std(axis=0), [2, 1.5], atol=0.04)
    expected = norm.logpdf(x[:5, 0], 3, 2) + norm.logpdf(x[:5, 1], 6, 1.5)
    np.testing.assert_allclose(problem.log_likelihood(x[:5], theta[:5]), expected)
    with pytest.raises(ValueError, match="positive"):
        problem.sample(np.array([[3, 6, 0]]), np.random.default_rng())


def test_balanced_joint_product_and_exact_sizes(tiny):
    problem = GaussianProblem(tiny["problem"])
    prior = make_prior(tiny, "uniform")
    x, theta, y = balanced_pairs(problem, prior, 40_000, np.random.default_rng(17))
    assert x.shape == theta.shape == (40_000, 1)
    assert (y == 1).sum() == (y == 0).sum() == 20_000
    assert abs(np.corrcoef(x[y == 0, 0], theta[y == 0, 0])[0, 1]) < 0.04
    assert np.corrcoef(x[y == 1, 0], theta[y == 1, 0])[0, 1] > 0.8
    with pytest.raises(ValueError):
        balanced_pairs(problem, prior, 5, np.random.default_rng())


def test_hld_matches_analytic_gaussian_interval():
    theta = np.linspace(0, 10, 20001)[:, None]
    sigma = 2 / np.sqrt(15)
    score = norm.logpdf(theta[:, 0], 5, sigma)
    result = summarize_density(theta, score, np.zeros(len(theta)), [0.68, 0.95])
    np.testing.assert_allclose(result["mode"], [5])
    widths = result["upper"][:, 0] - result["lower"][:, 0]
    np.testing.assert_allclose(widths, 2 * norm.ppf([0.84, 0.975]) * sigma, atol=0.002)
    assert (result["enclosed_mass"] >= [0.68, 0.95]).all()


def test_joint_hld_projection_is_not_marginal_hld():
    axis = np.linspace(-5, 5, 251)
    theta = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    result = summarize_density(theta, -0.5 * np.sum(theta ** 2, axis=1), np.zeros(len(theta)), [0.68])
    radius = np.sqrt(-2 * np.log(1 - 0.68))
    np.testing.assert_allclose(result["upper"][0], radius, atol=0.06)
    assert (result["upper"][0] > norm.ppf(0.84) + 0.3).all()


def test_hld_ranks_density_not_quadrature_mass_and_includes_ties():
    result = summarize_density(np.array([[0], [1]]), np.log([2, 1]), np.log([0.001, 100]), [0.5])
    assert result["mode"].tolist() == [0]
    assert result["lower"].tolist() == [[0]] and result["upper"].tolist() == [[1]]
    flat = summarize_density(np.arange(4)[:, None], np.zeros(4), np.zeros(4), [0.2, 1.0])
    np.testing.assert_allclose(flat["enclosed_mass"], [1, 1])
    np.testing.assert_allclose(flat["ess"], 4)


@pytest.mark.parametrize("scores", [[float("nan"), 0], [float("inf"), 0], [-np.inf, -np.inf]])
def test_invalid_density_rejected(scores):
    with pytest.raises(ValueError):
        summarize_density(np.array([[0], [1]]), scores, [0, 0], [0.68])


def test_extreme_scores_are_stable():
    weights = normalized_mass([10000, 10001, -10000], [0, 0, 0])
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1)
    assert classification(np.array([0, 1]), np.array([-10000, 10000]))["bce"] == 0
    tied = classification(np.array([0, 1, 0, 1]), np.zeros(4))
    assert tied["auc"] == pytest.approx(0.5)


class KnownLogits(torch.nn.Module):
    def forward(self, features):
        return features[:, 0] + 2 * features[:, 1]


def test_ensemble_log_ratio_and_chunk_equivalence():
    obs = np.array([[1], [2], [3]], dtype=np.float32)
    theta = np.arange(9, dtype=np.float32)[:, None]
    expected = 6 + 6 * theta[:, 0]
    for batch in (1, 4, 128):
        np.testing.assert_allclose(ensemble_log_ratio(KnownLogits(), obs, theta, batch, "cpu"), expected)
    np.testing.assert_allclose(predict_logits(KnownLogits(), obs, np.array([2]), 1, "cpu"), [5, 6, 7])


@pytest.mark.parametrize("kind", ["mlp", "residual"])
def test_models_return_scalar_logits_and_accept_mixed_inputs(tiny, kind):
    definition = {**tiny["model"], "kind": kind}
    model = build_network(5, definition)
    out = model(torch.randn(7, 5))
    assert out.shape == (7,)
    out.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert model(torch.randn(1, 5)).shape == (1,)


def test_weighted_wasserstein_matches_scipy():
    a, b = np.array([[0.], [1.], [3.]]), np.array([[2.], [5.]])
    w = np.array([0.1, 0.2, 0.7])
    expected = wasserstein_distance(a[:, 0], b[:, 0], u_weights=w)
    assert sliced_wasserstein(a, b, w, np.array([[1.]])) == pytest.approx(expected)
    assert sliced_wasserstein(a, a, np.ones(3), np.array([[1.], [-1.]])) == 0
