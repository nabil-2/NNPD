"""Verification datasets: sample once, reuse across metrics and plots."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

from nnpd.core.config import select
from nnpd.nre.training import predict_logits
from nnpd.nre.distributions import grid_axis
from .evaluation import unit_sobol


def verification_settings(context):
    return select(context.config, "problem", "verification")


def model_evaluation_settings(context):
    return {"evaluation_batch_size": context.config["training"]["evaluation_batch_size"],
            "device": context.runtime.device.type}


def test_predictions(context, dependencies, writer):
    data = dependencies["training"]
    logits = predict_logits(context.model, data.array("test_x"), data.array("test_theta"),
                            context.config["training"]["evaluation_batch_size"], context.runtime.device)
    writer.array("logits", logits)
    writer.array("labels", data.array("test_y"))
    return {"split": "held-out test", "count": len(logits)}


def verification_observations(context, dependencies, writer):
    cfg, problem = context.config["verification"], context.problem
    rng = context.rng("verification-observations", base=cfg["seed"], member=False)
    low, high = np.array([(p.low, p.high) for p in problem.parameters]).T
    seed = context.seed("verification-pairs", base=cfg["seed"], member=False)
    if cfg["pair_design"] == "grid":
        pairs = np.empty((cfg["pairs"], 2, len(low)))
        for j, parameter in enumerate(problem.parameters):
            axis = grid_axis(parameter.low, parameter.high, parameter.options["grid_step"])
            pairs[:, :, j] = axis[rng.integers(0, len(axis), size=(cfg["pairs"], 2))]
    else:
        unit = (unit_sobol(cfg["pairs"] * 2, len(low), seed).reshape(cfg["pairs"], 2, len(low))
                if cfg["pair_design"] == "sobol" else rng.random((cfg["pairs"], 2, len(low))))
        pairs = low + unit * (high - low)
    writer.array("pairs", pairs)
    source, target = [], []
    for pair in pairs:
        a = problem.sample(np.repeat(pair[0][None], cfg["samples_per_endpoint"], axis=0), rng)
        b = problem.sample(np.repeat(pair[1][None], cfg["samples_per_endpoint"], axis=0), rng)
        source.append(np.concatenate((a, b)))
        target.append(problem.log_likelihood(source[-1], pair[1]) -
                      problem.log_likelihood(source[-1], pair[0]))
    writer.array("probe_x", np.array(source))
    writer.array("exact_probe_log_ratio", np.array(target))
    for name, column in (("source", 0), ("target", 1)):
        theta = np.repeat(pairs[:, column, None, :], cfg[f"{name}_samples"], axis=1)
        writer.array(f"{name}_x", problem.sample(theta, rng))
    directions = rng.normal(size=(cfg["directions"], problem.observation_dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    writer.array("directions", directions)
    return {"ratio_orientation": "p(x|theta1) / p(x|theta0)", "pairs": cfg["pairs"],
            "shared_across_priors_and_training_replicas": True,
            "theta_design": cfg["pair_design"], "grid_step_source": "problem.parameters"}


def verification_predictions(context, dependencies, writer):
    data = dependencies["verification_observations"]
    pairs = data.array("pairs")
    batch, device = context.config["training"]["evaluation_batch_size"], context.runtime.device
    for label in ("probe", "source"):
        observations = data.array(f"{label}_x")
        predicted, exact = [], []
        for pair, x in zip(pairs, observations):
            predicted.append(predict_logits(context.model, x, pair[1], batch, device) -
                             predict_logits(context.model, x, pair[0], batch, device))
            exact.append(context.problem.log_likelihood(x, pair[1]) - context.problem.log_likelihood(x, pair[0]))
        writer.array(f"{label}_log_ratio", predicted)
        writer.array(f"{label}_exact_log_ratio", exact)
    return {"pairs": len(pairs), "ratio_orientation": data.metadata["ratio_orientation"]}


def sliced_wasserstein(source, target, weights, directions):
    """Mean exact weighted 1D W1 over supplied unit directions (not a histogram approximation)."""
    if len(source) != len(weights) or (np.asarray(weights) < 0).any() or not np.sum(weights) > 0:
        raise ValueError("Weights must be nonnegative, aligned with source, and have positive total.")
    return float(np.mean([wasserstein_distance(source @ direction, target @ direction,
                                               u_weights=weights) for direction in directions]))


def reweighting_distances(context, dependencies, writer):
    samples, predictions = dependencies["verification_observations"], dependencies["verification_predictions"]
    directions = samples.array("directions")
    source, target = samples.array("source_x"), samples.array("target_x")
    results = {name: [] for name in ("learned_w1", "exact_w1", "unweighted_w1", "ess", "exact_ess")}
    for i, (a, b) in enumerate(zip(source, target)):
        results["unweighted_w1"].append(sliced_wasserstein(a, b, np.ones(len(a)), directions))
        for kind, prefix in (("learned", "source_log_ratio"), ("exact", "source_exact_log_ratio")):
            score = predictions.array(prefix)[i]
            weights = np.exp(score - logsumexp(score))
            results[f"{kind}_w1"].append(sliced_wasserstein(a, b, weights, directions))
            results["ess" if kind == "learned" else "exact_ess"].append(1 / np.sum(weights ** 2))
    for name, values in results.items():
        writer.array(name, values)
    return {"distance": "mean weighted one-dimensional Wasserstein-1 over saved directions",
            "source_samples_per_pair": source.shape[1]}


def normalization_observations(context, dependencies, writer):
    cfg, problem = context.config["verification"], context.problem
    rng = context.rng("normalization", base=cfg["seed"])
    theta = context.prior.sample(cfg["marginal_samples"], rng)
    writer.array("x", problem.sample(theta, rng))
    if cfg["normalization_design"] == "prior":
        points = context.prior.sample(cfg["normalization_thetas"], rng)
    else:
        columns = []
        for parameter in problem.parameters:
            if cfg["normalization_on_grid"]:
                axis = grid_axis(parameter.low, parameter.high, parameter.options["grid_step"])
                indices = np.rint(np.linspace(0, len(axis) - 1, cfg["normalization_thetas"])).astype(int)
                columns.append(axis[indices])
            else:
                columns.append(np.linspace(parameter.low, parameter.high, cfg["normalization_thetas"]))
        points = np.column_stack(columns)
    writer.array("theta", points)
    return {"distribution": "prior-specific evidence p_i(x)", "expected_ratio_mean": 1.0}


def normalization_predictions(context, dependencies, writer):
    data = dependencies["normalization_observations"]
    x, theta = data.array("x"), data.array("theta")
    scores = np.array([predict_logits(context.model, x, point,
                                     context.config["training"]["evaluation_batch_size"], context.runtime.device)
                       for point in theta])
    writer.array("log_ratio", scores)
    return {"expected_ratio_mean": 1.0, "samples": len(x), "theta_count": len(theta)}


def showcase_observations(context, dependencies, writer):
    cfg = context.config["verification"]["showcase"]
    low, high = np.array([(p.low, p.high) for p in context.problem.parameters]).T
    points = low + np.array([cfg["source_fraction"], cfg["target_fraction"]])[:, None] * (high - low)
    writer.array("theta", points)
    rng = context.rng("showcase", base=context.config["verification"]["seed"], member=False)
    for name, point in zip(("source", "target"), points):
        writer.array(name, context.problem.sample(np.repeat(point[None], cfg["samples"], axis=0), rng))
    return {"orientation": "source -> target", "parameters": context.problem.names}


def showcase_predictions(context, dependencies, writer):
    data = dependencies["showcase_observations"]
    x, theta = data.array("source"), data.array("theta")
    batch, device = context.config["training"]["evaluation_batch_size"], context.runtime.device
    writer.array("log_ratio", predict_logits(context.model, x, theta[1], batch, device) -
                 predict_logits(context.model, x, theta[0], batch, device))
    return {"source_samples": len(x)}
