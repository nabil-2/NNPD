"""Reusable observation ensembles, candidate measures and model inference datasets."""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import qmc

from nnpd.core.config import select
from nnpd.nre.distributions import grid_axis
from nnpd.nre.inference import ensemble_log_ratio, summarize_density


def unit_sobol(count, dimension, seed):
    """Use a base-two draw then truncate; non-power-of-two counts are explicitly allowed."""
    if count < 1 or dimension < 1:
        raise ValueError("Sobol count and dimension must be positive.")
    return qmc.Sobol(dimension, scramble=True, seed=seed).random_base2((count - 1).bit_length())[:count]


def cartesian(axes):
    """Caller checks the product size before allocating."""
    shape = tuple(len(axis) for axis in axes)
    coordinates = np.indices(shape).reshape(len(axes), -1).T
    return np.column_stack([axis[coordinates[:, i]] for i, axis in enumerate(axes)])


def truth_settings(context):
    cfg = context.config
    names = ("seed", "observations", "repeats", "truth_design", "truth_points_per_axis",
             "truth_margin_fraction", "max_grid_truths", "sobol_truths", "align_truths_to_grid")
    return {"problem": cfg["problem"], "inference": select(cfg["inference"], *names)}


def candidate_settings(context):
    names = ("seed", "candidate_design", "candidate_count", "candidate_points_per_axis",
             "native_grid_prior", "max_candidate_points")
    return {"problem": context.config["problem"],
            "inference": select(context.config["inference"], *names),
            "support": [None if axis is None else axis.tolist() for axis in context.prior.support_axes]}


def inference_settings(context):
    return {"inference": select(context.config["inference"], "levels", "targets", "retain",
                                "diagnostic_cases", "max_saved_bytes"),
            "prior": context.config["priors"][context.member["prior"]],
            "evaluation_batch_size": context.config["training"]["evaluation_batch_size"],
            "device": context.runtime.device.type}


def truth_layout(problem, cfg):
    size = cfg["truth_points_per_axis"] ** len(problem.parameters)
    design = cfg["truth_design"]
    if design == "auto":
        design = "grid" if size <= cfg["max_grid_truths"] else "sobol"
    return design, size if design == "grid" else cfg["sobol_truths"]


def inference_observations(context, dependencies, writer):
    problem, cfg = context.problem, context.config["inference"]
    dimension = len(problem.parameters)
    design, count = truth_layout(problem, cfg)
    bounds_low = np.array([p.low for p in problem.parameters])
    bounds_high = np.array([p.high for p in problem.parameters])
    margin = cfg["truth_margin_fraction"] * (bounds_high - bounds_low)
    low, high = bounds_low + margin, bounds_high - margin
    if (low > high).any():
        raise ValueError("The truth margin removes the entire inference domain.")
    if design == "grid":
        truths = cartesian([np.linspace(a, b, cfg["truth_points_per_axis"]) for a, b in zip(low, high)])
    else:
        seed = context.seed("truth-design", base=cfg["seed"], member=False)
        truths = low + unit_sobol(count, dimension, seed) * (high - low)
    # Explicit legacy compatibility option, independent of the current training prior.
    # Do not silently change the truth set when changing the comparison cohort.
    if cfg["align_truths_to_grid"]:
        for j, parameter in enumerate(problem.parameters):
            axis = grid_axis(parameter.low, parameter.high, parameter.options["grid_step"])
            axis = axis[(axis >= low[j] - 1e-7) & (axis <= high[j] + 1e-7)]
            if not len(axis):
                raise ValueError("No grid support remains inside the requested truth margin.")
            right = np.searchsorted(axis, truths[:, j]).clip(0, len(axis) - 1)
            left = (right - 1).clip(0, len(axis) - 1)
            index = np.where(np.abs(truths[:, j] - axis[left]) <= np.abs(truths[:, j] - axis[right]), left, right)
            truths[:, j] = axis[index]
    unique_count = len(np.unique(truths, axis=0))
    truths = np.repeat(truths, cfg["repeats"], axis=0).astype(np.float32)
    writer.array("truth", truths)
    rng = context.rng("inference-observations", base=cfg["seed"], member=False)
    observations = writer.allocate("observations", (len(truths), cfg["observations"], problem.observation_dim),
                                   dtype="float32")
    # Chunk generation avoids a second full copy of high-dimensional ensembles.
    for start in range(0, len(truths), 1024):
        block = truths[start:start + 1024]
        observations[start:start + len(block)] = problem.sample(np.repeat(block[:, None, :],
                                                                        cfg["observations"], axis=1), rng)
    return {"parameters": problem.names, "truth_design_requested": cfg["truth_design"],
            "truth_design_actual": design, "truths_before_repeats": count, "unique_truths": unique_count,
            "cases": len(truths), "observations_per_case": cfg["observations"],
            "truths_aligned_to_grid": cfg["align_truths_to_grid"],
            "shared_across_priors_and_training_replicas": True}


def candidate_layout(problem, prior, cfg):
    native = cfg["native_grid_prior"]
    supports = prior.support_axes if native else (None,) * len(problem.parameters)
    if cfg["candidate_design"] == "grid":
        count = math.prod(len(axis) if axis is not None else cfg["candidate_points_per_axis"]
                          for axis in supports)
        return supports, count, "grid"
    if all(axis is not None for axis in supports):
        size = math.prod(len(axis) for axis in supports)
        if size <= cfg["candidate_count"]:
            return supports, size, "native-exact"
    return supports, cfg["candidate_count"], "sobol"


def candidates(context, dependencies, writer):
    problem, prior, cfg = context.problem, context.prior, context.config["inference"]
    supports, count, design = candidate_layout(problem, prior, cfg)
    if count > cfg["max_candidate_points"]:
        raise ValueError("Candidate allocation exceeds max_candidate_points; no automatic downsampling.")
    if any(axis is not None for axis in prior.support_axes) and not cfg["native_grid_prior"]:
        if "posterior" in cfg["targets"]:
            raise ValueError("A discrete prior posterior requires native_grid_prior=True.")
    axes = [axis if axis is not None else np.linspace(p.low, p.high, cfg["candidate_points_per_axis"])
            for p, axis in zip(problem.parameters, supports)] if design != "sobol" else None
    measure_name = "counting" if all(axis is not None for axis in supports) else (
        "mixed" if any(axis is not None for axis in supports) else "Lebesgue")
    if axes is not None:
        theta = cartesian(axes)
        axis_weights = []
        for axis, support in zip(axes, supports):
            weights = np.ones(len(axis))
            if support is None:
                weights *= (axis[-1] - axis[0]) / (len(axis) - 1)
                weights[[0, -1]] *= 0.5   # trapezoidal endpoint weights
            axis_weights.append(np.log(weights))
        log_measure = cartesian(axis_weights).sum(axis=1)
    else:
        seed = context.seed("candidate-design", base=cfg["seed"], member=False)
        unit = unit_sobol(count, len(problem.parameters), seed)
        theta = np.empty_like(unit)
        total_log_measure = 0.0
        for j, (parameter, support) in enumerate(zip(problem.parameters, supports)):
            if support is None:
                theta[:, j] = parameter.low + unit[:, j] * (parameter.high - parameter.low)
                total_log_measure += math.log(parameter.high - parameter.low)
            else:
                theta[:, j] = support[np.minimum((unit[:, j] * len(support)).astype(int), len(support) - 1)]
                total_log_measure += math.log(len(support))
        log_measure = np.full(count, total_log_measure - math.log(count))
    # Combine coincident lattice samples: preserve quadrature mass, do not treat duplicates as ESS.
    unique, inverse = np.unique(theta, axis=0, return_inverse=True)
    if len(unique) < len(theta):
        weights = np.bincount(inverse, weights=np.exp(log_measure))
        theta, log_measure = unique, np.log(weights)
    writer.array("theta", theta.astype(np.float32))
    writer.array("log_measure", log_measure)
    return {"design": design, "requested_count": count, "actual_count": len(theta),
            "base_measure": measure_name, "parameters": problem.names,
            "native_discrete_support": cfg["native_grid_prior"],
            "approximation": "exact support enumeration" if design == "native-exact" else "quadrature"}


def inference(context, dependencies, writer):
    cfg = context.config["inference"]
    observations_data, candidate_data = dependencies["observations"], dependencies["candidates"]
    truths, observations = observations_data.array("truth"), observations_data.array("observations")
    theta, measure = candidate_data.array("theta"), candidate_data.array("log_measure")
    n, p, levels = len(truths), theta.shape[1], cfg["levels"]
    targets = cfg["targets"]
    kept = np.arange(n if cfg["retain"] == "full" else min(cfg["diagnostic_cases"], n))
    writer.array("retained_cases", kept)
    saved = len(kept) * len(theta) * 8 * len(targets)
    summary_bytes = n * len(targets) * (p + len(levels) * (2 * p + 3) + 2) * 8
    if saved + summary_bytes > cfg["max_saved_bytes"]:
        raise ValueError("Inference dataset exceeds max_saved_bytes; change retention or design explicitly.")
    arrays = {}
    for target in targets:
        shapes = {"mode": (n, p), "lower": (n, len(levels), p), "upper": (n, len(levels), p),
                  "threshold": (n, len(levels)), "enclosed_mass": (n, len(levels)),
                  "ess": (n,), "log_normalizer": (n,), "joint_contains_truth": (n, len(levels)),
                  "log_score": (len(kept), len(theta))}
        arrays[target] = {name: writer.allocate(f"{target}_{name}", shape, dtype="float64")
                          for name, shape in shapes.items()}
    batch = context.config["training"]["evaluation_batch_size"]
    prior_log = context.prior.log_prob(theta) if "posterior" in targets else None
    for case in range(n):
        obs, truth = observations[case], truths[case]
        if any(target != "exact" for target in targets):
            ratio = ensemble_log_ratio(context.model, obs, theta, batch, context.runtime.device)
            true_ratio = ensemble_log_ratio(context.model, obs, truth[None, :], batch,
                                             context.runtime.device)[0]
        for target in targets:
            if target == "exact":
                score = np.zeros(len(theta))
                for x in obs:
                    score += context.problem.log_likelihood(x, theta)
                true_score = context.problem.log_likelihood(obs, truth).sum()
            elif target == "posterior":
                score = ratio + prior_log       # prior ONCE per observation ensemble
                true_score = true_ratio + context.prior.log_prob(truth)
            else:
                score, true_score = ratio, true_ratio
            summary = summarize_density(theta, score, measure, levels)
            for name, value in summary.items():
                arrays[target][name][case] = value
            arrays[target]["joint_contains_truth"][case] = true_score >= summary["threshold"]
            if case < len(kept):
                arrays[target]["log_score"][case] = score
    return {"levels": levels, "targets": targets, "cases": n, "parameters": context.problem.names,
            "candidate_count": len(theta), "base_measure": candidate_data.metadata["base_measure"],
            "hld": "joint density superlevel set, projected coordinate bounds; threshold ties included",
            "coverage": "true parameter membership, not mode membership",
            "posterior_prior_power": 1, "retention": cfg["retain"],
            "exact_target": "normalized Gaussian likelihood on the SAME candidate measure (not posterior)"}
