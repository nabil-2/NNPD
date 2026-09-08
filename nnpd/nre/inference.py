"""Inference algebra independent of the simulator and of the experiment runner."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from .training import predict_logits


def ensemble_log_ratio(model, observations, theta, batch_size, device):
    """Sum per-observation logits, without sigmoid clipping or ratio products.

    Temporary model inputs are bounded by batch_size (or one observation ensemble).
    Returns one score per candidate. All observations share the same candidate theta.
    """
    observations, theta = np.asarray(observations), np.asarray(theta)
    if observations.ndim != 2 or theta.ndim != 2 or not len(observations):
        raise ValueError("Expected nonempty observations[m,d] and candidates[k,p].")
    scores = np.empty(len(theta), dtype=np.float64)
    m = len(observations)
    chunk = max(1, batch_size // m)
    for start in range(0, len(theta), chunk):
        block = theta[start:start + chunk]
        x = np.tile(observations, (len(block), 1))
        parameters = np.repeat(block, m, axis=0)
        logits = predict_logits(model, x, parameters, batch_size, device)
        scores[start:start + len(block)] = logits.reshape(len(block), m).sum(axis=1)
    return scores


def summarize_density(theta, log_density, log_measure, levels):
    """Joint HLD sets, projected axis bounds, and a density-mode estimate.

    log_measure is quadrature volume (continuous) or counting weight (discrete).
    HLD ranking uses DENSITY, not quadrature mass. Boundary ties are included as a
    whole, so enclosed mass may exceed the request. This also handles flat densities.
    These are candidate approximations, not guarantees about the continuous region.
    """
    theta = np.asarray(theta, dtype=np.float64)
    score, measure = np.asarray(log_density, dtype=np.float64), np.asarray(log_measure, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    if theta.ndim != 2 or score.shape != (len(theta),) or measure.shape != score.shape:
        raise ValueError("Incompatible candidate, density and measure shapes.")
    if not len(theta) or not np.isfinite(theta).all() or not np.isfinite(measure).all():
        raise ValueError("Candidate locations and log measures must be finite and nonempty.")
    if np.isnan(score).any() or np.isposinf(score).any() or not np.isfinite(score).any():
        raise ValueError("Density must have finite positive mass and no NaN/+inf scores.")
    if levels.ndim != 1 or not len(levels) or not ((levels > 0) & (levels <= 1)).all():
        raise ValueError("HLD levels are enclosed masses in (0, 1].")
    norm = logsumexp(score + measure)
    mass = np.exp(score + measure - norm)
    order = np.argsort(-score, kind="stable")
    cumulative = np.cumsum(mass[order])
    lower, upper, thresholds, actual = [], [], [], []
    for level in levels:
        index = min(int(np.searchsorted(cumulative, level, side="left")), len(theta) - 1)
        threshold = score[order[index]]
        chosen = (score >= threshold) & np.isfinite(score)
        lower.append(theta[chosen].min(axis=0))
        upper.append(theta[chosen].max(axis=0))
        thresholds.append(threshold)
        actual.append(mass[chosen].sum())
    return {"mode": theta[np.argmax(score)], "lower": np.array(lower), "upper": np.array(upper),
            "threshold": np.array(thresholds), "enclosed_mass": np.array(actual),
            "ess": float(1 / np.sum(mass ** 2)), "log_normalizer": float(norm)}


def normalized_mass(log_density, log_measure):
    values = np.asarray(log_density) + np.asarray(log_measure)
    return np.exp(values - logsumexp(values))
