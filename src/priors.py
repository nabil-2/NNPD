from __future__ import annotations

import itertools

import numpy as np

try:
    import vegas
except ImportError:  # pragma: no cover - optional dependency
    vegas = None


def integrate_nd_vegas(func, bounds, nitn=10, neval=10_000):
    """Compute an nD integral using VEGAS Monte Carlo integration."""

    def _wrapped(x):
        x = np.asarray(x)
        if x.ndim == 1:
            return func(x)
        return np.asarray([func(xi) for xi in x])

    if vegas is None:
        rng = np.random.default_rng(0)
        lows = np.asarray([low for low, _ in bounds], dtype=float)
        highs = np.asarray([high for _, high in bounds], dtype=float)
        volume = float(np.prod(highs - lows))
        estimates = []
        for _ in range(max(1, int(nitn))):
            samples = rng.uniform(lows, highs, size=(int(neval), len(bounds)))
            values = np.asarray(_wrapped(samples), dtype=float).reshape(-1)
            estimates.append(volume * float(np.mean(values)))
        estimates = np.asarray(estimates, dtype=float)
        return float(np.mean(estimates)), float(np.std(estimates, ddof=0))

    integrator = vegas.Integrator(bounds)
    result = integrator(_wrapped, nitn=nitn, neval=neval)
    return result.mean, result.sdev


def normalize(function, parameter_range, n_dimensions, fct_args):
    fct_args = tuple(() if fct_args is None else fct_args)
    integral, _ = integrate_nd_vegas(
        lambda x: function(x, *fct_args),
        [(parameter_range[0], parameter_range[1])] * n_dimensions,
        nitn=20,
        neval=100_000,
    )

    def result_fct(z):
        return function(z, *fct_args) / integral

    result_fct.__name__ = function.__name__
    return result_fct


def uniform(x, parameter_range, n_dimensions):
    x = np.asarray(x)
    result = np.full(
        x.shape[:-1],
        1.0 / (parameter_range[1] - parameter_range[0]) ** n_dimensions,
        dtype=float,
    )
    mask = np.any((x < parameter_range[0]) | (x > parameter_range[1]), axis=-1)
    result[mask] = 0.0
    if result.ndim == 0:
        return float(result)
    return result


def normal(x: np.ndarray, mean=5, std=3):
    x = np.asarray(x)
    return np.exp(-0.5 * np.sum(((x - mean) / std) ** 2, axis=-1))


def exponential(x, lam=0.3):
    x = np.asarray(x)
    mask = np.all(x >= 0, axis=-1)
    vals = np.exp(-lam * np.sum(x, axis=-1))
    return np.where(mask, vals, 0.0)


def grid_fct(x, step=0.2):
    x = np.asarray(x)
    mask = np.all(np.isclose(np.mod(x, step), 0, atol=1e-3), axis=-1)
    return mask.astype(float)


def _coerce_shape(shape):
    if isinstance(shape, int):
        return (shape,)
    shape = tuple(shape)
    return shape or (1,)


def build_priors(config, generator):
    data_config = config["data"]
    n_dimensions = data_config["n_parameters"]
    parameter_range = data_config["parameter_range"]
    prior_args = data_config["prior_args"]
    param_min, param_max = parameter_range

    def uniform_pdf(x):
        return uniform(x, parameter_range, n_dimensions)

    uniform_pdf.__name__ = "uniform"

    def draw_uniform(shape):
        shape = _coerce_shape(shape)
        return generator.uniform(param_min, param_max, size=(*shape, n_dimensions))

    draw_uniform.__name__ = "uniform"

    normal_pdf = normalize(normal, parameter_range, n_dimensions, prior_args["normal"])

    def draw_normal(shape):
        shape = _coerce_shape(shape)
        mean, std = prior_args["normal"]
        return generator.normal(mean, std, size=(*shape, n_dimensions))

    draw_normal.__name__ = "normal"

    exponential_pdf = normalize(
        exponential,
        parameter_range,
        n_dimensions,
        prior_args["exponential"],
    )

    def draw_exponential(shape):
        shape = _coerce_shape(shape)
        lam = prior_args["exponential"][0]
        return generator.exponential(1 / lam, size=(*shape, n_dimensions))

    draw_exponential.__name__ = "exponential"

    grid_pdf = normalize(grid_fct, parameter_range, n_dimensions, prior_args["grid"])

    def draw_grid(shape):
        shape = _coerce_shape(shape)
        step = prior_args["grid"][0]
        n_points_per_axis = int((param_max - param_min) / step) + 1
        grid_points = np.linspace(param_min, param_max, n_points_per_axis)
        all_combinations = np.array(
            list(itertools.product(grid_points, repeat=n_dimensions)),
            dtype=float,
        )
        n_total = int(np.prod(shape))
        indices = generator.choice(all_combinations.shape[0], size=n_total, replace=True)
        return all_combinations[indices].reshape(*shape, n_dimensions)

    draw_grid.__name__ = "grid"

    _ = (grid_pdf, draw_grid)
    priors = [uniform_pdf, normal_pdf, exponential_pdf]
    prior_samplers = [draw_uniform, draw_normal, draw_exponential]
    return priors, prior_samplers, param_min, param_max
