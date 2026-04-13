from __future__ import annotations

import math

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


def _standard_normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _truncated_normal_integral(parameter_range, n_dimensions, mean, std):
    low, high = parameter_range
    delta_cdf = _standard_normal_cdf((high - mean) / std) - _standard_normal_cdf((low - mean) / std)
    return (std * math.sqrt(2.0 * math.pi) * delta_cdf) ** n_dimensions


def _truncated_exponential_integral(parameter_range, n_dimensions, lam):
    low, high = parameter_range
    support_low = max(low, 0.0)
    if high <= support_low:
        return 0.0
    one_dim = (math.exp(-lam * support_low) - math.exp(-lam * high)) / lam
    return one_dim**n_dimensions


def _attach_prior_metadata(prior_fn, *, integral_over_box, support_bounds):
    prior_fn.integral_over_box = float(integral_over_box)
    prior_fn.support_bounds = tuple(float(v) for v in support_bounds)
    return prior_fn


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
    _attach_prior_metadata(
        uniform_pdf,
        integral_over_box=1.0,
        support_bounds=parameter_range,
    )

    def draw_uniform(shape):
        shape = _coerce_shape(shape)
        return generator.uniform(param_min, param_max, size=(*shape, n_dimensions))

    draw_uniform.__name__ = "uniform"

    normal_mean, normal_std = prior_args["normal"]
    normal_integral = _truncated_normal_integral(
        parameter_range,
        n_dimensions,
        mean=normal_mean,
        std=normal_std,
    )

    def normal_pdf(x):
        return normal(x, normal_mean, normal_std) / normal_integral

    normal_pdf.__name__ = "normal"
    _attach_prior_metadata(
        normal_pdf,
        integral_over_box=1.0,
        support_bounds=parameter_range,
    )

    def draw_normal(shape):
        shape = _coerce_shape(shape)
        return generator.normal(normal_mean, normal_std, size=(*shape, n_dimensions))

    draw_normal.__name__ = "normal"

    exponential_lam = prior_args["exponential"][0]
    exponential_integral = _truncated_exponential_integral(
        parameter_range,
        n_dimensions,
        lam=exponential_lam,
    )

    def exponential_pdf(x):
        return exponential(x, exponential_lam) / exponential_integral

    exponential_pdf.__name__ = "exponential"
    _attach_prior_metadata(
        exponential_pdf,
        integral_over_box=1.0,
        support_bounds=parameter_range,
    )

    def draw_exponential(shape):
        shape = _coerce_shape(shape)
        return generator.exponential(1.0 / exponential_lam, size=(*shape, n_dimensions))

    draw_exponential.__name__ = "exponential"

    priors = [uniform_pdf, normal_pdf, exponential_pdf]
    prior_samplers = [draw_uniform, draw_normal, draw_exponential]
    return priors, prior_samplers, param_min, param_max
