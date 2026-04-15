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


def _box_support_mask(x, parameter_range):
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1)
    return np.all((x >= parameter_range[0]) & (x <= parameter_range[1]), axis=-1)


def _sample_truncated_normal(generator, mean, std, low, high, shape):
    shape = tuple(int(v) for v in shape)
    if high < low:
        raise ValueError(f"Expected low <= high, got {low} > {high}.")
    total = int(np.prod(shape, dtype=np.int64))
    if total == 0:
        return np.empty(shape, dtype=np.float32)

    accepted = np.empty(total, dtype=np.float32)
    filled = 0
    while filled < total:
        remaining = total - filled
        batch_size = max(remaining, int(math.ceil(remaining * 1.5)))
        draws = generator.normal(mean, std, size=batch_size).astype(np.float32)
        keep = draws[(draws >= low) & (draws <= high)]
        if keep.size == 0:
            continue
        take = min(remaining, keep.size)
        accepted[filled : filled + take] = keep[:take]
        filled += take
    return accepted.reshape(shape)


def _sample_truncated_exponential(generator, lam, low, high, shape):
    shape = tuple(int(v) for v in shape)
    if high < low:
        raise ValueError(f"Expected low <= high, got {low} > {high}.")
    if lam <= 0:
        raise ValueError("lam must be strictly positive.")
    if high == low:
        return np.full(shape, low, dtype=np.float32)

    total = int(np.prod(shape, dtype=np.int64))
    u = generator.uniform(0.0, 1.0, size=total).astype(np.float64)
    exp_low = math.exp(-lam * low)
    exp_high = math.exp(-lam * high)
    samples = -np.log(exp_low - u * (exp_low - exp_high)) / lam
    return samples.reshape(shape).astype(np.float32)


def build_grid_axis(parameter_range, step):
    low, high = (float(v) for v in parameter_range)
    step = float(step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"grid step must be positive, got {step!r}.")
    if high < low:
        raise ValueError(f"parameter_range must be increasing, got {parameter_range!r}.")

    n_steps = int(np.floor((high - low) / step + 1e-12))
    axis = low + step * np.arange(n_steps + 1, dtype=np.float64)
    axis = axis[axis <= high + max(1e-12, abs(step) * 1e-9)]
    if axis.size == 0:
        axis = np.asarray([low], dtype=np.float64)
    axis[0] = low
    return axis


def build_grid_support_axes(parameter_range, n_dimensions, step):
    axis = build_grid_axis(parameter_range, step)
    return tuple(axis.copy() for _ in range(int(n_dimensions)))


def _grid_match_indices(x, support_axes, step):
    points = np.asarray(x, dtype=float)
    if points.ndim == 0:
        points = points.reshape(1, 1)
    elif points.ndim == 1:
        points = points.reshape(1, -1)

    n_dims = len(support_axes)
    if points.shape[-1] != n_dims:
        raise ValueError(
            f"Expected points with final dimension {n_dims}, got shape {points.shape}."
        )

    flat = points.reshape(-1, n_dims)
    indices = np.empty_like(flat, dtype=np.int64)
    valid = np.ones(flat.shape[0], dtype=bool)
    atol = max(1e-6, abs(float(step)) * 1e-4)

    for dim, axis in enumerate(support_axes):
        axis = np.asarray(axis, dtype=float)
        scaled = (flat[:, dim] - float(axis[0])) / float(step)
        idx = np.rint(scaled).astype(np.int64)
        idx_clipped = np.clip(idx, 0, axis.size - 1)
        matched = np.isclose(flat[:, dim], axis[idx_clipped], atol=atol, rtol=0.0)
        matched &= idx >= 0
        matched &= idx < axis.size
        valid &= matched
        indices[:, dim] = idx_clipped

    return valid.reshape(points.shape[:-1]), indices.reshape(points.shape)


def grid_fct(x, step=0.2, parameter_range=(0.0, 10.0), support_axes=None):
    x = np.asarray(x, dtype=float)
    point_mode = x.ndim <= 1
    if support_axes is None:
        support_axes = build_grid_support_axes(
            parameter_range,
            n_dimensions=x.shape[-1] if x.ndim > 1 else 1,
            step=step,
        )
    support_axes = tuple(np.asarray(axis, dtype=float) for axis in support_axes)
    support_size = int(np.prod([len(axis) for axis in support_axes], dtype=np.int64))
    mask, _ = _grid_match_indices(x, support_axes, step)
    result = np.where(mask, 1.0 / max(support_size, 1), 0.0).astype(float)
    if point_mode:
        return float(np.reshape(result, (-1,))[0])
    return result


def draw_grid_samples(generator, support_axes, shape):
    shape = _coerce_shape(shape)
    support_axes = tuple(np.asarray(axis, dtype=np.float32) for axis in support_axes)
    draws = []
    for axis in support_axes:
        idx = generator.integers(0, len(axis), size=shape, endpoint=False)
        draws.append(axis[idx])
    return np.stack(draws, axis=-1).astype(np.float32)


def _attach_prior_metadata(
    prior_fn,
    *,
    integral_over_box=None,
    support_bounds,
    support_kind="continuous_box",
    normalization_mode="continuous_integral",
    support_axes=None,
    grid_step=None,
    discrete_mass=None,
    support_size=None,
):
    prior_fn.integral_over_box = None if integral_over_box is None else float(integral_over_box)
    prior_fn.support_bounds = tuple(float(v) for v in support_bounds)
    prior_fn.support_kind = str(support_kind)
    prior_fn.normalization_mode = str(normalization_mode)
    prior_fn.support_axes = (
        None
        if support_axes is None
        else tuple(np.asarray(axis, dtype=np.float32).copy() for axis in support_axes)
    )
    prior_fn.grid_step = None if grid_step is None else float(grid_step)
    prior_fn.discrete_mass = None if discrete_mass is None else float(discrete_mass)
    prior_fn.support_size = None if support_size is None else int(support_size)
    return prior_fn


def snap_to_support_axis(values, support_axis):
    support_axis = np.asarray(support_axis, dtype=np.float32).ravel()
    if support_axis.size == 0:
        raise ValueError("support_axis must contain at least one point.")
    values_arr = np.asarray(values, dtype=np.float32)
    if values_arr.ndim == 0:
        idx = int(np.abs(values_arr - support_axis).argmin())
        return float(support_axis[idx])
    idx = np.abs(values_arr[..., None] - support_axis[None, ...]).argmin(axis=-1)
    snapped = support_axis[idx]
    return np.asarray(snapped, dtype=np.float32)


def describe_prior_normalization(prior):
    mode = getattr(prior, "normalization_mode", "continuous_integral")
    if mode == "discrete_mass":
        return {
            "label": "Discrete mass",
            "value": float(getattr(prior, "discrete_mass", 1.0)),
        }
    integral = getattr(prior, "integral_over_box", None)
    return {
        "label": "Integral over box",
        "value": None if integral is None else float(integral),
    }


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


def _coerce_shape(shape):
    if isinstance(shape, int):
        return (shape,)
    shape = tuple(shape)
    return shape or (1,)


def get_alignment_support_axes(priors):
    for prior in priors:
        support_axes = getattr(prior, "support_axes", None)
        if getattr(prior, "support_kind", None) == "discrete_grid" and support_axes is not None:
            return tuple(np.asarray(axis, dtype=np.float32).copy() for axis in support_axes)
    return None


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
        support_kind="continuous_box",
        normalization_mode="continuous_integral",
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
        x = np.asarray(x, dtype=float)
        vals = normal(x, normal_mean, normal_std) / normal_integral
        return np.where(_box_support_mask(x, parameter_range), vals, 0.0)

    normal_pdf.__name__ = "normal"
    _attach_prior_metadata(
        normal_pdf,
        integral_over_box=1.0,
        support_bounds=parameter_range,
        support_kind="continuous_box",
        normalization_mode="continuous_integral",
    )

    def draw_normal(shape):
        shape = _coerce_shape(shape)
        return _sample_truncated_normal(
            generator,
            normal_mean,
            normal_std,
            param_min,
            param_max,
            (*shape, n_dimensions),
        )

    draw_normal.__name__ = "normal"

    exponential_lam = prior_args["exponential"][0]
    exponential_integral = _truncated_exponential_integral(
        parameter_range,
        n_dimensions,
        lam=exponential_lam,
    )
    exponential_support_low = max(param_min, 0.0)

    def exponential_pdf(x):
        x = np.asarray(x, dtype=float)
        vals = exponential(x, exponential_lam) / exponential_integral
        return np.where(_box_support_mask(x, parameter_range), vals, 0.0)

    exponential_pdf.__name__ = "exponential"
    _attach_prior_metadata(
        exponential_pdf,
        integral_over_box=1.0,
        support_bounds=parameter_range,
        support_kind="continuous_box",
        normalization_mode="continuous_integral",
    )

    def draw_exponential(shape):
        shape = _coerce_shape(shape)
        return _sample_truncated_exponential(
            generator,
            exponential_lam,
            exponential_support_low,
            param_max,
            (*shape, n_dimensions),
        )

    draw_exponential.__name__ = "exponential"

    grid_step = prior_args["grid"][0]
    grid_support_axes = build_grid_support_axes(parameter_range, n_dimensions, grid_step)
    grid_support_size = int(np.prod([len(axis) for axis in grid_support_axes], dtype=np.int64))

    def grid_pdf(x):
        return grid_fct(
            x,
            step=grid_step,
            parameter_range=parameter_range,
            support_axes=grid_support_axes,
        )

    grid_pdf.__name__ = "grid"
    _attach_prior_metadata(
        grid_pdf,
        integral_over_box=None,
        support_bounds=parameter_range,
        support_kind="discrete_grid",
        normalization_mode="discrete_mass",
        support_axes=grid_support_axes,
        grid_step=grid_step,
        discrete_mass=1.0,
        support_size=grid_support_size,
    )

    def draw_grid(shape):
        return draw_grid_samples(generator, grid_support_axes, shape)

    draw_grid.__name__ = "grid"

    priors = [uniform_pdf, normal_pdf, exponential_pdf, grid_pdf]
    prior_samplers = [draw_uniform, draw_normal, draw_exponential, draw_grid]
    return priors, prior_samplers, param_min, param_max
