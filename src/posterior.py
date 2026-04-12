from __future__ import annotations

import itertools

import numpy as np
import torch
from tqdm import tqdm

from .data import draw_data

try:  # pragma: no cover - depends on the local scipy build
    from scipy.optimize import brentq as _scipy_brentq
except Exception:  # pragma: no cover
    _scipy_brentq = None


def _get_model_lookup(models):
    if isinstance(models, dict):
        return models
    if isinstance(models, (list, tuple)) and models:
        if isinstance(models[0], dict):
            return models[0]
    raise TypeError("models must be a dict or a non-empty list/tuple of dicts.")


def _brentq(func, a, b, xtol=1e-12, maxiter=200):
    if _scipy_brentq is not None:
        return _scipy_brentq(func, a, b, xtol=xtol, maxiter=maxiter)

    fa = func(a)
    fb = func(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("Root is not bracketed on the interval.")

    lo, hi = float(a), float(b)
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = func(mid)
        if abs(fm) < xtol or abs(hi - lo) < xtol:
            return mid
        if fa * fm <= 0:
            hi = mid
            fb = fm
        else:
            lo = mid
            fa = fm
    return 0.5 * (lo + hi)


def _to_1d_torch_grid(x, device, dtype):
    t = torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
    if t.ndim != 1:
        raise ValueError("Each entry in x_range must be a 1D array.")
    if t.numel() < 2:
        raise ValueError("Each axis in x_range must contain at least 2 points.")
    return t


def integrate_nD(fct, x_range):
    result = fct
    for i in range(len(x_range) - 1, -1, -1):
        result = np.trapezoid(result, x_range[i], axis=i)
    return result


def integrate_nD_qmc_torch(
    fct_or_values,
    x_range,
    n_samples=2**18,
    batch_size=2**16,
    device=None,
    sobol_scramble=True,
    seed=0,
    use_float64=False,
    return_stderr=True,
):
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.float64 if use_float64 else torch.float32

    d = len(x_range)
    if d == 0:
        raise ValueError("x_range must contain at least one dimension.")

    x_t = [_to_1d_torch_grid(x, device=device, dtype=dtype) for x in x_range]
    lows = torch.stack([x[0] for x in x_t])
    highs = torch.stack([x[-1] for x in x_t])
    widths = highs - lows
    if torch.any(widths <= 0):
        raise ValueError("Each axis in x_range must be strictly increasing.")

    volume = torch.prod(widths.to(torch.float64))
    if not torch.isfinite(volume):
        raise FloatingPointError("Integration volume overflowed; check bounds or use log-domain scaling.")

    if callable(fct_or_values):

        def eval_fn(points):
            vals = fct_or_values(points)
            if isinstance(vals, np.ndarray):
                vals = torch.from_numpy(vals)
            vals = torch.as_tensor(vals, device=device)
            return vals.reshape(-1)

    else:
        y = torch.as_tensor(np.asarray(fct_or_values), device=device, dtype=dtype).contiguous()
        expected_shape = tuple(len(x) for x in x_range)
        if tuple(y.shape) != expected_shape:
            raise ValueError(
                f"fct shape {tuple(y.shape)} does not match expected grid shape {expected_shape}."
            )

        y_flat = y.reshape(-1)
        strides = []
        running = 1
        for size in reversed(expected_shape[1:]):
            running *= size
            strides.append(running)
        strides = [1] + strides
        strides = torch.as_tensor(list(reversed(strides)), device=device, dtype=torch.long)

        def eval_fn(points):
            lin_idx = torch.zeros(points.shape[0], device=device, dtype=torch.long)
            for dim, axis in enumerate(x_t):
                points_dim = points[:, dim].contiguous()
                idx_right = torch.searchsorted(axis, points_dim, right=False)
                idx_right = torch.clamp(idx_right, 0, axis.numel() - 1)
                idx_left = torch.clamp(idx_right - 1, 0, axis.numel() - 1)

                dist_left = torch.abs(points_dim - axis[idx_left])
                dist_right = torch.abs(axis[idx_right] - points_dim)
                idx = torch.where(dist_left <= dist_right, idx_left, idx_right)
                lin_idx = lin_idx + idx * strides[dim]
            return y_flat[lin_idx]

    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=sobol_scramble, seed=seed)
    n_done = 0
    sum_vals = torch.tensor(0.0, device=device, dtype=torch.float64)
    sum_sq_vals = torch.tensor(0.0, device=device, dtype=torch.float64)

    while n_done < n_samples:
        n_batch = min(batch_size, n_samples - n_done)
        u = sobol.draw(n_batch).to(device=device, dtype=dtype)
        pts = lows + u * widths
        vals = eval_fn(pts).to(torch.float64)

        sum_vals += torch.sum(vals)
        sum_sq_vals += torch.sum(vals * vals)
        n_done += n_batch

    mean = sum_vals / n_samples
    integral = volume * mean

    if return_stderr:
        var = torch.clamp(sum_sq_vals / n_samples - mean * mean, min=0.0)
        stderr = volume * torch.sqrt(var / n_samples)
        return float(integral.item()), float(stderr.item())

    return float(integral.item())


def integrate_nD_auto(fct_or_values, x_range, method="auto", qmc_threshold_dim=8, **qmc_kwargs):
    if method == "trapz":
        if callable(fct_or_values):
            raise TypeError("integrate_nD (trapz) expects tabulated values, not callable.")
        return integrate_nD(fct_or_values, x_range)

    if method == "qmc":
        return integrate_nD_qmc_torch(fct_or_values, x_range, **qmc_kwargs)

    if method == "auto":
        if callable(fct_or_values) or len(x_range) >= qmc_threshold_dim:
            return integrate_nD_qmc_torch(fct_or_values, x_range, **qmc_kwargs)
        if callable(fct_or_values):
            raise TypeError("Trapz path expects tabulated values.")
        return integrate_nD(fct_or_values, x_range)

    raise ValueError("method must be one of {'auto', 'trapz', 'qmc'}")


def to_grid(arr, x_ranges):
    return np.asarray(arr).reshape([len(r) for r in x_ranges])


def create_inference_parameters(
    n_parameters_to_infer,
    parameters_min_max,
    generator,
    margin=1 / 4,
    data_is_random=False,
):
    parameter_min, parameter_max = parameters_min_max
    parameter_range = parameter_max - parameter_min
    if data_is_random:
        return generator.uniform(
            low=parameter_min + parameter_range * margin,
            high=parameter_max - parameter_range * margin,
            size=n_parameters_to_infer,
        )
    return np.linspace(
        parameter_min + parameter_range * margin,
        parameter_max - parameter_range * margin,
        n_parameters_to_infer,
    )


def create_inference_data(
    parameters_min_max,
    config,
    generator,
    n_parameters_to_infer_per_dim=5,
    margin=1 / 4,
    data_is_random=False,
    n_repititions_per_parameter=1,
    parameters_to_infer=None,
):
    n_dimensions = config["data"]["n_parameters"]

    if parameters_to_infer is None:
        parameters_to_infer = []
        for _ in range(n_dimensions):
            parameters_to_infer.append(
                create_inference_parameters(
                    n_parameters_to_infer_per_dim,
                    parameters_min_max,
                    generator,
                    margin,
                    data_is_random,
                )
            )

    sampled_data = []
    parameter_combinations = list(itertools.product(*parameters_to_infer))
    for parameter_combination in parameter_combinations:
        sampled_data.append(
            draw_data(
                np.asarray([parameter_combination for _ in range(n_repititions_per_parameter)]),
                config,
                generator,
            )
        )
    sampled_data = np.asarray(sampled_data)

    return parameters_to_infer, parameter_combinations, sampled_data


def area_above_k(k, x_range, posterior_normalized, integration_device=None):
    mask = posterior_normalized >= k
    if not np.any(mask):
        return 0.0
    masked_posterior = np.where(mask, posterior_normalized, 0.0)
    area = integrate_nD_auto(
        to_grid(masked_posterior, x_range),
        x_range,
        method="qmc",
        device=integration_device,
    )
    if isinstance(area, tuple):
        area = area[0]
    return area


def compute_hpd_interval(x_range, posterior_normalized, alpha, integration_device=None):
    posterior_normalized = np.asarray(posterior_normalized)
    if posterior_normalized.ndim == 1:
        posterior_normalized = to_grid(posterior_normalized, x_range)

    def root_func(k):
        return area_above_k(k, x_range, posterior_normalized, integration_device) - (1 - alpha)

    p_min = np.min(posterior_normalized)
    p_max = np.max(posterior_normalized)
    func_lower = root_func(p_min)
    func_upper = root_func(p_max)

    if func_lower < 0 or func_upper > 0:
        raise ValueError("Cannot find a valid k in the given range. Check the posterior and alpha.")

    k = _brentq(root_func, p_min, p_max, xtol=1e-12)
    mask = posterior_normalized >= k

    n_dims = len(x_range)
    intervals_list = []
    for dim in range(n_dims):
        axes_to_reduce = tuple(i for i in range(n_dims) if i != dim)
        marginalized_mask = np.any(mask, axis=axes_to_reduce)
        indices = np.where(marginalized_mask)[0]
        theta = x_range[dim]

        intervals = []
        if indices.size > 0:
            breaks = np.where(np.diff(indices) > 1)[0]
            start_idx = 0
            breaks = np.append(breaks, len(indices) - 1)
            for break_idx in breaks:
                idx_range = indices[start_idx : break_idx + 1]
                intervals.append([theta[idx_range[0]], theta[idx_range[-1]]])
                start_idx = break_idx + 1

        intervals_list.append(intervals[0] if intervals else [])

    intervals_per_dim = np.asarray(intervals_list, dtype=object)
    valid_intervals = [iv for iv in intervals_list if len(iv) == 2]
    if valid_intervals:
        lows = [iv[0] for iv in valid_intervals]
        highs = [iv[1] for iv in valid_intervals]
        interval_combined = np.asarray([np.mean(lows), np.mean(highs)])
    else:
        interval_combined = np.asarray([])

    return intervals_per_dim, interval_combined


def _draw_qmc_theta_samples(all_parameters_in_range, n_samples, seed=0, scramble=True, dtype=torch.float32):
    n_dims = len(all_parameters_in_range)
    lows = torch.tensor([float(r[0]) for r in all_parameters_in_range], dtype=dtype)
    highs = torch.tensor([float(r[-1]) for r in all_parameters_in_range], dtype=dtype)
    widths = highs - lows

    sobol = torch.quasirandom.SobolEngine(dimension=n_dims, scramble=scramble, seed=seed)
    u = sobol.draw(n_samples).to(dtype=dtype)
    theta = lows[None, :] + u * widths[None, :]
    return theta.cpu().numpy().astype(np.float32)


def _normalize_log_weights(log_w):
    log_w = np.asarray(log_w, dtype=np.float64)
    max_log_w = np.max(log_w)
    w = np.exp(log_w - max_log_w)
    w_sum = np.sum(w)
    if not np.isfinite(w_sum) or w_sum <= 0:
        return np.full_like(w, 1.0 / len(w), dtype=np.float64)
    return w / w_sum


def _weighted_hpd_from_samples(theta_samples, weights, alpha):
    idx = np.argsort(weights)[::-1]
    cdf = np.cumsum(weights[idx])
    keep = idx[: np.searchsorted(cdf, 1.0 - alpha, side="left") + 1]
    selected = theta_samples[keep]

    lows = np.min(selected, axis=0)
    highs = np.max(selected, axis=0)
    intervals = np.stack([lows, highs], axis=1)
    interval_combined = np.asarray([np.mean(lows), np.mean(highs)], dtype=float)
    return intervals, interval_combined


def _evaluate_log_ratio_on_samples(model, data_point, theta_samples, eval_batch_size=32768):
    model_device = next(model.parameters()).device
    n_theta = theta_samples.shape[0]
    d = theta_samples.shape[1]

    data_point = np.asarray(data_point, dtype=np.float32)
    log_ratio = np.empty(n_theta, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n_theta, eval_batch_size):
            end = min(start + eval_batch_size, n_theta)
            theta_b = theta_samples[start:end]
            data_b = np.broadcast_to(data_point, (end - start, d)).astype(np.float32)
            inp = np.concatenate([data_b, theta_b], axis=1)
            inp_t = torch.from_numpy(inp).to(model_device)
            out = model(inp_t).detach().cpu().numpy().reshape(-1)
            out = np.clip(out, 1e-9, 1 - 1e-9)
            log_ratio[start:end] = np.log(out) - np.log1p(-out)

    return log_ratio


def get_posteriors_and_errors_qmc(
    inference_data,
    all_parameters_in_range,
    models,
    priors,
    device,
    n_qmc_samples=2**16,
    eval_batch_size=32768,
    max_model_evals_per_prior=int(2e15),
    qmc_seed=2026,
    show_progress=True,
):
    _, _, sampled_data = inference_data
    n_posterior_combinations, n_repititions_per_parameter, _ = sampled_data.shape

    all_posteriors = {}
    all_hpds_posterior, all_hpds_ratio = {}, {}
    all_ratios = {}

    requested_n_qmc = int(n_qmc_samples)
    safe_n_qmc = max(
        1024,
        min(
            requested_n_qmc,
            max_model_evals_per_prior // max(n_posterior_combinations * n_repititions_per_parameter, 1),
        ),
    )

    if safe_n_qmc < requested_n_qmc:
        print(
            f"[QMC] Reducing n_qmc_samples from {requested_n_qmc} to {safe_n_qmc} "
            f"to respect max_model_evals_per_prior={max_model_evals_per_prior}."
        )

    theta_samples = _draw_qmc_theta_samples(
        all_parameters_in_range,
        n_samples=safe_n_qmc,
        seed=qmc_seed,
        scramble=True,
        dtype=torch.float32,
    )

    model_lookup = _get_model_lookup(models)
    total_iterations = n_posterior_combinations * n_repititions_per_parameter * len(priors)
    progress = tqdm(
        total=total_iterations,
        desc="Calculating posteriors and errors (QMC samples)...",
        disable=not show_progress,
    )

    with progress as pbar:
        for prior in priors:
            prior_name = prior.__name__
            model = model_lookup[prior_name].to(device).eval()

            prior_vals = np.asarray(prior(theta_samples), dtype=np.float64)
            prior_vals = np.clip(prior_vals, 1e-300, None)
            log_prior = np.log(prior_vals)

            posteriors_grouped, ratios_grouped = [], []
            errors_posterior, errors_ratio = [], []

            for i_posterior_combination in range(n_posterior_combinations):
                sampled_block = sampled_data[i_posterior_combination].astype(np.float32)
                log_post_sum = np.zeros(safe_n_qmc, dtype=np.float64)
                log_ratio_sum = np.zeros(safe_n_qmc, dtype=np.float64)

                for data_point in sampled_block:
                    lratio = _evaluate_log_ratio_on_samples(
                        model,
                        data_point,
                        theta_samples,
                        eval_batch_size=eval_batch_size,
                    )
                    log_ratio_sum += lratio
                    log_post_sum += lratio + log_prior
                    pbar.update(1)

                post_w = _normalize_log_weights(log_post_sum)
                ratio_w = _normalize_log_weights(log_ratio_sum)

                map_post = theta_samples[int(np.argmax(log_post_sum))].astype(float)
                map_ratio = theta_samples[int(np.argmax(log_ratio_sum))].astype(float)

                posteriors_grouped.append({"map": map_post})
                ratios_grouped.append({"map": map_ratio})

                hpd_68_posterior = _weighted_hpd_from_samples(theta_samples, post_w, alpha=0.32)
                hpd_95_posterior = _weighted_hpd_from_samples(theta_samples, post_w, alpha=0.05)
                errors_posterior.append({"68": hpd_68_posterior, "95": hpd_95_posterior})

                hpd_68_ratio = _weighted_hpd_from_samples(theta_samples, ratio_w, alpha=0.32)
                hpd_95_ratio = _weighted_hpd_from_samples(theta_samples, ratio_w, alpha=0.05)
                errors_ratio.append({"68": hpd_68_ratio, "95": hpd_95_ratio})

            all_hpds_posterior[prior_name] = errors_posterior
            all_hpds_ratio[prior_name] = errors_ratio
            all_posteriors[prior_name] = posteriors_grouped
            all_ratios[prior_name] = ratios_grouped

            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_posteriors, all_ratios, all_hpds_posterior, all_hpds_ratio


def get_posteriors_and_errors(
    inference_data,
    all_parameters_in_range,
    models,
    priors,
    device,
    n_qmc_samples=2**16,
    eval_batch_size=32768,
    max_model_evals_per_prior=int(2e15),
    qmc_seed=2026,
    show_progress=True,
):
    return get_posteriors_and_errors_qmc(
        inference_data,
        all_parameters_in_range,
        models=models,
        priors=priors,
        device=device,
        n_qmc_samples=n_qmc_samples,
        eval_batch_size=eval_batch_size,
        max_model_evals_per_prior=max_model_evals_per_prior,
        qmc_seed=qmc_seed,
        show_progress=show_progress,
    )


def get_first_column_scatter_data(inference_data, all_posteriors, parameters_post, hpds):
    _, parameter_combinations, _ = inference_data
    true_params = np.asarray(parameter_combinations, dtype=float)
    n_dims = len(parameters_post)

    if true_params.ndim != 2 or true_params.shape[1] != n_dims:
        raise ValueError(
            f"Expected true parameters with shape (N, {n_dims}), got {true_params.shape}."
        )

    def _parse_hpd_68(entry):
        raw = entry["68"]
        if isinstance(raw, dict):
            intervals = raw.get("intervals", raw.get("interval", None))
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            intervals = raw[0]
        else:
            intervals = raw

        intervals = np.asarray(intervals, dtype=float)
        if intervals.shape != (n_dims, 2):
            raise ValueError(
                f"Expected HPD intervals with shape ({n_dims}, 2), got {intervals.shape}."
            )
        return intervals

    def _aggregate_curve(x, y, decimals=12):
        x = np.round(np.asarray(x, dtype=float), decimals=decimals)
        y = np.asarray(y, dtype=float)
        x_unique, inv = np.unique(x, return_inverse=True)
        y_sum = np.zeros(len(x_unique), dtype=float)
        counts = np.zeros(len(x_unique), dtype=float)
        np.add.at(y_sum, inv, y)
        np.add.at(counts, inv, 1.0)
        return x_unique, y_sum / np.clip(counts, 1.0, None)

    def _combine_dim_curves(curves):
        x_all = np.concatenate([c[0] for c in curves])
        y_all = np.concatenate([c[1] for c in curves])
        return _aggregate_curve(x_all, y_all)

    first_column_data = {}

    for prior_name, errors in hpds.items():
        if prior_name not in all_posteriors:
            raise KeyError(f"Prior '{prior_name}' not found in all_posteriors.")

        n_points = len(errors)
        if n_points != len(all_posteriors[prior_name]):
            raise ValueError(
                f"For prior '{prior_name}', HPD count ({n_points}) does not match "
                f"posterior count ({len(all_posteriors[prior_name])})."
            )
        if n_points != true_params.shape[0]:
            raise ValueError(
                f"For prior '{prior_name}', HPD count ({n_points}) does not match "
                f"true points ({true_params.shape[0]})."
            )

        width_dims = np.zeros((n_points, n_dims), dtype=float)
        bias_dims = np.zeros((n_points, n_dims), dtype=float)

        for i, entry in enumerate(errors):
            intervals = _parse_hpd_68(entry)
            lows, highs = intervals[:, 0], intervals[:, 1]
            width_dims[i, :] = highs - lows
            bias_dims[i, :] = 0.5 * (lows + highs) - true_params[i, :]

        width_curves = [_aggregate_curve(true_params[:, d], width_dims[:, d]) for d in range(n_dims)]
        bias_curves = [_aggregate_curve(true_params[:, d], bias_dims[:, d]) for d in range(n_dims)]

        x_comb, bias_comb = _combine_dim_curves(bias_curves)
        _, width_comb = _combine_dim_curves(width_curves)

        first_column_data[prior_name] = {
            "x": x_comb,
            "avg_bias": bias_comb,
            "avg_width_68": width_comb,
        }

    return first_column_data
