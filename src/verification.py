from __future__ import annotations

import concurrent.futures
import multiprocessing as mp

import numpy as np
import torch

from .data import draw_data
from .posterior import _build_model_from_state, _select_num_cuda_workers, _state_dict_to_cpu


def _get_model_lookup(models):
    if isinstance(models, dict):
        return models
    if isinstance(models, (list, tuple)) and models:
        if isinstance(models[0], dict):
            return models[0]
    raise TypeError("models must be a dict or a non-empty list/tuple of dicts.")


def _as_2d_data(data):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        return data.reshape(1, -1)
    if data.ndim == 2:
        return data
    raise ValueError("Expected scalar, 1D, or 2D data input.")


def _parameter_matrix(parameter, n_samples, n_dimensions):
    parameter = np.asarray(parameter, dtype=np.float32)
    if parameter.ndim == 0:
        return np.full((n_samples, n_dimensions), float(parameter), dtype=np.float32)
    if parameter.ndim == 1:
        if parameter.shape[0] == n_dimensions:
            return np.broadcast_to(parameter, (n_samples, n_dimensions)).astype(np.float32)
        if n_dimensions == 1 and parameter.shape[0] == n_samples:
            return parameter.reshape(n_samples, 1).astype(np.float32)
    if parameter.ndim == 2 and parameter.shape == (n_samples, n_dimensions):
        return parameter.astype(np.float32)
    raise ValueError("Parameter shape is incompatible with the provided data.")


def compute_ratio(data, parameter, model, device):
    data_matrix = _as_2d_data(data)
    n_samples, n_dimensions = data_matrix.shape
    parameter_matrix = _parameter_matrix(parameter, n_samples, n_dimensions)
    model_device = next(model.parameters()).device

    model_in = torch.as_tensor(
        np.concatenate([data_matrix, parameter_matrix], axis=1),
        dtype=torch.float32,
        device=model_device,
    )
    model_out = model(model_in).detach().cpu().numpy().flatten()
    model_out = np.clip(model_out, 1e-9, 1 - 1e-9)
    ratio = model_out / (1 - model_out)
    return ratio, model_out


def calculate_individual_ratios(data_inputs, parameter, models, priors, device):
    data_inputs = _as_2d_data(data_inputs)
    model_lookup = _get_model_lookup(models)

    ratios = {}
    for prior in priors:
        ratio_values, _ = compute_ratio(
            data_inputs,
            parameter,
            model_lookup[prior.__name__],
            device,
        )
        ratios[prior.__name__] = ratio_values.tolist()
    return ratios


def _calculate_prior_ratio_series(prior_name, model, data_inputs, test_parameters, device):
    ratio_series = []
    for parameter in test_parameters:
        ratio_values, _ = compute_ratio(data_inputs, parameter, model, device)
        ratio_series.append(ratio_values.tolist())
    return prior_name, ratio_series


def _ratio_worker(task):
    device = torch.device(task["device"])
    model = _build_model_from_state(task["config"], task["model_state"], device)
    return _calculate_prior_ratio_series(
        task["prior_name"],
        model,
        task["data_inputs"],
        task["test_parameters"],
        device,
    )


def calculate_all_ratios(data_inputs, test_parameters, models, priors, device, config=None, max_gpus=1):
    all_ratios = {}
    model_lookup = _get_model_lookup(models)
    n_cuda_workers = _select_num_cuda_workers(device, max_gpus, len(priors))

    if n_cuda_workers <= 1:
        for prior in priors:
            prior_name, ratio_series = _calculate_prior_ratio_series(
                prior.__name__,
                model_lookup[prior.__name__],
                data_inputs,
                test_parameters,
                device,
            )
            all_ratios[prior_name] = ratio_series
        return all_ratios

    if config is None:
        raise ValueError("config is required when max_gpus > 1 for verification parallelism.")

    ctx = mp.get_context("spawn")
    device_names = [f"cuda:{idx}" for idx in range(n_cuda_workers)]
    tasks = [
        {
            "prior_name": prior.__name__,
            "model_state": _state_dict_to_cpu(model_lookup[prior.__name__]),
            "config": config,
            "data_inputs": _as_2d_data(data_inputs),
            "test_parameters": list(test_parameters),
            "device": device_names[idx % n_cuda_workers],
        }
        for idx, prior in enumerate(priors)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cuda_workers, mp_context=ctx) as executor:
        future_map = {executor.submit(_ratio_worker, task): task["prior_name"] for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            prior_name, ratio_series = future.result()
            all_ratios[prior_name] = ratio_series
    return all_ratios


def _parameter_vector(value, n_dimensions):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 0:
        return np.full(n_dimensions, float(value), dtype=np.float32)
    if value.shape == (n_dimensions,):
        return value.astype(np.float32)
    raise ValueError("Each test parameter must be scalar or length n_dimensions.")


def _build_reweight_inputs(parameter_vectors, std_dev, n_test_data, generator):
    parameter_array = np.asarray(parameter_vectors, dtype=np.float32)
    noise = generator.normal(
        loc=0.0,
        scale=float(std_dev),
        size=(len(parameter_vectors), int(n_test_data), parameter_array.shape[1]),
    ).astype(np.float32)
    return parameter_array[:, None, :] + noise


def _compute_prior_reweight_result(
    prior_name,
    model,
    parameter_vectors,
    data_x_full,
    projection_dim,
    n_bins,
    device,
):
    results = [compute_ratio(data_x_full[0], parameter, model, device) for parameter in parameter_vectors]
    ratios = [result[0] for result in results]
    model_outs = [result[1] for result in results]

    reweighting = ratios[1] / np.clip(ratios[0], 1e-12, None)
    data_x_plot = [data[:, projection_dim] for data in data_x_full]
    reweighted_distribution, edges = np.histogram(
        data_x_plot[0],
        bins=n_bins,
        weights=reweighting,
        density=False,
    )
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return prior_name, data_x_plot, ratios, (bin_centers, reweighted_distribution), reweighting, model_outs


def _reweight_worker(task):
    device = torch.device(task["device"])
    model = _build_model_from_state(task["config"], task["model_state"], device)
    return _compute_prior_reweight_result(
        task["prior_name"],
        model,
        task["parameter_vectors"],
        task["data_x_full"],
        task["projection_dim"],
        task["n_bins"],
        device,
    )


def reweight_distributions(
    test_parameters,
    model_arr,
    priors,
    config,
    generator,
    device,
    n_test_data=int(1e6),
    n_bins=int(1e3),
    projection_dim=0,
    max_gpus=1,
):
    n_dimensions = config["data"]["n_parameters"]
    std_dev = config["data"]["std_dev"]
    parameter_vectors = [_parameter_vector(value, n_dimensions) for value in test_parameters]
    data_x_full = _build_reweight_inputs(parameter_vectors, std_dev, n_test_data, generator)
    data_x_plot = [data[:, projection_dim] for data in data_x_full]

    model_lookup = model_arr if isinstance(model_arr, dict) else _get_model_lookup(model_arr)
    n_cuda_workers = _select_num_cuda_workers(device, max_gpus, len(priors))

    all_outs = {}
    reweighting_distributions = {}
    reweighted_distributions = {}

    if n_cuda_workers <= 1:
        for prior in priors:
            prior_name, _, _, reweighted_distribution, reweighting, model_outs = _compute_prior_reweight_result(
                prior.__name__,
                model_lookup[prior.__name__],
                parameter_vectors,
                data_x_full,
                projection_dim,
                n_bins,
                device,
            )
            all_outs[prior_name] = model_outs
            reweighting_distributions[prior_name] = reweighting
            reweighted_distributions[prior_name] = reweighted_distribution
        return data_x_plot, reweighted_distributions, reweighting_distributions, all_outs

    ctx = mp.get_context("spawn")
    device_names = [f"cuda:{idx}" for idx in range(n_cuda_workers)]
    tasks = [
        {
            "prior_name": prior.__name__,
            "model_state": _state_dict_to_cpu(model_lookup[prior.__name__]),
            "config": config,
            "parameter_vectors": parameter_vectors,
            "data_x_full": data_x_full,
            "projection_dim": projection_dim,
            "n_bins": n_bins,
            "device": device_names[idx % n_cuda_workers],
        }
        for idx, prior in enumerate(priors)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cuda_workers, mp_context=ctx) as executor:
        future_map = {executor.submit(_reweight_worker, task): task["prior_name"] for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            prior_name, _, _, reweighted_distribution, reweighting, model_outs = future.result()
            all_outs[prior_name] = model_outs
            reweighting_distributions[prior_name] = reweighting
            reweighted_distributions[prior_name] = reweighted_distribution

    return data_x_plot, reweighted_distributions, reweighting_distributions, all_outs


def _safe_logit(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return np.log(probabilities) - np.log1p(-probabilities)


def _normalize_log_weights(log_weights):
    log_weights = np.asarray(log_weights, dtype=np.float64).ravel()
    if log_weights.size == 0:
        return np.asarray([], dtype=np.float64)
    max_log_weight = np.max(log_weights)
    shifted = np.exp(log_weights - max_log_weight)
    total = np.sum(shifted)
    if not np.isfinite(total) or total <= 0:
        return np.full(log_weights.shape, 1.0 / log_weights.size, dtype=np.float64)
    return shifted / total


def compute_effective_sample_size(weights):
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.size == 0:
        return 0.0
    weight_sum = np.sum(weights)
    weight_sq_sum = np.sum(np.square(weights))
    if not np.isfinite(weight_sum) or not np.isfinite(weight_sq_sum) or weight_sq_sum <= 0:
        return 0.0
    return float((weight_sum**2) / weight_sq_sum)


def _rankdata_average(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    ranks = np.empty(values.shape[0], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman_correlation(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    rank_x = _rankdata_average(x)
    rank_y = _rankdata_average(y)
    std_x = np.std(rank_x)
    std_y = np.std(rank_y)
    if std_x <= 0 or std_y <= 0:
        return float("nan")
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def sample_theta_pairs(pair_count, parameter_range, n_dimensions, generator):
    pair_count = int(pair_count)
    if pair_count < 1:
        raise ValueError("pair_count must be at least 1.")

    low, high = parameter_range
    theta_0 = generator.uniform(low=low, high=high, size=(pair_count, int(n_dimensions))).astype(np.float32)
    theta_1 = generator.uniform(low=low, high=high, size=(pair_count, int(n_dimensions))).astype(np.float32)
    distances = np.linalg.norm(theta_1 - theta_0, axis=1).astype(np.float64)
    return theta_0, theta_1, distances


def _sample_data_for_theta_pairs(theta, n_samples_per_pair, config, generator):
    theta = np.asarray(theta, dtype=np.float32)
    n_samples_per_pair = int(n_samples_per_pair)
    if n_samples_per_pair < 1:
        raise ValueError("n_samples_per_pair must be at least 1.")
    repeated_theta = np.repeat(theta, n_samples_per_pair, axis=0)
    sampled = draw_data(repeated_theta, config, generator)
    return sampled.reshape(theta.shape[0], n_samples_per_pair, theta.shape[1])


def sample_unit_directions(n_projections, n_dimensions, generator):
    n_projections = int(n_projections)
    if n_projections < 1:
        raise ValueError("n_projections must be at least 1.")

    directions = generator.normal(size=(n_projections, int(n_dimensions))).astype(np.float64)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    zero_mask = np.squeeze(norms <= 0, axis=1)
    if np.any(zero_mask):
        directions[zero_mask, 0] = 1.0
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return (directions / norms).astype(np.float32)


def build_quantitative_verification_bundle(
    config,
    generator,
    pair_count,
    model_samples_per_endpoint,
    reweight_source_samples,
    reweight_target_samples,
    swd_projections,
):
    n_dimensions = config["data"]["n_parameters"]
    parameter_range = config["data"]["parameter_range"]

    theta_0, theta_1, distances = sample_theta_pairs(
        pair_count=pair_count,
        parameter_range=parameter_range,
        n_dimensions=n_dimensions,
        generator=generator,
    )

    model_eval_source = _sample_data_for_theta_pairs(theta_0, model_samples_per_endpoint, config, generator)
    model_eval_target = _sample_data_for_theta_pairs(theta_1, model_samples_per_endpoint, config, generator)
    model_eval_data = np.concatenate([model_eval_source, model_eval_target], axis=1).astype(np.float32)

    reweight_source_data = _sample_data_for_theta_pairs(theta_0, reweight_source_samples, config, generator)
    reweight_target_data = _sample_data_for_theta_pairs(theta_1, reweight_target_samples, config, generator)
    swd_directions = sample_unit_directions(
        n_projections=swd_projections,
        n_dimensions=n_dimensions,
        generator=generator,
    )

    return {
        "theta_0": theta_0,
        "theta_1": theta_1,
        "distances": distances,
        "model_eval_data": model_eval_data,
        "reweight_source_data": reweight_source_data.astype(np.float32),
        "reweight_target_data": reweight_target_data.astype(np.float32),
        "swd_directions": swd_directions,
        "std_dev": float(config["data"]["std_dev"]),
    }


def exact_log_r10_gaussian(data, theta_0, theta_1, std_dev):
    data_matrix = _as_2d_data(data).astype(np.float64)
    n_samples, n_dimensions = data_matrix.shape
    theta_0_matrix = _parameter_matrix(theta_0, n_samples, n_dimensions).astype(np.float64)
    theta_1_matrix = _parameter_matrix(theta_1, n_samples, n_dimensions).astype(np.float64)
    sigma_sq = float(std_dev) ** 2
    if sigma_sq <= 0:
        raise ValueError("std_dev must be strictly positive.")

    sq_dist_0 = np.sum(np.square(data_matrix - theta_0_matrix), axis=1)
    sq_dist_1 = np.sum(np.square(data_matrix - theta_1_matrix), axis=1)
    return (sq_dist_0 - sq_dist_1) / (2.0 * sigma_sq)


def predict_pairwise_log_ratio(data, theta_0, theta_1, model, device):
    _, model_out_0 = compute_ratio(data, theta_0, model, device)
    _, model_out_1 = compute_ratio(data, theta_1, model, device)
    return _safe_logit(model_out_1) - _safe_logit(model_out_0)


def _weighted_wasserstein_1d(source_values, target_values, source_weights, target_weights):
    source_values = np.asarray(source_values, dtype=np.float64).ravel()
    target_values = np.asarray(target_values, dtype=np.float64).ravel()
    source_weights = np.asarray(source_weights, dtype=np.float64).ravel()
    target_weights = np.asarray(target_weights, dtype=np.float64).ravel()

    if source_values.size == 0 or target_values.size == 0:
        return 0.0

    source_weights = source_weights / np.sum(source_weights)
    target_weights = target_weights / np.sum(target_weights)

    values = np.concatenate([source_values, target_values])
    deltas = np.concatenate([source_weights, -target_weights])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_deltas = deltas[order]

    if sorted_values.size < 2:
        return 0.0

    cumulative_diff = np.cumsum(sorted_deltas)[:-1]
    intervals = np.diff(sorted_values)
    return float(np.sum(np.abs(cumulative_diff) * intervals))


def weighted_sliced_wasserstein_distance(
    source,
    target,
    source_weights=None,
    target_weights=None,
    directions=None,
    n_projections=32,
    generator=None,
):
    source = _as_2d_data(source).astype(np.float64)
    target = _as_2d_data(target).astype(np.float64)
    n_dimensions = source.shape[1]

    if target.shape[1] != n_dimensions:
        raise ValueError("source and target must have the same number of dimensions.")

    if source_weights is None:
        source_weights = np.full(source.shape[0], 1.0 / source.shape[0], dtype=np.float64)
    else:
        source_weights = np.asarray(source_weights, dtype=np.float64).ravel()
    if target_weights is None:
        target_weights = np.full(target.shape[0], 1.0 / target.shape[0], dtype=np.float64)
    else:
        target_weights = np.asarray(target_weights, dtype=np.float64).ravel()

    if directions is None:
        if generator is None:
            generator = np.random.default_rng(0)
        directions = sample_unit_directions(n_projections, n_dimensions, generator)
    else:
        directions = np.asarray(directions, dtype=np.float64)

    distances = []
    for direction in directions:
        source_projection = source @ direction
        target_projection = target @ direction
        distances.append(
            _weighted_wasserstein_1d(
                source_projection,
                target_projection,
                source_weights=source_weights,
                target_weights=target_weights,
            )
        )
    return float(np.mean(distances)) if distances else 0.0


def _summarize_model_quality(pair_metrics, pooled_exact, pooled_predicted):
    rmse = np.asarray(pair_metrics["rmse_log_r10"], dtype=np.float64)
    distances = np.asarray(pair_metrics["distance"], dtype=np.float64)
    pooled_exact = np.asarray(pooled_exact, dtype=np.float64)
    pooled_predicted = np.asarray(pooled_predicted, dtype=np.float64)

    return {
        "median_rmse_log_r10": float(np.median(rmse)),
        "worst_rmse_log_r10": float(np.max(rmse)),
        "pooled_rmse_log_r10": float(np.sqrt(np.mean(np.square(pooled_predicted - pooled_exact)))),
        "spearman_rmse_vs_distance": _spearman_correlation(distances, rmse),
    }


def _evaluate_prior_pairwise_log_ratio_quality(prior_name, model, bundle, device):
    theta_0 = np.asarray(bundle["theta_0"], dtype=np.float32)
    theta_1 = np.asarray(bundle["theta_1"], dtype=np.float32)
    heldout_data = np.asarray(bundle["model_eval_data"], dtype=np.float32)
    distances = np.asarray(bundle["distances"], dtype=np.float64)
    std_dev = float(bundle["std_dev"])

    pair_metrics = {
        "distance": distances.copy(),
        "rmse_log_r10": np.zeros(theta_0.shape[0], dtype=np.float64),
        "mae_log_r10": np.zeros(theta_0.shape[0], dtype=np.float64),
        "max_abs_log_r10": np.zeros(theta_0.shape[0], dtype=np.float64),
    }
    pooled_exact = []
    pooled_predicted = []

    for idx in range(theta_0.shape[0]):
        data_batch = heldout_data[idx]
        exact_log_ratio = exact_log_r10_gaussian(data_batch, theta_0[idx], theta_1[idx], std_dev)
        predicted_log_ratio = predict_pairwise_log_ratio(data_batch, theta_0[idx], theta_1[idx], model, device)
        error = predicted_log_ratio - exact_log_ratio

        pair_metrics["rmse_log_r10"][idx] = float(np.sqrt(np.mean(np.square(error))))
        pair_metrics["mae_log_r10"][idx] = float(np.mean(np.abs(error)))
        pair_metrics["max_abs_log_r10"][idx] = float(np.max(np.abs(error)))
        pooled_exact.append(exact_log_ratio.astype(np.float32))
        pooled_predicted.append(predicted_log_ratio.astype(np.float32))

    pooled_exact = np.concatenate(pooled_exact) if pooled_exact else np.asarray([], dtype=np.float32)
    pooled_predicted = np.concatenate(pooled_predicted) if pooled_predicted else np.asarray([], dtype=np.float32)
    return prior_name, {
        "pair_metrics": pair_metrics,
        "pooled": {
            "exact_log_r10": pooled_exact,
            "predicted_log_r10": pooled_predicted,
        },
        "summary": _summarize_model_quality(pair_metrics, pooled_exact, pooled_predicted),
    }


def _model_quality_worker(task):
    device = torch.device(task["device"])
    model = _build_model_from_state(task["config"], task["model_state"], device)
    return _evaluate_prior_pairwise_log_ratio_quality(task["prior_name"], model, task["bundle"], device)


def evaluate_pairwise_log_ratio_quality(bundle, models, priors, device, config=None, max_gpus=1):
    results = {}
    model_lookup = _get_model_lookup(models)
    n_cuda_workers = _select_num_cuda_workers(device, max_gpus, len(priors))

    if n_cuda_workers <= 1:
        for prior in priors:
            prior_name, result = _evaluate_prior_pairwise_log_ratio_quality(
                prior.__name__,
                model_lookup[prior.__name__],
                bundle,
                device,
            )
            results[prior_name] = result
        return results

    if config is None:
        raise ValueError("config is required when max_gpus > 1 for verification parallelism.")

    ctx = mp.get_context("spawn")
    device_names = [f"cuda:{idx}" for idx in range(n_cuda_workers)]
    tasks = [
        {
            "prior_name": prior.__name__,
            "model_state": _state_dict_to_cpu(model_lookup[prior.__name__]),
            "config": config,
            "bundle": bundle,
            "device": device_names[idx % n_cuda_workers],
        }
        for idx, prior in enumerate(priors)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cuda_workers, mp_context=ctx) as executor:
        future_map = {executor.submit(_model_quality_worker, task): task["prior_name"] for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            prior_name, result = future.result()
            results[prior_name] = result
    return results


def _summarize_reweighting_quality(pair_metrics):
    swd = np.asarray(pair_metrics["swd"], dtype=np.float64)
    ess_fraction = np.asarray(pair_metrics["ess_fraction"], dtype=np.float64)
    distances = np.asarray(pair_metrics["distance"], dtype=np.float64)

    return {
        "median_swd": float(np.median(swd)),
        "worst_swd": float(np.max(swd)),
        "median_ess_fraction": float(np.median(ess_fraction)),
        "worst_ess_fraction": float(np.min(ess_fraction)),
        "spearman_swd_vs_distance": _spearman_correlation(distances, swd),
    }


def _evaluate_prior_nd_reweighting_quality(prior_name, model, bundle, device):
    theta_0 = np.asarray(bundle["theta_0"], dtype=np.float32)
    theta_1 = np.asarray(bundle["theta_1"], dtype=np.float32)
    source_data = np.asarray(bundle["reweight_source_data"], dtype=np.float32)
    target_data = np.asarray(bundle["reweight_target_data"], dtype=np.float32)
    directions = np.asarray(bundle["swd_directions"], dtype=np.float32)
    distances = np.asarray(bundle["distances"], dtype=np.float64)

    pair_metrics = {
        "distance": distances.copy(),
        "swd": np.zeros(theta_0.shape[0], dtype=np.float64),
        "ess": np.zeros(theta_0.shape[0], dtype=np.float64),
        "ess_fraction": np.zeros(theta_0.shape[0], dtype=np.float64),
    }

    for idx in range(theta_0.shape[0]):
        predicted_log_weights = predict_pairwise_log_ratio(source_data[idx], theta_0[idx], theta_1[idx], model, device)
        normalized_weights = _normalize_log_weights(predicted_log_weights)
        ess = compute_effective_sample_size(normalized_weights)
        swd = weighted_sliced_wasserstein_distance(
            source_data[idx],
            target_data[idx],
            source_weights=normalized_weights,
            directions=directions,
        )

        pair_metrics["swd"][idx] = swd
        pair_metrics["ess"][idx] = ess
        pair_metrics["ess_fraction"][idx] = ess / max(1, source_data[idx].shape[0])

    return prior_name, {
        "pair_metrics": pair_metrics,
        "summary": _summarize_reweighting_quality(pair_metrics),
    }


def _nd_reweight_worker(task):
    device = torch.device(task["device"])
    model = _build_model_from_state(task["config"], task["model_state"], device)
    return _evaluate_prior_nd_reweighting_quality(task["prior_name"], model, task["bundle"], device)


def evaluate_nd_reweighting_quality(bundle, models, priors, device, config=None, max_gpus=1):
    results = {}
    model_lookup = _get_model_lookup(models)
    n_cuda_workers = _select_num_cuda_workers(device, max_gpus, len(priors))

    if n_cuda_workers <= 1:
        for prior in priors:
            prior_name, result = _evaluate_prior_nd_reweighting_quality(
                prior.__name__,
                model_lookup[prior.__name__],
                bundle,
                device,
            )
            results[prior_name] = result
        return results

    if config is None:
        raise ValueError("config is required when max_gpus > 1 for verification parallelism.")

    ctx = mp.get_context("spawn")
    device_names = [f"cuda:{idx}" for idx in range(n_cuda_workers)]
    tasks = [
        {
            "prior_name": prior.__name__,
            "model_state": _state_dict_to_cpu(model_lookup[prior.__name__]),
            "config": config,
            "bundle": bundle,
            "device": device_names[idx % n_cuda_workers],
        }
        for idx, prior in enumerate(priors)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cuda_workers, mp_context=ctx) as executor:
        future_map = {executor.submit(_nd_reweight_worker, task): task["prior_name"] for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            prior_name, result = future.result()
            results[prior_name] = result
    return results
