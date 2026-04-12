from __future__ import annotations

import numpy as np
import torch


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

    model_in = torch.tensor(
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


def calculate_all_ratios(data_inputs, test_parameters, models, priors, device):
    all_ratios = {}
    for parameter in test_parameters:
        ratios = calculate_individual_ratios(data_inputs, parameter, models, priors, device)
        for prior_name, ratio_values in ratios.items():
            all_ratios.setdefault(prior_name, []).append(ratio_values)
    return all_ratios


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
):
    n_dimensions = config["data"]["n_parameters"]
    std_dev = config["data"]["std_dev"]
    covariance = np.eye(n_dimensions) * std_dev**2

    def _parameter_vector(value):
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 0:
            return np.full(n_dimensions, float(value), dtype=np.float32)
        if value.shape == (n_dimensions,):
            return value.astype(np.float32)
        raise ValueError("Each test parameter must be scalar or length n_dimensions.")

    parameter_vectors = [_parameter_vector(value) for value in test_parameters]
    data_x_full = [
        generator.multivariate_normal(mean=parameter, cov=covariance, size=n_test_data)
        for parameter in parameter_vectors
    ]
    data_x_plot = [data[:, projection_dim] for data in data_x_full]

    model_lookup = model_arr if isinstance(model_arr, dict) else _get_model_lookup(model_arr)

    all_ratios = {}
    all_outs = {}
    reweighting_distributions = {}
    reweighted_distributions = {}

    for prior in priors:
        model = model_lookup[prior.__name__]
        results = [compute_ratio(data_x_full[0], parameter, model, device) for parameter in parameter_vectors]
        ratios = [result[0] for result in results]
        model_outs = [result[1] for result in results]
        all_ratios[prior.__name__] = ratios
        all_outs[prior.__name__] = model_outs

        reweighting = ratios[1] / np.clip(ratios[0], 1e-12, None)
        reweighting_distributions[prior.__name__] = reweighting
        reweighted_distribution, edges = np.histogram(
            data_x_plot[0],
            bins=n_bins,
            weights=reweighting,
            density=False,
        )
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        reweighted_distributions[prior.__name__] = (bin_centers, reweighted_distribution)

    return data_x_plot, reweighted_distributions, reweighting_distributions, all_outs
