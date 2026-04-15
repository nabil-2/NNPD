from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import numpy as np

def build_config(args: argparse.Namespace) -> dict:
    return {
        "data": {
            "prior_args": {
                "uniform": None,
                "normal": (5, 4),
                "exponential": (0.1,),
                "grid": (0.2,),
            },
            "parameter_range": (0, 10),
            "std_dev": args.std_dev,
            "n_parameters": args.n_parameters,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "n_validation": args.n_validation,
        },
        "classifier": {
            "n_inputs": args.n_parameters,
            "n_hidden_layers": args.n_hidden_layers,
            "n_units": args.n_units,
        },
        "training": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "model_iterations": args.model_iterations,
        },
    }


def close_figure(fig, plt_module) -> None:
    if fig is not None:
        plt_module.close(fig)


def save_json(data: dict, filename: Path | None) -> None:
    if filename is None:
        return
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(data, indent=2, sort_keys=True))


def _json_safe_value(value):
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) else value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_safe_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _verification_results_to_json(results: dict) -> dict:
    payload = {}
    for prior_name, result in results.items():
        payload[prior_name] = {
            "summary": _json_safe_value(result.get("summary", {})),
            "pair_metrics": _json_safe_value(result.get("pair_metrics", {})),
        }
    return payload


def _aggregate_curve(x, y, decimals=12):
    x = np.round(np.asarray(x, dtype=float), decimals=decimals)
    y = np.asarray(y, dtype=float)
    x_unique, inverse = np.unique(x, return_inverse=True)
    y_sum = np.zeros(len(x_unique), dtype=float)
    counts = np.zeros(len(x_unique), dtype=float)
    np.add.at(y_sum, inverse, y)
    np.add.at(counts, inverse, 1.0)
    return x_unique, y_sum / np.clip(counts, 1.0, None)


def _combine_dim_curves(curves):
    x_all = np.concatenate([curve[0] for curve in curves])
    y_all = np.concatenate([curve[1] for curve in curves])
    return _aggregate_curve(x_all, y_all)


def _build_width_curve(true_params: np.ndarray, intervals: np.ndarray) -> dict:
    n_dims = int(true_params.shape[1])
    width_dims = np.asarray(intervals[:, :, 1] - intervals[:, :, 0], dtype=float)
    width_curves = [_aggregate_curve(true_params[:, dim], width_dims[:, dim]) for dim in range(n_dims)]
    x_combined, avg_width = _combine_dim_curves(width_curves)
    return {
        "x": x_combined,
        "avg_width": avg_width,
    }


def _relative_path(path: str | Path | None, *, base_dir: Path | None) -> str | None:
    if path is None or base_dir is None:
        return None
    return os.path.relpath(str(Path(path).resolve()), str(base_dir.resolve()))


def _collect_array_ref_paths(value) -> set[str]:
    paths = set()
    if isinstance(value, dict):
        if {"path", "shape", "dtype"} <= set(value):
            paths.add(str(value["path"]))
        else:
            for item in value.values():
                paths.update(_collect_array_ref_paths(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            paths.update(_collect_array_ref_paths(item))
    return paths


def _summarize_hpd_collection(
    hpds_by_prior: dict,
    true_params: np.ndarray,
    *,
    n_dims: int,
    extract_hpd_level,
) -> dict:
    summary = {}
    for prior_name, raw_hpds in hpds_by_prior.items():
        prior_summary = {}
        for level in ("68", "95"):
            intervals, interval_combined = extract_hpd_level(raw_hpds, level, n_dims)
            intervals = np.asarray(intervals, dtype=float)
            interval_combined = np.asarray(interval_combined, dtype=float)
            widths = intervals[:, :, 1] - intervals[:, :, 0]
            combined_widths = interval_combined[:, 1] - interval_combined[:, 0]
            contains_true = (true_params >= intervals[:, :, 0]) & (true_params <= intervals[:, :, 1])

            prior_summary[level] = {
                "n_points": int(intervals.shape[0]),
                "coverage_fraction_per_dim": np.mean(contains_true, axis=0),
                "coverage_fraction_mean": float(np.mean(contains_true)),
                "mean_interval_width_per_dim": np.mean(widths, axis=0),
                "median_interval_width_per_dim": np.median(widths, axis=0),
                "mean_interval_width": float(np.mean(widths)),
                "median_interval_width": float(np.median(widths)),
                "mean_combined_width": float(np.mean(combined_widths)),
                "median_combined_width": float(np.median(combined_widths)),
            }
        summary[prior_name] = prior_summary
    return summary


def _build_analysis_bundle(
    *,
    args: argparse.Namespace,
    config: dict,
    run_info_payload: dict,
    inference_data,
    priors,
    all_parameters_in_range,
    all_posteriors: dict,
    all_ratios: dict,
    all_hpds_posterior: dict,
    all_hpds_ratio: dict,
    averaged_bias_error: dict,
    posterior_storage_dir: Path | None,
    filename_analysis_bundle: Path,
    filename_posteriors_data: Path | None,
    filename_scatter_data: Path | None,
    filename_run_info: Path | None,
    extract_hpd_level,
) -> dict:
    _, parameter_combinations, sampled_data = inference_data
    true_params = np.asarray(parameter_combinations, dtype=float)
    prior_names = [prior.__name__ for prior in priors]
    bundle_dir = filename_analysis_bundle.parent
    artifact_files = {
        "config_json": _relative_path(filename_analysis_bundle.parents[1] / "config.json", base_dir=bundle_dir),
        "results_data_pkl": _relative_path(filename_posteriors_data, base_dir=bundle_dir),
        "first_column_scatter_data_pkl": _relative_path(filename_scatter_data, base_dir=bundle_dir),
        "run_info_json": _relative_path(filename_run_info, base_dir=bundle_dir),
    }

    compact_store_files = {}
    for prior_name in prior_names:
        stored_paths = set()
        for value in (
            all_posteriors.get(prior_name),
            all_ratios.get(prior_name),
            all_hpds_posterior.get(prior_name),
            all_hpds_ratio.get(prior_name),
        ):
            stored_paths.update(_collect_array_ref_paths(value))
        compact_store_files[prior_name] = sorted(
            _relative_path(path, base_dir=bundle_dir) for path in stored_paths
        )

    bias_summary = {}
    for prior_name, prior_bias in averaged_bias_error.items():
        intervals_95, _ = extract_hpd_level(all_hpds_ratio[prior_name], level="95", n_dims=args.n_parameters)
        width_curve_95 = _build_width_curve(true_params, np.asarray(intervals_95, dtype=float))
        bias_summary[prior_name] = {
            "source": {
                "map": "ratio",
                "hpd": "ratio",
            },
            "x": prior_bias["x"],
            "avg_bias": prior_bias["avg_bias"],
            "avg_width_68": prior_bias["avg_width_68"],
            "width_curves": {
                "68": {
                    "x": prior_bias["x"],
                    "avg_width": prior_bias["avg_width_68"],
                },
                "95": width_curve_95,
            },
        }

    n_inferred_parameter_settings = int(true_params.shape[0])
    n_repetitions_per_setting = int(np.asarray(sampled_data).shape[1])

    return {
        "schema_version": 1,
        "run_metadata": {
            **run_info_payload,
            "effective_config": config,
        },
        "posterior_workload": {
            "prior_names": prior_names,
            "n_priors": int(len(prior_names)),
            "n_inferred_parameter_settings": n_inferred_parameter_settings,
            "n_repetitions_per_setting": n_repetitions_per_setting,
            "total_posterior_combinations_per_prior": (
                n_inferred_parameter_settings * n_repetitions_per_setting
            ),
            "total_posterior_combinations_all_priors": (
                n_inferred_parameter_settings * n_repetitions_per_setting * len(prior_names)
            ),
            "posterior_grid_shape": [len(axis) for axis in all_parameters_in_range],
            "compact_posterior_storage_written": posterior_storage_dir is not None,
        },
        "bias_summary": bias_summary,
        "hpd_summary": {
            "posterior": _summarize_hpd_collection(
                all_hpds_posterior,
                true_params,
                n_dims=args.n_parameters,
                extract_hpd_level=extract_hpd_level,
            ),
            "ratio": _summarize_hpd_collection(
                all_hpds_ratio,
                true_params,
                n_dims=args.n_parameters,
                extract_hpd_level=extract_hpd_level,
            ),
        },
        "artifacts": {
            "relative_to": "bundle_dir",
            "files": artifact_files,
            "compact_store_dir": _relative_path(posterior_storage_dir, base_dir=bundle_dir),
            "compact_store_files": compact_store_files,
        },
    }


def _action_dest(action: argparse.Action) -> str | None:
    if not getattr(action, "dest", None):
        return None
    if action.dest == argparse.SUPPRESS:
        return None
    return action.dest


def print_argument_summary(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    raw_argv: list[str],
) -> None:
    print("Argument summary")
    print(f"  raw argv: {raw_argv}")

    optional_overrides = {}
    optional_defaults_used = {}
    required_or_positional = {}

    for action in parser._actions:
        dest = _action_dest(action)
        if dest is None or not hasattr(args, dest):
            continue

        value = getattr(args, dest)
        default = action.default

        if action.option_strings:
            if default is argparse.SUPPRESS:
                optional_overrides[dest] = value
            elif value != default:
                optional_overrides[dest] = value
            else:
                optional_defaults_used[dest] = value
        else:
            required_or_positional[dest] = value

    print("  required/positional arguments:")
    print(json.dumps(required_or_positional, indent=2, sort_keys=True))

    print("  optional arguments overridden from defaults:")
    print(json.dumps(optional_overrides, indent=2, sort_keys=True))

    print("  optional arguments using defaults:")
    print(json.dumps(optional_defaults_used, indent=2, sort_keys=True))
    print("")


def run(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import numpy as np
    import torch
    from matplotlib import pyplot as plt

    from src.data import draw_data, get_data
    from src.models import BinaryClassifier
    from src.plotting import (
        _enforce_markers,
        plot_all_rocs,
        plot_error_and_hld_width,
        plot_errorbars,
        plot_log_ratio_error_vs_distance,
        plot_log_ratio_exact_vs_predicted,
        plot_prior_contours,
        plot_prior_sample_histograms,
        plot_ratio_violins,
        plot_reweighting_summary,
        plot_reweighted_distributions,
        plot_reweighting_swd_vs_distance,
        plot_training_tests,
    )
    from src.posterior import (
        _extract_hpd_level,
        create_inference_data,
        create_inference_parameters,
        get_first_column_scatter_data,
        get_posteriors_and_errors,
    )
    from src.priors import (
        build_priors,
        describe_prior_normalization,
        draw_grid_samples,
        get_alignment_support_axes,
        integrate_nd_vegas,
        snap_to_support_axis,
    )
    from src.training import test_model, train
    from src.utils import get_filepath
    from src.verification import (
        build_quantitative_verification_bundle,
        calculate_all_ratios,
        evaluate_nd_reweighting_quality,
        evaluate_pairwise_log_ratio_quality,
        reweight_distributions,
    )

    if args.n_parameters < 1:
        raise ValueError("n_parameters must be at least 1.")
    if args.max_gpus < 1:
        raise ValueError("max_gpus must be at least 1.")

    plt.rc("axes", prop_cycle=plt.cycler(color=plt.get_cmap("Set1").colors))

    def get_savepath(relative_path: str, config: dict, output_root: str, save: bool) -> Path | None:
        if not save:
            return None
        return get_filepath(relative_path, config, root=output_root)

    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")

    config = build_config(args)
    output_root = args.output_root

    if config["training"]["model_iterations"] != 1:
        raise ValueError(
            "model_iterations must be 1 for posterior and verification stages; "
            f"got {config['training']['model_iterations']}."
        )

    print(f"Using device: {device}")
    print(f"Running toy_example_nD_tidy with {args.n_parameters} parameter dimensions")
    print(json.dumps(config, indent=2, sort_keys=True))

    priors, prior_samplers, param_min, param_max = build_priors(config, generator)
    alignment_support_axes = get_alignment_support_axes(priors)
    if alignment_support_axes is not None:
        print(
            "Detected lattice-aligned prior support:",
            {
                "points_per_dim": [int(len(axis)) for axis in alignment_support_axes],
                "grid_step": float(getattr(priors[-1], "grid_step", 0.0)),
            },
        )

    if args.n_parameters >= 2:
        filename_prior_contours = get_savepath(
            "priors/prior_contours.pdf",
            config,
            output_root,
            args.save,
        )
        fig, _ = plot_prior_contours(
            priors,
            parameter_range=config["data"]["parameter_range"],
            n_parameters=config["data"]["n_parameters"],
        )
        if filename_prior_contours is not None:
            fig.savefig(filename_prior_contours)
            print(f"Saved prior contours to {filename_prior_contours}")
        close_figure(fig, plt)

        histogram_factor = max(1, math.ceil(args.prior_histogram_samples / config["data"]["n_train"]))
        filename_prior_histograms = get_savepath(
            "priors/prior_sample_histograms.pdf",
            config,
            output_root,
            args.save,
        )
        fig, _ = plot_prior_sample_histograms(
            prior_samplers,
            config,
            n_bins=args.prior_histogram_bins,
            factor_n=histogram_factor,
        )
        if filename_prior_histograms is not None:
            fig.savefig(filename_prior_histograms)
            print(f"Saved prior sample histograms to {filename_prior_histograms}")
        close_figure(fig, plt)
    else:
        print("Skipping 2D prior plots because n_parameters < 2.")

    for prior in priors:
        normalization = describe_prior_normalization(prior)
        normalization_value = normalization["value"]
        if normalization["label"] == "Integral over box" and normalization_value is None:
            integral = integrate_nd_vegas(
                prior,
                [(param_min, param_max)] * config["data"]["n_parameters"],
                nitn=args.vegas_nitn,
                neval=args.vegas_neval,
            )[0]
            normalization_value = float(integral)
        print(f"{normalization['label']} for {prior.__name__}: {normalization_value:.5f}")

    example_train_set, example_validation_set, example_test_set = get_data(
        prior_samplers[0],
        config,
        generator,
        show_output=True,
    )
    print(
        "Example dataset captured from the first prior:",
        {
            "train": tuple(example_train_set[0].shape),
            "validation": tuple(example_validation_set[0].shape),
            "test": tuple(example_test_set[0].shape),
        },
    )

    models = {}
    test_results = {}

    for prior_sampler in prior_samplers:
        prior_name = prior_sampler.__name__
        print(f"Training models for prior '{prior_name}'")
        train_set, validation_set, test_set = get_data(
            prior_sampler,
            config,
            generator,
            show_output=False,
        )
        model = BinaryClassifier(config)
        model, training_results = train(
            model,
            *train_set,
            *validation_set,
            config=config,
            device=device,
            show_output=args.verbose_training,
        )
        filename_plots = get_savepath(
            f"training/training_0/training_{prior_name}_prior.pdf",
            config,
            output_root,
            args.save,
        )
        filename_model = get_savepath(
            f"models/models_0/model_{prior_name}_prior.pth",
            config,
            output_root,
            args.save,
        )
        if filename_model is not None:
            torch.save(model.state_dict(), filename_model)
            print(f"Saved model to {filename_model}")
        test_result = test_model(
            model,
            *test_set,
            device=device,
            show_output=True,
        )
        fig, _ = plot_training_tests(
            training_results,
            test_result,
            f"0. {prior_name}",
            filename=filename_plots,
        )
        close_figure(fig, plt)
        models[prior_name] = model.to("cpu")
        test_results[prior_name] = test_result

    fig = plot_all_rocs(
        test_results,
        filename=get_savepath("training/all_roc_curves.pdf", config, output_root, args.save),
    )
    close_figure(fig, plt)

    n_parameter_grid_points = args.posterior_grid_points
    parameters_min_max = config["data"]["parameter_range"]
    all_parameters_in_range = [
        np.linspace(parameters_min_max[0], parameters_min_max[1], n_parameter_grid_points)
        for _ in range(config["data"]["n_parameters"])
    ]
    print(f"GPU: {device.type}; CPUs: {mp.cpu_count()}; requested max GPUs: {args.max_gpus}")

    filename_errorbars = get_savepath(
        "posterior_errors/errorbars.pdf",
        config,
        output_root,
        args.save,
    )
    filename_errorbars_ratios = get_savepath(
        "posterior_errors/errorbars_ratios.pdf",
        config,
        output_root,
        args.save,
    )
    filename_error_and_hld_width = get_savepath(
        "posterior_errors/error_and_hpd_width.pdf",
        config,
        output_root,
        args.save,
    )
    filename_error_and_hld_width_ratios = get_savepath(
        "posterior_errors/error_and_hpd_width_ratios.pdf",
        config,
        output_root,
        args.save,
    )
    filename_posteriors_data = get_savepath(
        "posterior_errors/results_data.pkl",
        config,
        output_root,
        args.save,
    )
    filename_scatter_data = get_savepath(
        "posterior_errors/first_column_scatter_data.pkl",
        config,
        output_root,
        args.save,
    )
    filename_run_info = get_savepath(
        "posterior_errors/run_info.json",
        config,
        output_root,
        args.save,
    )
    filename_analysis_bundle = get_savepath(
        "posterior_errors/analysis_bundle.json",
        config,
        output_root,
        args.save,
    )

    inference_data = create_inference_data(
        parameters_min_max,
        config,
        generator,
        n_parameters_to_infer_per_dim=args.n_parameters_to_infer_per_dim,
        margin=args.inference_margin,
        n_repititions_per_parameter=args.n_repetitions_per_parameter,
        support_axes=alignment_support_axes,
    )
    posterior_storage_dir = None
    if filename_posteriors_data is not None:
        posterior_storage_dir = filename_posteriors_data.parent / "compact_store"
    all_posteriors, all_ratios, all_HPDs_posterior, all_HPDs_ratio = get_posteriors_and_errors(
        inference_data,
        all_parameters_in_range,
        models=models,
        priors=priors,
        device=device,
        config=config,
        n_qmc_samples=args.posterior_qmc_samples,
        eval_batch_size=args.posterior_eval_batch_size,
        max_model_evals_per_prior=args.posterior_max_model_evals_per_prior,
        qmc_seed=args.posterior_qmc_seed,
        max_gpus=args.max_gpus,
        show_progress=True,
        storage_dir=posterior_storage_dir,
    )

    if filename_posteriors_data is not None:
        with open(filename_posteriors_data, "wb") as handle:
            pickle.dump(
                {
                    "all_HPDs_posterior": all_HPDs_posterior,
                    "all_HPDs_ratio": all_HPDs_ratio,
                    "inference_data": inference_data,
                    "all_posteriors": all_posteriors,
                    "all_ratios": all_ratios,
                    "all_parameters_in_range": all_parameters_in_range,
                    "n_repititions_per_parameter": args.n_repetitions_per_parameter,
                    "n_dimensions": args.n_parameters,
                    "posterior_grid_points": n_parameter_grid_points,
                },
                handle,
            )
        print(f"Saved posterior data to {filename_posteriors_data}")

    fig, _ = plot_errorbars(
        all_HPDs_posterior,
        inference_data,
        all_posteriors,
        all_parameters_in_range,
        show_annotations=False,
        filename=filename_errorbars,
    )
    close_figure(fig, plt)
    fig, _ = plot_errorbars(
        all_HPDs_ratio,
        inference_data,
        all_ratios,
        all_parameters_in_range,
        show_annotations=False,
        filename=filename_errorbars_ratios,
    )
    close_figure(fig, plt)

    fig, _ = plot_error_and_hld_width(
        all_HPDs_posterior,
        inference_data,
        all_posteriors,
        all_parameters_in_range,
        filename=filename_error_and_hld_width,
    )
    if filename_error_and_hld_width is not None:
        _enforce_markers(fig, filename=filename_error_and_hld_width)
    close_figure(fig, plt)

    fig, _ = plot_error_and_hld_width(
        all_HPDs_ratio,
        inference_data,
        all_ratios,
        all_parameters_in_range,
        filename=filename_error_and_hld_width_ratios,
    )
    if filename_error_and_hld_width_ratios is not None:
        _enforce_markers(fig, filename=filename_error_and_hld_width_ratios)
    close_figure(fig, plt)

    averaged_bias_error = get_first_column_scatter_data(
        inference_data,
        all_ratios,
        all_parameters_in_range,
        all_HPDs_ratio,
    )
    if filename_scatter_data is not None:
        with open(filename_scatter_data, "wb") as handle:
            pickle.dump(averaged_bias_error, handle)
        print(f"Saved averaged bias data to {filename_scatter_data}")

    n_ratio_samples = args.n_ratio_samples
    verification_data_by_prior = {}
    for prior_sampler in prior_samplers:
        verification_parameters = prior_sampler((n_ratio_samples,))
        verification_data_by_prior[prior_sampler.__name__] = draw_data(
            verification_parameters,
            config,
            generator,
        )
    print(
        "Check I uses prior-specific marginal samples for E[r(x|theta)] verification:",
        {
            prior_name: tuple(values.shape)
            for prior_name, values in verification_data_by_prior.items()
        },
    )
    test_parameters = create_inference_parameters(
        args.n_ratio_test_parameters,
        parameters_min_max,
        generator,
        margin=0.0,
        data_is_random=False,
        support_axis=None if alignment_support_axes is None else alignment_support_axes[0],
    )
    all_ratio_checks = calculate_all_ratios(
        verification_data_by_prior,
        test_parameters,
        models=models,
        priors=priors,
        device=device,
        config=config,
        max_gpus=args.max_gpus,
    )

    filename_ratio_violins = get_savepath(
        "verifications/ratio_violins.pdf",
        config,
        output_root,
        args.save,
    )
    fig, _ = plot_ratio_violins(all_ratio_checks, test_parameters, filename=filename_ratio_violins)
    close_figure(fig, plt)

    reweight_theta_0 = float(args.reweighting_theta_0)
    reweight_theta_1 = float(args.reweighting_theta_1)
    if alignment_support_axes is not None:
        snapped_theta_0 = snap_to_support_axis(reweight_theta_0, alignment_support_axes[0])
        snapped_theta_1 = snap_to_support_axis(reweight_theta_1, alignment_support_axes[0])
        if not np.isclose(snapped_theta_0, reweight_theta_0):
            print(f"Snapped reweighting_theta_0 from {reweight_theta_0} to {snapped_theta_0}.")
        if not np.isclose(snapped_theta_1, reweight_theta_1):
            print(f"Snapped reweighting_theta_1 from {reweight_theta_1} to {snapped_theta_1}.")
        reweight_theta_0 = float(snapped_theta_0)
        reweight_theta_1 = float(snapped_theta_1)

    test_parameter_sets = [(reweight_theta_0, reweight_theta_1)]
    for test_parameter_set in test_parameter_sets:
        data_x, reweighted_distributions, reweighting_distributions, _ = reweight_distributions(
            test_parameter_set,
            models,
            priors=priors,
            config=config,
            generator=generator,
            device=device,
            n_test_data=args.n_test_data,
            n_bins=args.n_bins,
            projection_dim=args.reweighting_projection_dim,
            max_gpus=args.max_gpus,
        )
        filename_reweighted_distributions = get_savepath(
            (
                "verifications/"
                "reweighting_0/"
                f"reweighted_distributions_{test_parameter_set[0]}_{test_parameter_set[1]}.pdf"
            ),
            config,
            output_root,
            args.save,
        )
        fig, _ = plot_reweighted_distributions(
            priors,
            data_x,
            reweighted_distributions,
            test_parameter_set,
            args.n_bins,
            filename=filename_reweighted_distributions,
            reweighting_distributions=reweighting_distributions,
        )
        close_figure(fig, plt)

    quantitative_bundle = build_quantitative_verification_bundle(
        config=config,
        generator=generator,
        pair_count=args.verification_pair_count,
        model_samples_per_endpoint=args.verification_model_samples_per_endpoint,
        reweight_source_samples=args.verification_reweight_source_samples,
        reweight_target_samples=args.verification_reweight_target_samples,
        swd_projections=args.verification_swd_projections,
        support_axes=alignment_support_axes,
    )

    print("Check III: exact pairwise log-ratio error")
    model_quality_results = evaluate_pairwise_log_ratio_quality(
        quantitative_bundle,
        models=models,
        priors=priors,
        device=device,
        config=config,
        max_gpus=args.max_gpus,
    )
    for prior_name, result in model_quality_results.items():
        print(f"  {prior_name}: {json.dumps(_json_safe_value(result['summary']), sort_keys=True)}")

    filename_log_ratio_error_vs_distance = get_savepath(
        "verifications/log_ratio_error_vs_distance.pdf",
        config,
        output_root,
        args.save,
    )
    fig, _ = plot_log_ratio_error_vs_distance(
        model_quality_results,
        filename=filename_log_ratio_error_vs_distance,
    )
    close_figure(fig, plt)

    filename_log_ratio_exact_vs_predicted = get_savepath(
        "verifications/log_ratio_exact_vs_predicted.pdf",
        config,
        output_root,
        args.save,
    )
    fig, _ = plot_log_ratio_exact_vs_predicted(
        model_quality_results,
        filename=filename_log_ratio_exact_vs_predicted,
    )
    close_figure(fig, plt)

    save_json(
        _verification_results_to_json(model_quality_results),
        get_savepath(
            "verifications/model_quality_summary.json",
            config,
            output_root,
            args.save,
        ),
    )

    print("Check IV: nD reweighting quality")
    reweighting_quality_results = evaluate_nd_reweighting_quality(
        quantitative_bundle,
        models=models,
        priors=priors,
        device=device,
        config=config,
        max_gpus=args.max_gpus,
    )
    for prior_name, result in reweighting_quality_results.items():
        print(f"  {prior_name}: {json.dumps(_json_safe_value(result['summary']), sort_keys=True)}")

    filename_reweighting_swd_vs_distance = get_savepath(
        "verifications/reweighting_swd_vs_distance.pdf",
        config,
        output_root,
        args.save,
    )
    fig, _ = plot_reweighting_swd_vs_distance(
        reweighting_quality_results,
        filename=filename_reweighting_swd_vs_distance,
    )
    close_figure(fig, plt)

    filename_reweighting_summary = get_savepath(
        "verifications/reweighting_summary.pdf",
        config,
        output_root,
        args.save,
    )
    fig, _ = plot_reweighting_summary(
        reweighting_quality_results,
        filename=filename_reweighting_summary,
    )
    close_figure(fig, plt)

    save_json(
        _verification_results_to_json(reweighting_quality_results),
        get_savepath(
            "verifications/reweighting_quality_summary.json",
            config,
            output_root,
            args.save,
        ),
    )

    run_info_payload = {
        "device": str(device),
        "seed": args.seed,
        "output_root": output_root,
        "n_parameters": args.n_parameters,
        "posterior_grid_points": n_parameter_grid_points,
        "n_parameters_to_infer_per_dim": args.n_parameters_to_infer_per_dim,
        "n_repetitions_per_parameter": args.n_repetitions_per_parameter,
        "verification_pair_count": args.verification_pair_count,
        "verification_model_samples_per_endpoint": args.verification_model_samples_per_endpoint,
        "verification_reweight_source_samples": args.verification_reweight_source_samples,
        "verification_reweight_target_samples": args.verification_reweight_target_samples,
        "verification_swd_projections": args.verification_swd_projections,
        "max_gpus": args.max_gpus,
    }
    save_json(run_info_payload, filename_run_info)

    if filename_analysis_bundle is not None:
        analysis_bundle = _build_analysis_bundle(
            args=args,
            config=config,
            run_info_payload=run_info_payload,
            inference_data=inference_data,
            priors=priors,
            all_parameters_in_range=all_parameters_in_range,
            all_posteriors=all_posteriors,
            all_ratios=all_ratios,
            all_hpds_posterior=all_HPDs_posterior,
            all_hpds_ratio=all_HPDs_ratio,
            averaged_bias_error=averaged_bias_error,
            posterior_storage_dir=posterior_storage_dir,
            filename_analysis_bundle=filename_analysis_bundle,
            filename_posteriors_data=filename_posteriors_data,
            filename_scatter_data=filename_scatter_data,
            filename_run_info=filename_run_info,
            extract_hpd_level=_extract_hpd_level,
        )
        save_json(_json_safe_value(analysis_bundle), filename_analysis_bundle)
        print(f"Saved analysis bundle to {filename_analysis_bundle}")

    print("Batch run completed successfully.")
    if filename_run_info is not None:
        print(f"Run metadata saved to {filename_run_info}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the toy_example_nD_tidy notebook logic as a batch-safe Python script. "
            "The positional argument sets the number of parameter dimensions."
        )
    )
    parser.add_argument("n_parameters", type=int, help="Number of parameter dimensions.")
    parser.add_argument("--output-root", default="figs", help="Root directory for saved outputs.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--std-dev", type=float, default=2.0)
    parser.add_argument("--n-train", type=int, default=70_000)
    parser.add_argument("--n-test", type=int, default=15_000)
    parser.add_argument("--n-validation", type=int, default=15_000)
    parser.add_argument("--n-hidden-layers", type=int, default=3)
    parser.add_argument("--n-units", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--model-iterations", type=int, default=1)
    parser.add_argument("--verbose-training", action="store_true")
    parser.add_argument("--prior-histogram-samples", type=int, default=200_000)
    parser.add_argument("--prior-histogram-bins", type=int, default=100)
    parser.add_argument("--vegas-nitn", type=int, default=20)
    parser.add_argument("--vegas-neval", type=int, default=10_000)
    parser.add_argument("--posterior-grid-points", type=int, default=200)
    parser.add_argument("--n-parameters-to-infer-per-dim", type=int, default=25)
    parser.add_argument(
        "--inference-margin",
        type=float,
        default=0.1,
        help="Fraction of the full parameter range trimmed from each boundary when choosing inference points.",
    )
    parser.add_argument(
        "--n-repetitions-per-parameter",
        type=int,
        default=15,
        help="Number of repeated observations drawn for each inferred parameter setting.",
    )
    parser.add_argument("--posterior-qmc-samples", type=int, default=16384)
    parser.add_argument("--posterior-eval-batch-size", type=int, default=2**21)
    parser.add_argument("--posterior-max-model-evals-per-prior", type=int, default=int(2e12))
    parser.add_argument("--posterior-qmc-seed", type=int, default=2026)
    parser.add_argument("--n-ratio-samples", type=int, default=1_000)
    parser.add_argument("--n-ratio-test-parameters", type=int, default=5)
    parser.add_argument("--n-test-data", type=int, default=int(1e6))
    parser.add_argument("--n-bins", type=int, default=int(1e3))
    parser.add_argument("--reweighting-theta-0", type=float, default=3.0)
    parser.add_argument("--reweighting-theta-1", type=float, default=7.0)
    parser.add_argument("--reweighting-projection-dim", type=int, default=0)
    parser.add_argument("--verification-pair-count", type=int, default=256)
    parser.add_argument("--verification-model-samples-per-endpoint", type=int, default=512)
    parser.add_argument("--verification-reweight-source-samples", type=int, default=2048)
    parser.add_argument("--verification-reweight-target-samples", type=int, default=2048)
    parser.add_argument("--verification-swd-projections", type=int, default=32)
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.set_defaults(save=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print_argument_summary(parser, args, sys.argv[1:])
    run(args)


if __name__ == "__main__":
    main()
