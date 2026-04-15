from __future__ import annotations

import json
import subprocess
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.data import draw_data
from src.models import BinaryClassifier
from src.plotting import (
    plot_error_and_hpd_width,
    plot_errorbars,
    plot_log_ratio_error_vs_distance,
    plot_log_ratio_exact_vs_predicted,
    plot_prior_contours,
    plot_reweighted_distributions,
    plot_reweighting_summary,
    plot_reweighting_swd_vs_distance,
)
from src.posterior import (
    _draw_prior_theta_samples,
    _extract_hpd_level,
    _extract_map_array,
    _normalize_log_weights,
    _weighted_hpd_from_samples,
    create_inference_data,
    get_first_column_scatter_data,
    get_posteriors_and_errors,
)
from src.priors import build_priors, get_alignment_support_axes
from src.verification import (
    build_quantitative_verification_bundle,
    calculate_all_ratios,
    compute_ratio,
    compute_effective_sample_size,
    evaluate_nd_reweighting_quality,
    evaluate_pairwise_log_ratio_quality,
    exact_log_r10_gaussian,
    weighted_sliced_wasserstein_distance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "toy_example_nD_tidy.py"
SWEEP_SCRIPT_PATH = REPO_ROOT / "submit_toy_example_nd_sweep.sh"


def _small_config(n_parameters):
    return {
        "data": {
            "prior_args": {
                "uniform": None,
                "normal": (5, 4),
                "exponential": (0.1,),
                "grid": (0.2,),
            },
            "parameter_range": (0, 10),
            "std_dev": 2.0,
            "n_parameters": n_parameters,
            "n_train": 48,
            "n_test": 12,
            "n_validation": 12,
        },
        "classifier": {
            "n_inputs": n_parameters,
            "n_hidden_layers": 1,
            "n_units": 8,
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 16,
            "n_epochs": 1,
            "model_iterations": 1,
        },
    }


def _reference_posteriors(
    inference_data,
    all_parameters_in_range,
    model,
    prior,
    device,
    *,
    n_qmc_samples,
    eval_batch_size,
    max_model_evals_per_prior,
    qmc_seed,
):
    _, _, sampled_data = inference_data
    n_posterior_combinations, n_repititions_per_parameter, _ = sampled_data.shape
    safe_n_qmc = max(
        1024,
        min(
            int(n_qmc_samples),
            int(max_model_evals_per_prior) // max(n_posterior_combinations * n_repititions_per_parameter, 1),
        ),
    )
    theta_samples = _draw_prior_theta_samples(
        prior,
        all_parameters_in_range,
        n_samples=safe_n_qmc,
        seed=qmc_seed,
        scramble=True,
        dtype=torch.float32,
    )
    n_theta = int(theta_samples.shape[0])
    prior_vals = np.asarray(prior(theta_samples), dtype=np.float64)
    prior_vals = np.clip(prior_vals, 1e-300, None)
    log_prior = np.log(prior_vals)

    posteriors_grouped, ratios_grouped = [], []
    errors_posterior, errors_ratio = [], []

    model = model.to(device).eval()
    for combo_idx in range(n_posterior_combinations):
        sampled_block = sampled_data[combo_idx].astype(np.float32)
        log_ratio_sum = np.zeros(n_theta, dtype=np.float64)

        for data_point in sampled_block:
            log_ratio = np.empty(n_theta, dtype=np.float64)
            for start in range(0, n_theta, eval_batch_size):
                end = min(start + eval_batch_size, n_theta)
                theta_chunk = theta_samples[start:end]
                data_chunk = np.broadcast_to(data_point, (end - start, data_point.shape[0])).astype(np.float32)
                inputs = np.concatenate([data_chunk, theta_chunk], axis=1)
                inputs_t = torch.as_tensor(inputs, dtype=torch.float32, device=device)
                outputs = model(inputs_t).detach().cpu().numpy().reshape(-1)
                outputs = np.clip(outputs, 1e-9, 1 - 1e-9)
                log_ratio[start:end] = np.log(outputs) - np.log1p(-outputs)

            log_ratio_sum += log_ratio

        log_post_sum = log_ratio_sum + log_prior

        post_w = _normalize_log_weights(log_post_sum)
        ratio_w = _normalize_log_weights(log_ratio_sum)

        posteriors_grouped.append({"map": theta_samples[int(np.argmax(log_post_sum))].astype(float)})
        ratios_grouped.append({"map": theta_samples[int(np.argmax(log_ratio_sum))].astype(float)})
        errors_posterior.append(
            {
                "68": _weighted_hpd_from_samples(theta_samples, post_w, alpha=0.32),
                "95": _weighted_hpd_from_samples(theta_samples, post_w, alpha=0.05),
            }
        )
        errors_ratio.append(
            {
                "68": _weighted_hpd_from_samples(theta_samples, ratio_w, alpha=0.32),
                "95": _weighted_hpd_from_samples(theta_samples, ratio_w, alpha=0.05),
            }
        )

    model.to("cpu")
    return posteriors_grouped, ratios_grouped, errors_posterior, errors_ratio


class SamplingTests(unittest.TestCase):
    def test_draw_data_vectorized_preserves_shape_and_statistics(self):
        config = _small_config(3)
        generator = np.random.default_rng(1234)
        args = np.zeros((20000, 3), dtype=np.float32)
        samples = draw_data(args, config, generator)

        self.assertEqual(samples.shape, args.shape)
        self.assertEqual(samples.dtype, np.float32)
        self.assertLess(np.max(np.abs(samples.mean(axis=0))), 0.08)
        self.assertTrue(np.allclose(samples.var(axis=0), 4.0, atol=0.2))

    def test_build_priors_includes_grid_sampler_and_metadata(self):
        config = _small_config(2)
        generator = np.random.default_rng(42)
        priors, prior_samplers, _, _ = build_priors(config, generator)

        prior_names = [prior.__name__ for prior in priors]
        self.assertEqual(prior_names, ["uniform", "normal", "exponential", "grid"])
        self.assertEqual([sampler.__name__ for sampler in prior_samplers], prior_names)

        grid_prior = priors[-1]
        grid_sampler = prior_samplers[-1]
        self.assertEqual(grid_prior.support_kind, "discrete_grid")
        self.assertEqual(grid_prior.normalization_mode, "discrete_mass")
        self.assertAlmostEqual(grid_prior.discrete_mass, 1.0)
        self.assertIsNotNone(grid_prior.support_axes)

        samples = grid_sampler((256,))
        self.assertEqual(samples.shape, (256, 2))
        for dim, axis in enumerate(grid_prior.support_axes):
            self.assertTrue(np.all(np.isin(samples[:, dim], axis)))

        prior_values = np.asarray(grid_prior(samples), dtype=float)
        self.assertTrue(np.all(prior_values > 0.0))
        self.assertTrue(
            np.allclose(
                prior_values,
                np.full(samples.shape[0], 1.0 / grid_prior.support_size, dtype=float),
            )
        )

        off_grid = samples.copy()
        off_grid[0, 0] += 0.07
        self.assertEqual(float(grid_prior(off_grid[0])), 0.0)

    def test_truncated_continuous_priors_stay_in_box_and_normalize_in_1d(self):
        config = _small_config(1)
        generator = np.random.default_rng(123)
        priors, prior_samplers, param_min, param_max = build_priors(config, generator)
        prior_lookup = {prior.__name__: prior for prior in priors}
        sampler_lookup = {sampler.__name__: sampler for sampler in prior_samplers}

        xs = np.linspace(param_min, param_max, 4001, dtype=np.float64)
        x_eval = xs[:, None]

        for prior_name in ("normal", "exponential"):
            samples = sampler_lookup[prior_name]((50_000,))
            self.assertEqual(samples.shape, (50_000, 1))
            self.assertTrue(np.all(samples >= param_min))
            self.assertTrue(np.all(samples <= param_max))

            prior = prior_lookup[prior_name]
            integral = np.trapezoid(np.asarray(prior(x_eval), dtype=np.float64), xs)
            self.assertAlmostEqual(float(integral), 1.0, places=3)
            self.assertEqual(float(prior(np.array([param_min - 1.0], dtype=np.float64))), 0.0)
            self.assertEqual(float(prior(np.array([param_max + 1.0], dtype=np.float64))), 0.0)


class PosteriorTests(unittest.TestCase):
    def test_grid_prior_theta_support_stays_on_lattice(self):
        config = _small_config(3)
        generator = np.random.default_rng(9)
        priors, _, _, _ = build_priors(config, generator)
        grid_prior = next(prior for prior in priors if prior.__name__ == "grid")
        all_parameters_in_range = [np.linspace(0, 10, 11) for _ in range(config["data"]["n_parameters"])]

        theta_samples = _draw_prior_theta_samples(
            grid_prior,
            all_parameters_in_range,
            n_samples=2048,
            seed=17,
            scramble=True,
            dtype=torch.float32,
        )
        self.assertGreater(theta_samples.shape[0], 0)
        for dim, axis in enumerate(grid_prior.support_axes):
            self.assertTrue(np.all(np.isin(theta_samples[:, dim], axis)))
        prior_vals = np.asarray(grid_prior(theta_samples), dtype=float)
        self.assertTrue(np.all(prior_vals > 0.0))

    def test_batched_posterior_matches_reference_on_tiny_case(self):
        config = _small_config(2)
        generator = np.random.default_rng(7)
        priors, prior_samplers, _, _ = build_priors(config, generator)
        support_axes = get_alignment_support_axes(priors)

        torch.manual_seed(5)
        models = {}
        for prior_sampler in prior_samplers:
            model = BinaryClassifier(config)
            model.eval()
            models[prior_sampler.__name__] = model

        inference_data = create_inference_data(
            config["data"]["parameter_range"],
            config,
            generator,
            n_parameters_to_infer_per_dim=2,
            n_repititions_per_parameter=2,
            support_axes=support_axes,
        )
        all_parameters_in_range = [np.linspace(0, 10, 11) for _ in range(config["data"]["n_parameters"])]

        actual = get_posteriors_and_errors(
            inference_data,
            all_parameters_in_range,
            models=models,
            priors=priors,
            device=torch.device("cpu"),
            config=config,
            n_qmc_samples=1024,
            eval_batch_size=512,
            max_model_evals_per_prior=int(1e9),
            qmc_seed=123,
            max_gpus=1,
            show_progress=False,
        )

        expected_posteriors = {}
        expected_ratios = {}
        expected_hpds_posterior = {}
        expected_hpds_ratio = {}
        for prior in priors:
            posteriors, ratios, hpds_post, hpds_ratio = _reference_posteriors(
                inference_data,
                all_parameters_in_range,
                models[prior.__name__],
                prior,
                torch.device("cpu"),
                n_qmc_samples=1024,
                eval_batch_size=512,
                max_model_evals_per_prior=int(1e9),
                qmc_seed=123,
            )
            expected_posteriors[prior.__name__] = posteriors
            expected_ratios[prior.__name__] = ratios
            expected_hpds_posterior[prior.__name__] = hpds_post
            expected_hpds_ratio[prior.__name__] = hpds_ratio

        actual_posteriors, actual_ratios, actual_hpds_posterior, actual_hpds_ratio = actual

        for prior_name in expected_posteriors:
            actual_post_maps = _extract_map_array(
                actual_posteriors[prior_name],
                parameters_post=all_parameters_in_range,
                n_dims=config["data"]["n_parameters"],
            )
            expected_post_maps = _extract_map_array(
                expected_posteriors[prior_name],
                parameters_post=all_parameters_in_range,
                n_dims=config["data"]["n_parameters"],
            )
            np.testing.assert_allclose(actual_post_maps, expected_post_maps, atol=1e-6)

            actual_ratio_maps = _extract_map_array(
                actual_ratios[prior_name],
                parameters_post=all_parameters_in_range,
                n_dims=config["data"]["n_parameters"],
            )
            expected_ratio_maps = _extract_map_array(
                expected_ratios[prior_name],
                parameters_post=all_parameters_in_range,
                n_dims=config["data"]["n_parameters"],
            )
            np.testing.assert_allclose(actual_ratio_maps, expected_ratio_maps, atol=1e-6)

            if prior_name == "grid":
                np.testing.assert_allclose(actual_post_maps, actual_ratio_maps, atol=1e-6)

            for level in ("68", "95"):
                actual_intervals, actual_combined = _extract_hpd_level(
                    actual_hpds_posterior[prior_name],
                    level,
                    config["data"]["n_parameters"],
                )
                expected_intervals, expected_combined = _extract_hpd_level(
                    expected_hpds_posterior[prior_name],
                    level,
                    config["data"]["n_parameters"],
                )
                np.testing.assert_allclose(actual_intervals, expected_intervals, atol=1e-6)
                np.testing.assert_allclose(actual_combined, expected_combined, atol=1e-6)

                if prior_name == "grid":
                    np.testing.assert_allclose(
                        actual_intervals,
                        _extract_hpd_level(
                            actual_hpds_ratio[prior_name],
                            level,
                            config["data"]["n_parameters"],
                        )[0],
                        atol=1e-6,
                    )

                actual_intervals, actual_combined = _extract_hpd_level(
                    actual_hpds_ratio[prior_name],
                    level,
                    config["data"]["n_parameters"],
                )
                expected_intervals, expected_combined = _extract_hpd_level(
                    expected_hpds_ratio[prior_name],
                    level,
                    config["data"]["n_parameters"],
                )
                np.testing.assert_allclose(actual_intervals, expected_intervals, atol=1e-6)
                np.testing.assert_allclose(actual_combined, expected_combined, atol=1e-6)


class PlottingTests(unittest.TestCase):
    def test_plot_prior_contours_renders_discrete_grid_prior(self):
        config = _small_config(2)
        generator = np.random.default_rng(11)
        priors, _, _, _ = build_priors(config, generator)

        fig, axes = plot_prior_contours(
            priors,
            parameter_range=config["data"]["parameter_range"],
            n_parameters=config["data"]["n_parameters"],
            n_points=25,
        )
        titles = [axis.get_title() for axis in axes]
        self.assertIn("grid", titles)
        plt.close(fig)

    def test_nd_plotting_accepts_tuple_and_dict_hpd_entries(self):
        n_dims = 3
        true_params = np.asarray(
            [
                [2.0, 3.0, 4.0],
                [4.0, 5.0, 6.0],
                [6.0, 7.0, 8.0],
            ],
            dtype=float,
        )
        inference_data = (None, true_params, None)
        parameters_post = [np.linspace(0, 10, 5) for _ in range(n_dims)]
        all_posteriors = {
            "uniform": [
                {"map": np.array([2.5, 3.5, 4.5])},
                {"map": np.array([4.5, 5.5, 6.5])},
                {"map": np.array([6.5, 7.5, 8.5])},
            ]
        }

        tuple_hpds = {
            "uniform": [
                {
                    "68": (np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]), np.array([3.0, 4.0])),
                    "95": (np.array([[1.5, 3.5], [2.5, 4.5], [3.5, 5.5]]), np.array([2.5, 4.5])),
                },
                {
                    "68": (np.array([[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]]), np.array([5.0, 6.0])),
                    "95": (np.array([[3.5, 5.5], [4.5, 6.5], [5.5, 7.5]]), np.array([4.5, 6.5])),
                },
                {
                    "68": (np.array([[6.0, 7.0], [7.0, 8.0], [8.0, 9.0]]), np.array([7.0, 8.0])),
                    "95": (np.array([[5.5, 7.5], [6.5, 8.5], [7.5, 9.5]]), np.array([6.5, 8.5])),
                },
            ]
        }
        dict_hpds = {
            "uniform": [
                {
                    "68": {
                        "intervals": np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
                        "interval_combined": np.array([3.0, 4.0]),
                    },
                    "95": {
                        "intervals": np.array([[1.5, 3.5], [2.5, 4.5], [3.5, 5.5]]),
                        "interval_combined": np.array([2.5, 4.5]),
                    },
                },
                {
                    "68": {
                        "intervals": np.array([[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]]),
                        "interval_combined": np.array([5.0, 6.0]),
                    },
                    "95": {
                        "intervals": np.array([[3.5, 5.5], [4.5, 6.5], [5.5, 7.5]]),
                        "interval_combined": np.array([4.5, 6.5]),
                    },
                },
                {
                    "68": {
                        "intervals": np.array([[6.0, 7.0], [7.0, 8.0], [8.0, 9.0]]),
                        "interval_combined": np.array([7.0, 8.0]),
                    },
                    "95": {
                        "intervals": np.array([[5.5, 7.5], [6.5, 8.5], [7.5, 9.5]]),
                        "interval_combined": np.array([6.5, 8.5]),
                    },
                },
            ]
        }
        compact_posteriors = {
            "uniform": {
                "format": "compact_map_store",
                "n_points": 3,
                "n_dims": n_dims,
                "dtype": "float32",
                "map": np.array(
                    [
                        [2.5, 3.5, 4.5],
                        [4.5, 5.5, 6.5],
                        [6.5, 7.5, 8.5],
                    ],
                    dtype=np.float32,
                ),
            }
        }
        compact_hpds = {
            "uniform": {
                "format": "compact_hpd_store",
                "n_points": 3,
                "n_dims": n_dims,
                "dtype": "float32",
                "68": {
                    "intervals": np.array(
                        [
                            [[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                            [[6.0, 7.0], [7.0, 8.0], [8.0, 9.0]],
                        ],
                        dtype=np.float32,
                    ),
                    "interval_combined": np.array(
                        [
                            [3.0, 4.0],
                            [5.0, 6.0],
                            [7.0, 8.0],
                        ],
                        dtype=np.float32,
                    ),
                },
                "95": {
                    "intervals": np.array(
                        [
                            [[1.5, 3.5], [2.5, 4.5], [3.5, 5.5]],
                            [[3.5, 5.5], [4.5, 6.5], [5.5, 7.5]],
                            [[5.5, 7.5], [6.5, 8.5], [7.5, 9.5]],
                        ],
                        dtype=np.float32,
                    ),
                    "interval_combined": np.array(
                        [
                            [2.5, 4.5],
                            [4.5, 6.5],
                            [6.5, 8.5],
                        ],
                        dtype=np.float32,
                    ),
                },
            }
        }

        for hpds in (tuple_hpds, dict_hpds):
            fig, _ = plot_errorbars(hpds, inference_data, all_posteriors, parameters_post, show_annotations=False)
            plt.close(fig)
            fig, _ = plot_error_and_hpd_width(hpds, inference_data, all_posteriors, parameters_post)
            plt.close(fig)
            scatter = get_first_column_scatter_data(inference_data, all_posteriors, parameters_post, hpds)
            self.assertIn("uniform", scatter)
            self.assertEqual(scatter["uniform"]["avg_bias"].shape[0], scatter["uniform"]["x"].shape[0])

        fig, _ = plot_errorbars(compact_hpds, inference_data, compact_posteriors, parameters_post, show_annotations=False)
        plt.close(fig)
        fig, _ = plot_error_and_hpd_width(compact_hpds, inference_data, compact_posteriors, parameters_post)
        plt.close(fig)
        scatter = get_first_column_scatter_data(inference_data, compact_posteriors, parameters_post, compact_hpds)
        self.assertIn("uniform", scatter)
        self.assertEqual(scatter["uniform"]["avg_bias"].shape[0], scatter["uniform"]["x"].shape[0])

    def test_reweighted_distribution_plot_includes_n_and_ess(self):
        def _prior(name):
            def prior(_):
                return np.ones(1)

            prior.__name__ = name
            return prior

        priors = [_prior("uniform"), _prior("normal")]
        data_x = [
            np.array([0.0, 0.5, 1.0, 1.5], dtype=float),
            np.array([1.0, 1.5, 2.0, 2.5], dtype=float),
        ]
        reweighted_distributions = {
            "uniform": (np.array([0.25, 0.75, 1.25]), np.array([1.0, 2.0, 1.0])),
            "normal": (np.array([0.25, 0.75, 1.25]), np.array([0.5, 1.5, 2.5])),
        }
        reweighting_distributions = {
            "uniform": np.ones(4, dtype=float),
            "normal": np.array([4.0, 0.0, 0.0, 0.0], dtype=float),
        }

        fig, axes = plot_reweighted_distributions(
            priors,
            data_x,
            reweighted_distributions,
            test_parameters=(3, 7),
            n_bins=4,
            reweighting_distributions=reweighting_distributions,
        )
        self.assertEqual(axes[0].get_title(), "uniform prior\nN=4 | ESS=4")
        self.assertEqual(axes[1].get_title(), "normal prior\nN=4 | ESS=1")
        self.assertEqual(axes[0].get_ylabel(), "Density")
        first_hist_area = sum(patch.get_width() * patch.get_height() for patch in axes[0].patches[:4])
        second_hist_area = sum(patch.get_width() * patch.get_height() for patch in axes[0].patches[4:8])
        self.assertAlmostEqual(first_hist_area, 1.0, places=7)
        self.assertAlmostEqual(second_hist_area, 1.0, places=7)
        plt.close(fig)

    def test_quantitative_verification_plots_render(self):
        model_quality_results = {
            "uniform": {
                "pair_metrics": {
                    "distance": np.array([1.0, 2.0, 3.0], dtype=float),
                    "rmse_log_r10": np.array([0.2, 0.4, 0.6], dtype=float),
                },
                "pooled": {
                    "exact_log_r10": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float),
                    "predicted_log_r10": np.array([-1.8, -0.9, 0.1, 0.8, 2.2], dtype=float),
                },
                "summary": {
                    "median_rmse_log_r10": 0.4,
                    "worst_rmse_log_r10": 0.6,
                    "pooled_rmse_log_r10": 0.16,
                    "spearman_rmse_vs_distance": 1.0,
                },
            },
            "normal": {
                "pair_metrics": {
                    "distance": np.array([1.5, 2.5, 3.5], dtype=float),
                    "rmse_log_r10": np.array([0.3, 0.35, 0.5], dtype=float),
                },
                "pooled": {
                    "exact_log_r10": np.array([-1.0, 0.0, 1.0], dtype=float),
                    "predicted_log_r10": np.array([-0.8, 0.05, 0.9], dtype=float),
                },
                "summary": {
                    "median_rmse_log_r10": 0.35,
                    "worst_rmse_log_r10": 0.5,
                    "pooled_rmse_log_r10": 0.14,
                    "spearman_rmse_vs_distance": 0.5,
                },
            },
        }
        reweighting_results = {
            "uniform": {
                "pair_metrics": {
                    "distance": np.array([1.0, 2.0, 3.0], dtype=float),
                    "swd": np.array([0.2, 0.4, 0.8], dtype=float),
                    "ess_fraction": np.array([0.9, 0.6, 0.2], dtype=float),
                },
                "summary": {
                    "median_swd": 0.4,
                    "worst_swd": 0.8,
                    "median_ess_fraction": 0.6,
                    "worst_ess_fraction": 0.2,
                    "spearman_swd_vs_distance": 1.0,
                },
            },
            "normal": {
                "pair_metrics": {
                    "distance": np.array([1.2, 2.2, 3.2], dtype=float),
                    "swd": np.array([0.1, 0.3, 0.5], dtype=float),
                    "ess_fraction": np.array([0.8, 0.5, 0.3], dtype=float),
                },
                "summary": {
                    "median_swd": 0.3,
                    "worst_swd": 0.5,
                    "median_ess_fraction": 0.5,
                    "worst_ess_fraction": 0.3,
                    "spearman_swd_vs_distance": 1.0,
                },
            },
        }

        fig, axes = plot_log_ratio_error_vs_distance(model_quality_results, n_bins=2)
        self.assertIn("uniform prior", axes[0].get_title())
        self.assertEqual(axes[0].get_legend().get_texts()[0].get_text(), "parameter pair")
        plt.close(fig)

        fig, axes = plot_log_ratio_exact_vs_predicted(model_quality_results)
        self.assertIn("pooled RMSE", axes[0].get_title())
        plt.close(fig)

        fig, axes = plot_reweighting_swd_vs_distance(reweighting_results, n_bins=2)
        self.assertEqual(axes[0].get_ylabel(), r"$\mathrm{SW}_1$")
        plt.close(fig)

        fig, axes = plot_reweighting_summary(reweighting_results)
        self.assertEqual(axes[1].get_ylabel(), r"$\mathrm{ESS}/N$")
        plt.close(fig)

    def test_verification_prior_grids_use_two_columns_for_four_priors(self):
        model_quality_results = {}
        for prior_name, offset in zip(("uniform", "normal", "exponential", "grid"), (0.0, 0.1, 0.2, 0.3)):
            model_quality_results[prior_name] = {
                "pair_metrics": {
                    "distance": np.array([1.0, 2.0, 3.0], dtype=float),
                    "rmse_log_r10": np.array([0.2, 0.4, 0.6], dtype=float) + offset,
                },
                "pooled": {
                    "exact_log_r10": np.array([-1.0, 0.0, 1.0], dtype=float),
                    "predicted_log_r10": np.array([-0.9, 0.1, 1.1], dtype=float) + offset,
                },
                "summary": {
                    "median_rmse_log_r10": 0.4 + offset,
                    "worst_rmse_log_r10": 0.6 + offset,
                    "pooled_rmse_log_r10": 0.15 + offset,
                    "spearman_rmse_vs_distance": 1.0,
                },
            }

        fig, axes = plot_log_ratio_error_vs_distance(model_quality_results, n_bins=2)
        grid_spec = axes[0].get_subplotspec().get_gridspec()
        self.assertEqual(grid_spec.ncols, 2)
        self.assertEqual(grid_spec.nrows, 2)
        plt.close(fig)

        def _prior(name):
            def prior(_):
                return np.ones(1)

            prior.__name__ = name
            return prior

        priors = [_prior(name) for name in ("uniform", "normal", "exponential", "grid")]
        data_x = [
            np.array([0.0, 0.5, 1.0, 1.5], dtype=float),
            np.array([1.0, 1.5, 2.0, 2.5], dtype=float),
        ]
        reweighted_distributions = {
            name: (np.array([0.25, 0.75, 1.25]), np.array([1.0, 2.0, 1.0]) + idx)
            for idx, name in enumerate(("uniform", "normal", "exponential", "grid"))
        }

        fig, axes = plot_reweighted_distributions(
            priors,
            data_x,
            reweighted_distributions,
            test_parameters=(3, 7),
            n_bins=4,
        )
        grid_spec = axes[0].get_subplotspec().get_gridspec()
        self.assertEqual(grid_spec.ncols, 2)
        self.assertEqual(grid_spec.nrows, 2)
        plt.close(fig)


class VerificationMetricTests(unittest.TestCase):
    def test_calculate_all_ratios_supports_shared_and_prior_specific_inputs(self):
        config = _small_config(1)
        generator = np.random.default_rng(19)
        priors, prior_samplers, _, _ = build_priors(config, generator)

        torch.manual_seed(13)
        models = {}
        for prior_sampler in prior_samplers:
            model = BinaryClassifier(config)
            model.eval()
            models[prior_sampler.__name__] = model

        test_parameters = [2.0, 6.0]
        shared_inputs = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)
        shared_ratios = calculate_all_ratios(
            shared_inputs,
            test_parameters,
            models=models,
            priors=priors,
            device=torch.device("cpu"),
            config=config,
            max_gpus=1,
        )

        for prior in priors:
            expected = []
            for parameter in test_parameters:
                ratio_values, _ = compute_ratio(shared_inputs, parameter, models[prior.__name__], torch.device("cpu"))
                expected.append(ratio_values.tolist())
            self.assertEqual(shared_ratios[prior.__name__], expected)

        prior_specific_inputs = {
            prior.__name__: np.full((3, 1), float(idx + 1), dtype=np.float32)
            for idx, prior in enumerate(priors)
        }
        routed_ratios = calculate_all_ratios(
            prior_specific_inputs,
            test_parameters,
            models=models,
            priors=priors,
            device=torch.device("cpu"),
            config=config,
            max_gpus=1,
        )

        for idx, prior in enumerate(priors):
            expected = []
            expected_inputs = prior_specific_inputs[prior.__name__]
            for parameter in test_parameters:
                ratio_values, _ = compute_ratio(
                    expected_inputs,
                    parameter,
                    models[prior.__name__],
                    torch.device("cpu"),
                )
                expected.append(ratio_values.tolist())
            self.assertEqual(routed_ratios[prior.__name__], expected)

    def test_quantitative_verification_bundle_can_sample_on_grid_support(self):
        config = _small_config(2)
        generator = np.random.default_rng(31)
        priors, _, _, _ = build_priors(config, generator)
        support_axes = get_alignment_support_axes(priors)

        bundle = build_quantitative_verification_bundle(
            config,
            generator,
            pair_count=8,
            model_samples_per_endpoint=4,
            reweight_source_samples=8,
            reweight_target_samples=8,
            swd_projections=4,
            support_axes=support_axes,
        )

        for dim, axis in enumerate(support_axes):
            self.assertTrue(np.all(np.isin(bundle["theta_0"][:, dim], axis)))
            self.assertTrue(np.all(np.isin(bundle["theta_1"][:, dim], axis)))

    def test_exact_log_r10_matches_closed_form_in_1d_and_nd(self):
        sigma = 2.0

        data_1d = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        theta_0_1d = np.array([0.5], dtype=np.float32)
        theta_1_1d = np.array([2.5], dtype=np.float32)
        expected_1d = (
            np.square(data_1d[:, 0] - theta_0_1d[0]) - np.square(data_1d[:, 0] - theta_1_1d[0])
        ) / (2.0 * sigma**2)
        np.testing.assert_allclose(
            exact_log_r10_gaussian(data_1d, theta_0_1d, theta_1_1d, sigma),
            expected_1d,
            atol=1e-7,
        )

        data_nd = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        theta_0_nd = np.array([0.0, 1.0], dtype=np.float32)
        theta_1_nd = np.array([2.0, 5.0], dtype=np.float32)
        expected_nd = (
            np.sum(np.square(data_nd - theta_0_nd), axis=1) - np.sum(np.square(data_nd - theta_1_nd), axis=1)
        ) / (2.0 * sigma**2)
        np.testing.assert_allclose(
            exact_log_r10_gaussian(data_nd, theta_0_nd, theta_1_nd, sigma),
            expected_nd,
            atol=1e-7,
        )

    def test_weighted_sliced_wasserstein_is_zero_for_identical_and_larger_for_shifted(self):
        source = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        target = source.copy()
        shifted_target = target + 1.5
        weights = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        directions = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            ],
            dtype=np.float32,
        )

        identical_distance = weighted_sliced_wasserstein_distance(
            source,
            target,
            source_weights=weights,
            target_weights=weights,
            directions=directions,
        )
        shifted_distance = weighted_sliced_wasserstein_distance(
            source,
            shifted_target,
            source_weights=weights,
            target_weights=weights,
            directions=directions,
        )

        self.assertAlmostEqual(identical_distance, 0.0, places=7)
        self.assertGreater(shifted_distance, identical_distance + 0.5)

    def test_effective_sample_size_matches_balanced_and_concentrated_weights(self):
        balanced_weights = np.full(8, 1.0 / 8.0, dtype=np.float64)
        concentrated_weights = np.array([0.97, 0.01, 0.01, 0.01], dtype=np.float64)

        self.assertAlmostEqual(compute_effective_sample_size(balanced_weights), 8.0)
        self.assertLess(compute_effective_sample_size(concentrated_weights), 1.1)

    def test_quantitative_verification_returns_finite_summaries(self):
        config = _small_config(2)
        generator = np.random.default_rng(23)
        priors, prior_samplers, _, _ = build_priors(config, generator)
        support_axes = get_alignment_support_axes(priors)

        torch.manual_seed(17)
        models = {}
        for prior_sampler in prior_samplers:
            model = BinaryClassifier(config)
            model.eval()
            models[prior_sampler.__name__] = model

        bundle = build_quantitative_verification_bundle(
            config,
            generator,
            pair_count=4,
            model_samples_per_endpoint=8,
            reweight_source_samples=16,
            reweight_target_samples=16,
            swd_projections=4,
            support_axes=support_axes,
        )

        model_quality_results = evaluate_pairwise_log_ratio_quality(
            bundle,
            models,
            priors,
            device=torch.device("cpu"),
            config=config,
            max_gpus=1,
        )
        reweighting_results = evaluate_nd_reweighting_quality(
            bundle,
            models,
            priors,
            device=torch.device("cpu"),
            config=config,
            max_gpus=1,
        )

        for prior in priors:
            prior_name = prior.__name__
            self.assertIn(prior_name, model_quality_results)
            self.assertIn(prior_name, reweighting_results)

            model_summary = model_quality_results[prior_name]["summary"]
            reweight_summary = reweighting_results[prior_name]["summary"]
            self.assertTrue(np.isfinite(model_summary["median_rmse_log_r10"]))
            self.assertTrue(np.isfinite(model_summary["worst_rmse_log_r10"]))
            self.assertTrue(np.isfinite(model_summary["pooled_rmse_log_r10"]))
            self.assertTrue(np.isfinite(reweight_summary["median_swd"]))
            self.assertTrue(np.isfinite(reweight_summary["worst_swd"]))
            self.assertTrue(np.isfinite(reweight_summary["median_ess_fraction"]))
            self.assertTrue(np.isfinite(reweight_summary["worst_ess_fraction"]))

            self.assertEqual(model_quality_results[prior_name]["pair_metrics"]["distance"].shape[0], 4)
            self.assertEqual(reweighting_results[prior_name]["pair_metrics"]["swd"].shape[0], 4)
            self.assertTrue(
                np.all(np.isfinite(model_quality_results[prior_name]["pooled"]["predicted_log_r10"]))
            )
            self.assertTrue(np.all(np.isfinite(reweighting_results[prior_name]["pair_metrics"]["ess_fraction"])))


class SweepLauncherTests(unittest.TestCase):
    def _create_stubbed_sweep_scripts(self, tmp_path: Path):
        sweep_copy = tmp_path / "submit_toy_example_nd_sweep.sh"
        submit_stub = tmp_path / "submit_toy_example_nd.sh"
        calls_file = tmp_path / "submit_calls.txt"

        shutil.copyfile(SWEEP_SCRIPT_PATH, sweep_copy)
        submit_stub.write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            f"printf '%s\\n' \"$*\" >> \"{calls_file}\"\n"
        )
        sweep_copy.chmod(0o755)
        submit_stub.chmod(0o755)
        return sweep_copy, calls_file

    def test_sweep_submits_dimensions_in_descending_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_copy, calls_file = self._create_stubbed_sweep_scripts(Path(tmpdir))
            subprocess.run(
                [
                    "bash",
                    str(sweep_copy),
                    "--device",
                    "cpu",
                    "--max-gpus",
                    "2",
                    "--seed",
                    "17",
                    "--no-save",
                ],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertTrue(calls_file.exists())
            self.assertEqual(
                calls_file.read_text().splitlines(),
                [
                    "7 --device cpu --max-gpus 2 --seed 17 --no-save",
                    "6 --device cpu --max-gpus 2 --seed 17 --no-save",
                    "5 --device cpu --max-gpus 2 --seed 17 --no-save",
                    "4 --device cpu --max-gpus 2 --seed 17 --no-save",
                    "3 --device cpu --max-gpus 2 --seed 17 --no-save",
                    "2 --device cpu --max-gpus 2 --seed 17 --no-save",
                    "1 --device cpu --max-gpus 2 --seed 17 --no-save",
                ],
            )

    def test_sweep_rejects_interactive_and_follow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_copy, _ = self._create_stubbed_sweep_scripts(Path(tmpdir))
            for flag in ("--interactive", "--follow"):
                result = subprocess.run(
                    ["bash", str(sweep_copy), flag],
                    cwd=tmpdir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{flag} is not supported", result.stdout)


class ScriptSmokeTests(unittest.TestCase):
    def _run_script(self, *extra_args):
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            *map(str, extra_args),
        ]
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def _assert_no_nan_values(self, value):
        if isinstance(value, dict):
            for item in value.values():
                self._assert_no_nan_values(item)
            return
        if isinstance(value, list):
            for item in value:
                self._assert_no_nan_values(item)
            return
        if isinstance(value, float):
            self.assertFalse(np.isnan(value))

    def test_cpu_no_save_smoke(self):
        result = self._run_script(
            3,
            "--device",
            "cpu",
            "--max-gpus",
            1,
            "--no-save",
            "--n-train",
            48,
            "--n-test",
            12,
            "--n-validation",
            12,
            "--n-epochs",
            1,
            "--n-hidden-layers",
            1,
            "--n-units",
            8,
            "--prior-histogram-samples",
            96,
            "--prior-histogram-bins",
            8,
            "--posterior-grid-points",
            11,
            "--n-parameters-to-infer-per-dim",
            2,
            "--n-repetitions-per-parameter",
            1,
            "--posterior-qmc-samples",
            1024,
            "--posterior-eval-batch-size",
            512,
            "--n-ratio-samples",
            32,
            "--n-ratio-test-parameters",
            2,
            "--n-test-data",
            128,
            "--n-bins",
            16,
            "--verification-pair-count",
            4,
            "--verification-model-samples-per-endpoint",
            8,
            "--verification-reweight-source-samples",
            16,
            "--verification-reweight-target-samples",
            16,
            "--verification-swd-projections",
            4,
        )
        self.assertIn("Batch run completed successfully.", result.stdout)

    def test_saved_outputs_include_analysis_bundle_and_legacy_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"
            self._run_script(
                2,
                "--device",
                "cpu",
                "--max-gpus",
                1,
                "--output-root",
                output_root,
                "--n-train",
                48,
                "--n-test",
                12,
                "--n-validation",
                12,
                "--n-epochs",
                1,
                "--n-hidden-layers",
                1,
                "--n-units",
                8,
                "--prior-histogram-samples",
                96,
                "--prior-histogram-bins",
                8,
                "--posterior-grid-points",
                11,
                "--n-parameters-to-infer-per-dim",
                2,
                "--n-repetitions-per-parameter",
                1,
                "--posterior-qmc-samples",
                1024,
                "--posterior-eval-batch-size",
                512,
                "--n-ratio-samples",
                32,
                "--n-ratio-test-parameters",
                2,
                "--n-test-data",
                128,
                "--n-bins",
                16,
                "--verification-pair-count",
                4,
                "--verification-model-samples-per-endpoint",
                8,
                "--verification-reweight-source-samples",
                16,
                "--verification-reweight-target-samples",
                16,
                "--verification-swd-projections",
                4,
            )

            analysis_bundle_paths = list(output_root.glob("config_*/posterior_errors/analysis_bundle.json"))
            self.assertEqual(len(analysis_bundle_paths), 1)
            analysis_bundle_path = analysis_bundle_paths[0]
            posterior_errors_dir = analysis_bundle_path.parent
            bundle = json.loads(analysis_bundle_path.read_text())

            self.assertEqual(bundle["schema_version"], 1)
            self.assertEqual(bundle["run_metadata"]["n_parameters"], 2)
            self.assertEqual(bundle["run_metadata"]["max_gpus"], 1)
            self.assertEqual(bundle["posterior_workload"]["n_repetitions_per_setting"], 1)
            self.assertEqual(bundle["posterior_workload"]["n_priors"], 4)
            self.assertIn("grid", bundle["posterior_workload"]["prior_names"])
            self.assertIn("effective_config", bundle["run_metadata"])
            self.assertIn("posterior", bundle["hpd_summary"])
            self.assertIn("ratio", bundle["hpd_summary"])
            self.assertTrue(bundle["posterior_workload"]["compact_posterior_storage_written"])

            artifact_files = bundle["artifacts"]["files"]
            for relative_path in artifact_files.values():
                self.assertIsInstance(relative_path, str)
                self.assertFalse(Path(relative_path).is_absolute())
                self.assertTrue((posterior_errors_dir / relative_path).exists())

            compact_store_dir = bundle["artifacts"]["compact_store_dir"]
            self.assertIsInstance(compact_store_dir, str)
            self.assertFalse(Path(compact_store_dir).is_absolute())
            self.assertTrue((posterior_errors_dir / compact_store_dir).is_dir())

            compact_store_files = bundle["artifacts"]["compact_store_files"]
            self.assertTrue(compact_store_files)
            for relative_paths in compact_store_files.values():
                self.assertTrue(relative_paths)
                for relative_path in relative_paths:
                    self.assertFalse(Path(relative_path).is_absolute())
                    self.assertTrue((posterior_errors_dir / relative_path).exists())

            for prior_bias in bundle["bias_summary"].values():
                self.assertIsInstance(prior_bias["x"], list)
                self.assertIsInstance(prior_bias["avg_bias"], list)
                self.assertIn("68", prior_bias["width_curves"])
                self.assertIn("95", prior_bias["width_curves"])

            self._assert_no_nan_values(bundle)

            self.assertTrue((posterior_errors_dir / "results_data.pkl").exists())
            self.assertTrue((posterior_errors_dir / "first_column_scatter_data.pkl").exists())
            self.assertTrue((posterior_errors_dir / "run_info.json").exists())
            self.assertTrue((posterior_errors_dir / "analysis_bundle.json").exists())

            config_dir = posterior_errors_dir.parent
            self.assertTrue((config_dir / "models/models_0/model_grid_prior.pth").exists())
            self.assertTrue((config_dir / "training/training_0/training_grid_prior.pdf").exists())

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "requires at least 2 GPUs")
    def test_cuda_two_gpu_no_save_smoke(self):
        result = self._run_script(
            3,
            "--device",
            "cuda",
            "--max-gpus",
            2,
            "--no-save",
            "--n-train",
            48,
            "--n-test",
            12,
            "--n-validation",
            12,
            "--n-epochs",
            1,
            "--n-hidden-layers",
            1,
            "--n-units",
            8,
            "--prior-histogram-samples",
            96,
            "--prior-histogram-bins",
            8,
            "--posterior-grid-points",
            11,
            "--n-parameters-to-infer-per-dim",
            2,
            "--n-repetitions-per-parameter",
            1,
            "--posterior-qmc-samples",
            1024,
            "--posterior-eval-batch-size",
            512,
            "--n-ratio-samples",
            32,
            "--n-ratio-test-parameters",
            2,
            "--n-test-data",
            128,
            "--n-bins",
            16,
            "--verification-pair-count",
            4,
            "--verification-model-samples-per-endpoint",
            8,
            "--verification-reweight-source-samples",
            16,
            "--verification-reweight-target-samples",
            16,
            "--verification-swd-projections",
            4,
        )
        self.assertIn("Batch run completed successfully.", result.stdout)

    def test_pdf_outputs_and_no_svg_emission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"
            self._run_script(
                2,
                "--device",
                "cpu",
                "--max-gpus",
                1,
                "--output-root",
                output_root,
                "--n-train",
                48,
                "--n-test",
                12,
                "--n-validation",
                12,
                "--n-epochs",
                1,
                "--n-hidden-layers",
                1,
                "--n-units",
                8,
                "--prior-histogram-samples",
                96,
                "--prior-histogram-bins",
                8,
                "--posterior-grid-points",
                11,
                "--n-parameters-to-infer-per-dim",
                2,
                "--n-repetitions-per-parameter",
                1,
                "--posterior-qmc-samples",
                1024,
                "--posterior-eval-batch-size",
                512,
                "--n-ratio-samples",
                32,
                "--n-ratio-test-parameters",
                2,
                "--n-test-data",
                128,
                "--n-bins",
                16,
                "--verification-pair-count",
                4,
                "--verification-model-samples-per-endpoint",
                8,
                "--verification-reweight-source-samples",
                16,
                "--verification-reweight-target-samples",
                16,
                "--verification-swd-projections",
                4,
            )

            pdfs = list(output_root.glob("**/*.pdf"))
            svgs = list(output_root.glob("**/*.svg"))
            self.assertTrue(pdfs)
            self.assertFalse(svgs)
            pdf_names = {path.name for path in pdfs}
            self.assertIn("log_ratio_error_vs_distance.pdf", pdf_names)
            self.assertIn("log_ratio_exact_vs_predicted.pdf", pdf_names)
            self.assertIn("reweighting_swd_vs_distance.pdf", pdf_names)
            self.assertIn("reweighting_summary.pdf", pdf_names)


if __name__ == "__main__":
    unittest.main()
