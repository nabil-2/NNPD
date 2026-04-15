from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dimensionality_analysis import (
    DEFAULT_CONFIG_IDS,
    DEFAULT_PRIORS,
    load_dimensionality_summary,
    load_raw_compact_store,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGS_ROOT = REPO_ROOT / "figs"


class DimensionalityAnalysisLoaderTests(unittest.TestCase):
    def test_default_prior_list_includes_grid(self):
        self.assertIn("grid", DEFAULT_PRIORS)

    def test_summary_loader_uses_clean_1d_to_6d_sweep(self):
        payload = load_dimensionality_summary(FIGS_ROOT, DEFAULT_CONFIG_IDS, DEFAULT_PRIORS)

        config_overview_df = payload["config_overview_df"]
        runs_df = payload["runs_df"]
        pairs_df = payload["pairs_df"]
        curves_df = payload["curves_df"]

        self.assertEqual(config_overview_df["dimension"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(config_overview_df["config_id"].tolist(), DEFAULT_CONFIG_IDS)
        self.assertEqual(len(runs_df), 24)
        self.assertEqual(len(pairs_df), 24 * 256)
        self.assertFalse(curves_df.empty)
        self.assertEqual(
            sorted(runs_df["prior"].unique().tolist()),
            ["exponential", "grid", "normal", "uniform"],
        )

    def test_summary_loader_skips_legacy_config_without_new_summary_jsons(self):
        payload = load_dimensionality_summary(FIGS_ROOT, ["config_011", "config_018"], DEFAULT_PRIORS)

        self.assertEqual(payload["resolved_config_ids"], ["config_018"])
        self.assertEqual(payload["config_overview_df"]["config_id"].tolist(), ["config_018"])
        self.assertEqual(payload["runs_df"]["dimension"].unique().tolist(), [1])

    def test_raw_loader_samples_large_high_dimensional_runs(self):
        raw_payload = load_raw_compact_store(
            FIGS_ROOT,
            ["config_018", "config_017"],
            ["uniform"],
            max_points_per_config=256,
        )

        raw_map_df = raw_payload["raw_map_df"]
        raw_intervals_df = raw_payload["raw_intervals_df"]
        raw_metadata_df = raw_payload["raw_metadata_df"]

        self.assertFalse(raw_map_df.empty)
        self.assertFalse(raw_intervals_df.empty)
        self.assertEqual(sorted(raw_metadata_df["config_id"].tolist()), ["config_017", "config_018"])

        six_d_row = raw_metadata_df.loc[raw_metadata_df["config_id"] == "config_017"].iloc[0]
        self.assertTrue(bool(six_d_row["sampled"]))
        self.assertEqual(int(six_d_row["n_loaded_points"]), 256)

        six_d_map = raw_map_df.loc[raw_map_df["config_id"] == "config_017"]
        six_d_intervals = raw_intervals_df.loc[raw_intervals_df["config_id"] == "config_017"]
        self.assertEqual(sorted(six_d_map["dim_index"].unique().tolist()), [0, 1, 2, 3, 4, 5])
        self.assertEqual(sorted(six_d_intervals["interval_level"].unique().tolist()), ["68", "95"])

    def test_grid_prior_is_loadable_when_summary_and_raw_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            figs_root = Path(tmp_dir) / "figs"
            config_dir = figs_root / "config_grid_fixture"
            (config_dir / "posterior_errors" / "compact_store").mkdir(parents=True)
            (config_dir / "verifications").mkdir(parents=True)

            (config_dir / "config.json").write_text(
                json.dumps({"data": {"n_parameters": 2}})
            )

            analysis_bundle = {
                "posterior_workload": {
                    "prior_names": ["grid"],
                    "n_inferred_parameter_settings": 9,
                    "n_repetitions_per_setting": 1,
                    "total_posterior_combinations_per_prior": 9,
                    "total_posterior_combinations_all_priors": 9,
                    "posterior_grid_shape": [11, 11],
                    "compact_posterior_storage_written": True,
                },
                "bias_summary": {
                    "grid": {
                        "x": [0.0, 0.5, 1.0],
                        "avg_bias": [0.1, 0.0, -0.1],
                        "avg_width_68": [0.2, 0.25, 0.3],
                        "width_curves": {
                            "95": {
                                "x": [0.0, 0.5, 1.0],
                                "avg_width": [0.4, 0.45, 0.5],
                            }
                        },
                    }
                },
                "hpd_summary": {
                    "ratio": {
                        "grid": {
                            "68": {
                                "coverage_fraction_mean": 0.66,
                                "mean_interval_width": 0.23,
                                "mean_combined_width": 0.24,
                            },
                            "95": {
                                "coverage_fraction_mean": 0.93,
                                "mean_interval_width": 0.47,
                                "mean_combined_width": 0.48,
                            },
                        }
                    }
                },
                "artifacts": {
                    "compact_store_files": {
                        "grid": [
                            "compact_store/grid_ratio_map.npy",
                            "compact_store/grid_ratio_hpd_68_intervals.npy",
                            "compact_store/grid_ratio_hpd_95_intervals.npy",
                        ]
                    }
                },
            }
            (config_dir / "posterior_errors" / "analysis_bundle.json").write_text(
                json.dumps(analysis_bundle)
            )

            model_quality = {
                "grid": {
                    "summary": {
                        "median_rmse_log_r10": 0.11,
                        "pooled_rmse_log_r10": 0.12,
                        "worst_rmse_log_r10": 0.2,
                        "spearman_rmse_vs_distance": 0.8,
                    },
                    "pair_metrics": {
                        "distance": [0.1, 0.2, 0.3],
                        "rmse_log_r10": [0.1, 0.12, 0.14],
                        "mae_log_r10": [0.09, 0.1, 0.12],
                        "max_abs_log_r10": [0.2, 0.22, 0.25],
                    },
                }
            }
            (config_dir / "verifications" / "model_quality_summary.json").write_text(
                json.dumps(model_quality)
            )

            reweight_quality = {
                "grid": {
                    "summary": {
                        "median_ess_fraction": 0.8,
                        "median_swd": 0.03,
                        "worst_ess_fraction": 0.5,
                        "worst_swd": 0.08,
                        "spearman_swd_vs_distance": 0.6,
                    },
                    "pair_metrics": {
                        "distance": [0.1, 0.2, 0.3],
                        "ess": [80.0, 70.0, 60.0],
                        "ess_fraction": [0.8, 0.7, 0.6],
                        "swd": [0.02, 0.03, 0.05],
                    },
                }
            }
            (config_dir / "verifications" / "reweighting_quality_summary.json").write_text(
                json.dumps(reweight_quality)
            )

            ratio_map = np.array(
                [
                    [0.00, 0.00],
                    [0.55, 0.45],
                    [1.02, 0.98],
                ],
                dtype=np.float32,
            )
            hpd_68 = np.array(
                [
                    [[-0.10, 0.10], [-0.10, 0.10]],
                    [[0.40, 0.70], [0.30, 0.60]],
                    [[0.90, 1.10], [0.85, 1.05]],
                ],
                dtype=np.float32,
            )
            hpd_95 = np.array(
                [
                    [[-0.20, 0.20], [-0.20, 0.20]],
                    [[0.30, 0.80], [0.20, 0.70]],
                    [[0.80, 1.20], [0.75, 1.15]],
                ],
                dtype=np.float32,
            )
            np.save(config_dir / "posterior_errors" / "compact_store" / "grid_ratio_map.npy", ratio_map)
            np.save(
                config_dir / "posterior_errors" / "compact_store" / "grid_ratio_hpd_68_intervals.npy",
                hpd_68,
            )
            np.save(
                config_dir / "posterior_errors" / "compact_store" / "grid_ratio_hpd_95_intervals.npy",
                hpd_95,
            )

            summary_payload = load_dimensionality_summary(
                figs_root,
                ["config_grid_fixture"],
                ["grid"],
            )
            raw_payload = load_raw_compact_store(
                figs_root,
                ["config_grid_fixture"],
                ["grid"],
                max_points_per_config=10,
            )

            self.assertEqual(summary_payload["resolved_config_ids"], ["config_grid_fixture"])
            self.assertEqual(summary_payload["runs_df"]["prior"].tolist(), ["grid"])
            self.assertEqual(summary_payload["pairs_df"]["prior"].unique().tolist(), ["grid"])
            self.assertEqual(summary_payload["curves_df"]["prior"].unique().tolist(), ["grid"])

            self.assertEqual(raw_payload["raw_map_df"]["prior"].unique().tolist(), ["grid"])
            self.assertEqual(raw_payload["raw_intervals_df"]["prior"].unique().tolist(), ["grid"])
            self.assertEqual(raw_payload["raw_metadata_df"]["prior"].tolist(), ["grid"])
            self.assertFalse(bool(raw_payload["raw_metadata_df"].iloc[0]["sampled"]))


if __name__ == "__main__":
    unittest.main()
