from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONFIG_IDS = [
    "config_018",
    "config_013",
    "config_012",
    "config_014",
    "config_015",
    "config_017",
]
DEFAULT_PRIORS = ["uniform", "normal", "exponential", "grid"]
RAW_MAX_POINTS_PER_CONFIG = 10_000


def _normalize_config_id(config_id: str | Path) -> str:
    value = Path(config_id).name
    if not value.startswith("config_"):
        value = f"config_{value}"
    return value


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _summary_paths(figs_root: Path, config_id: str) -> dict[str, Path]:
    config_dir = figs_root / config_id
    return {
        "config_json": config_dir / "config.json",
        "analysis_bundle_json": config_dir / "posterior_errors" / "analysis_bundle.json",
        "model_quality_summary_json": config_dir / "verifications" / "model_quality_summary.json",
        "reweighting_quality_summary_json": config_dir / "verifications" / "reweighting_quality_summary.json",
    }


def _raw_paths(figs_root: Path, config_id: str) -> dict[str, Path]:
    config_dir = figs_root / config_id
    return {
        "config_json": config_dir / "config.json",
        "analysis_bundle_json": config_dir / "posterior_errors" / "analysis_bundle.json",
    }


def _coerce_requested_priors(priors: list[str] | tuple[str, ...] | None) -> list[str]:
    if priors is None:
        return list(DEFAULT_PRIORS)
    return [str(prior) for prior in priors]


def _warn_skip(config_id: str, reason: str) -> None:
    warnings.warn(f"Skipping {config_id}: {reason}", stacklevel=2)


def _resolve_existing_summary_payload(
    figs_root: Path,
    config_id: str,
) -> tuple[dict[str, Path], dict, dict, dict, dict] | None:
    paths = _summary_paths(figs_root, config_id)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        _warn_skip(config_id, f"missing required summary files: {', '.join(missing)}")
        return None
    return (
        paths,
        _load_json(paths["config_json"]),
        _load_json(paths["analysis_bundle_json"]),
        _load_json(paths["model_quality_summary_json"]),
        _load_json(paths["reweighting_quality_summary_json"]),
    )


def _merge_curve_payload(prior_curve: dict) -> pd.DataFrame:
    curve_68 = pd.DataFrame(
        {
            "x_value": np.asarray(prior_curve["x"], dtype=float),
            "avg_bias": np.asarray(prior_curve["avg_bias"], dtype=float),
            "avg_width_68": np.asarray(prior_curve["avg_width_68"], dtype=float),
        }
    )
    width_95 = prior_curve.get("width_curves", {}).get("95", {})
    curve_95 = pd.DataFrame(
        {
            "x_value": np.asarray(width_95.get("x", []), dtype=float),
            "avg_width_95": np.asarray(width_95.get("avg_width", []), dtype=float),
        }
    )
    if curve_95.empty:
        curve_68["avg_width_95"] = np.nan
        return curve_68
    return curve_68.merge(curve_95, on="x_value", how="outer", sort=True)


def _pair_frame(
    *,
    config_id: str,
    dimension: int,
    prior: str,
    model_pair_metrics: dict,
    reweight_pair_metrics: dict,
) -> pd.DataFrame:
    model_lengths = {name: len(values) for name, values in model_pair_metrics.items()}
    reweight_lengths = {name: len(values) for name, values in reweight_pair_metrics.items()}
    n_pairs = min([*model_lengths.values(), *reweight_lengths.values()])
    if n_pairs == 0:
        return pd.DataFrame()

    distance_model = np.asarray(model_pair_metrics["distance"][:n_pairs], dtype=float)
    distance_reweight = np.asarray(reweight_pair_metrics["distance"][:n_pairs], dtype=float)
    if not np.allclose(distance_model, distance_reweight, equal_nan=True):
        warnings.warn(
            f"{config_id}/{prior} has mismatched pairwise distance arrays; using model_quality_summary distances.",
            stacklevel=2,
        )

    return pd.DataFrame(
        {
            "config_id": config_id,
            "dimension": int(dimension),
            "prior": prior,
            "pair_index": np.arange(n_pairs, dtype=int),
            "distance": distance_model,
            "rmse_log_r10": np.asarray(model_pair_metrics["rmse_log_r10"][:n_pairs], dtype=float),
            "mae_log_r10": np.asarray(model_pair_metrics["mae_log_r10"][:n_pairs], dtype=float),
            "max_abs_log_r10": np.asarray(model_pair_metrics["max_abs_log_r10"][:n_pairs], dtype=float),
            "ess": np.asarray(reweight_pair_metrics["ess"][:n_pairs], dtype=float),
            "ess_fraction": np.asarray(reweight_pair_metrics["ess_fraction"][:n_pairs], dtype=float),
            "swd": np.asarray(reweight_pair_metrics["swd"][:n_pairs], dtype=float),
        }
    )


def load_dimensionality_summary(
    figs_root: str | Path,
    config_ids: list[str] | tuple[str, ...] | None = None,
    priors: list[str] | tuple[str, ...] | None = None,
) -> dict[str, pd.DataFrame | list[str]]:
    figs_root = Path(figs_root)
    requested_priors = _coerce_requested_priors(priors)
    requested_configs = [
        _normalize_config_id(config_id)
        for config_id in (config_ids if config_ids is not None else DEFAULT_CONFIG_IDS)
    ]

    overview_rows: list[dict] = []
    run_rows: list[dict] = []
    pair_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    resolved_configs: list[str] = []

    for config_id in requested_configs:
        payload = _resolve_existing_summary_payload(figs_root, config_id)
        if payload is None:
            continue
        _, config, analysis_bundle, model_quality, reweight_quality = payload

        dimension = int(config["data"]["n_parameters"])
        workload = analysis_bundle["posterior_workload"]
        available_priors = [
            prior
            for prior in requested_priors
            if prior in workload["prior_names"]
            and prior in analysis_bundle["bias_summary"]
            and prior in analysis_bundle["hpd_summary"]["ratio"]
            and prior in model_quality
            and prior in reweight_quality
        ]
        if not available_priors:
            _warn_skip(config_id, "none of the requested priors were present in the summary artifacts")
            continue

        resolved_configs.append(config_id)
        overview_rows.append(
            {
                "config_id": config_id,
                "dimension": dimension,
                "priors_present": ", ".join(available_priors),
                "n_priors_present": len(available_priors),
                "n_inferred_parameter_settings": int(workload["n_inferred_parameter_settings"]),
                "n_repetitions_per_setting": int(workload["n_repetitions_per_setting"]),
                "total_posterior_combinations_per_prior": int(
                    workload["total_posterior_combinations_per_prior"]
                ),
                "total_posterior_combinations_all_priors": int(
                    workload["total_posterior_combinations_all_priors"]
                ),
                "posterior_grid_shape": tuple(int(value) for value in workload["posterior_grid_shape"]),
                "compact_posterior_storage_written": bool(
                    workload["compact_posterior_storage_written"]
                ),
            }
        )

        for prior in available_priors:
            hpd_ratio = analysis_bundle["hpd_summary"]["ratio"][prior]
            model_summary = model_quality[prior]["summary"]
            reweight_summary = reweight_quality[prior]["summary"]

            run_rows.append(
                {
                    "config_id": config_id,
                    "dimension": dimension,
                    "prior": prior,
                    "n_inferred_parameter_settings": int(workload["n_inferred_parameter_settings"]),
                    "n_repetitions_per_setting": int(workload["n_repetitions_per_setting"]),
                    "total_posterior_combinations_per_prior": int(
                        workload["total_posterior_combinations_per_prior"]
                    ),
                    "total_posterior_combinations_all_priors": int(
                        workload["total_posterior_combinations_all_priors"]
                    ),
                    "posterior_grid_shape": tuple(int(value) for value in workload["posterior_grid_shape"]),
                    "median_rmse_log_r10": float(model_summary["median_rmse_log_r10"]),
                    "pooled_rmse_log_r10": float(model_summary["pooled_rmse_log_r10"]),
                    "worst_rmse_log_r10": float(model_summary["worst_rmse_log_r10"]),
                    "spearman_rmse_vs_distance": float(model_summary["spearman_rmse_vs_distance"]),
                    "median_ess_fraction": float(reweight_summary["median_ess_fraction"]),
                    "median_swd": float(reweight_summary["median_swd"]),
                    "worst_ess_fraction": float(reweight_summary["worst_ess_fraction"]),
                    "worst_swd": float(reweight_summary["worst_swd"]),
                    "spearman_swd_vs_distance": float(reweight_summary["spearman_swd_vs_distance"]),
                    "coverage_68": float(hpd_ratio["68"]["coverage_fraction_mean"]),
                    "coverage_95": float(hpd_ratio["95"]["coverage_fraction_mean"]),
                    "mean_interval_width_68": float(hpd_ratio["68"]["mean_interval_width"]),
                    "mean_interval_width_95": float(hpd_ratio["95"]["mean_interval_width"]),
                    "mean_combined_width_68": float(hpd_ratio["68"]["mean_combined_width"]),
                    "mean_combined_width_95": float(hpd_ratio["95"]["mean_combined_width"]),
                }
            )

            pair_frame = _pair_frame(
                config_id=config_id,
                dimension=dimension,
                prior=prior,
                model_pair_metrics=model_quality[prior]["pair_metrics"],
                reweight_pair_metrics=reweight_quality[prior]["pair_metrics"],
            )
            if not pair_frame.empty:
                pair_frames.append(pair_frame)

            curve_frame = _merge_curve_payload(analysis_bundle["bias_summary"][prior])
            curve_frame.insert(0, "prior", prior)
            curve_frame.insert(0, "dimension", dimension)
            curve_frame.insert(0, "config_id", config_id)
            curve_frames.append(curve_frame)

    config_overview_df = pd.DataFrame(overview_rows).sort_values(
        ["dimension", "config_id"], ignore_index=True
    )
    runs_df = pd.DataFrame(run_rows).sort_values(
        ["dimension", "prior", "config_id"], ignore_index=True
    )
    pairs_df = (
        pd.concat(pair_frames, ignore_index=True)
        if pair_frames
        else pd.DataFrame(
            columns=[
                "config_id",
                "dimension",
                "prior",
                "pair_index",
                "distance",
                "rmse_log_r10",
                "mae_log_r10",
                "max_abs_log_r10",
                "ess",
                "ess_fraction",
                "swd",
            ]
        )
    )
    curves_df = (
        pd.concat(curve_frames, ignore_index=True).sort_values(
            ["dimension", "prior", "x_value"], ignore_index=True
        )
        if curve_frames
        else pd.DataFrame(
            columns=[
                "config_id",
                "dimension",
                "prior",
                "x_value",
                "avg_bias",
                "avg_width_68",
                "avg_width_95",
            ]
        )
    )

    return {
        "config_overview_df": config_overview_df,
        "runs_df": runs_df,
        "pairs_df": pairs_df,
        "curves_df": curves_df,
        "resolved_config_ids": resolved_configs,
    }


def _find_compact_store_artifact(
    analysis_bundle: dict,
    *,
    bundle_dir: Path,
    prior: str,
    suffix: str,
) -> Path | None:
    for relative_path in analysis_bundle["artifacts"]["compact_store_files"].get(prior, []):
        if str(relative_path).endswith(suffix):
            return bundle_dir / relative_path
    return None


def _select_raw_indices(n_points: int, max_points_per_config: int) -> tuple[np.ndarray, bool]:
    if n_points <= max_points_per_config:
        return np.arange(n_points, dtype=np.int64), False
    indices = np.linspace(0, n_points - 1, num=max_points_per_config, dtype=np.int64)
    return np.unique(indices), True


def _true_parameter_values_from_indices(
    *,
    x_values: np.ndarray,
    n_dimensions: int,
    flat_indices: np.ndarray,
) -> np.ndarray:
    grid_shape = (int(x_values.size),) * int(n_dimensions)
    coordinates = np.unravel_index(flat_indices, grid_shape)
    return np.stack([x_values[np.asarray(coord, dtype=int)] for coord in coordinates], axis=1)


def _append_raw_frames(
    *,
    map_frames: list[pd.DataFrame],
    interval_frames: list[pd.DataFrame],
    metadata_rows: list[dict],
    config_id: str,
    dimension: int,
    prior: str,
    selected_indices: np.ndarray,
    true_values: np.ndarray,
    ratio_map_subset: np.ndarray,
    hpd_68_subset: np.ndarray,
    hpd_95_subset: np.ndarray,
    n_total_points: int,
    was_sampled: bool,
) -> None:
    metadata_rows.append(
        {
            "config_id": config_id,
            "dimension": dimension,
            "prior": prior,
            "n_total_points": int(n_total_points),
            "n_loaded_points": int(selected_indices.size),
            "sampled": bool(was_sampled),
        }
    )

    for dim_index in range(dimension):
        true_column = true_values[:, dim_index]
        map_column = ratio_map_subset[:, dim_index]
        map_frames.append(
            pd.DataFrame(
                {
                    "config_id": config_id,
                    "dimension": dimension,
                    "prior": prior,
                    "sample_index": selected_indices,
                    "dim_index": dim_index,
                    "true_value": true_column,
                    "map_estimate": map_column,
                    "map_error": map_column - true_column,
                    "n_total_points": int(n_total_points),
                    "sampled": bool(was_sampled),
                }
            )
        )

        for level_name, interval_subset in (("68", hpd_68_subset), ("95", hpd_95_subset)):
            low = interval_subset[:, dim_index, 0]
            high = interval_subset[:, dim_index, 1]
            interval_frames.append(
                pd.DataFrame(
                    {
                        "config_id": config_id,
                        "dimension": dimension,
                        "prior": prior,
                        "interval_level": level_name,
                        "sample_index": selected_indices,
                        "dim_index": dim_index,
                        "true_value": true_column,
                        "low": low,
                        "high": high,
                        "width": high - low,
                        "contains_true": (true_column >= low) & (true_column <= high),
                        "n_total_points": int(n_total_points),
                        "sampled": bool(was_sampled),
                    }
                )
            )


def load_raw_compact_store(
    figs_root: str | Path,
    config_ids: list[str] | tuple[str, ...],
    priors: list[str] | tuple[str, ...] | None = None,
    *,
    max_points_per_config: int = RAW_MAX_POINTS_PER_CONFIG,
) -> dict[str, pd.DataFrame]:
    figs_root = Path(figs_root)
    requested_priors = _coerce_requested_priors(priors)
    requested_configs = [_normalize_config_id(config_id) for config_id in config_ids]

    map_frames: list[pd.DataFrame] = []
    interval_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []

    for config_id in requested_configs:
        raw_paths = _raw_paths(figs_root, config_id)
        missing = [name for name, path in raw_paths.items() if not path.exists()]
        if missing:
            _warn_skip(config_id, f"missing raw-analysis prerequisites: {', '.join(missing)}")
            continue

        config = _load_json(raw_paths["config_json"])
        analysis_bundle = _load_json(raw_paths["analysis_bundle_json"])
        dimension = int(config["data"]["n_parameters"])
        bundle_dir = raw_paths["analysis_bundle_json"].parent
        raw_diagnostic_files = analysis_bundle.get("artifacts", {}).get("raw_diagnostic_files", {})
        if raw_diagnostic_files:
            available_priors = [
                prior
                for prior in requested_priors
                if prior in raw_diagnostic_files
            ]
            for prior in available_priors:
                raw_path = bundle_dir / raw_diagnostic_files[prior]
                if not raw_path.exists():
                    _warn_skip(
                        config_id,
                        f"raw diagnostic artifact for prior '{prior}' is missing: {raw_path}",
                    )
                    continue

                with np.load(raw_path) as raw:
                    sample_index = np.asarray(raw["sample_index"], dtype=np.int64)
                    true_values_all = np.asarray(raw["true_params"], dtype=np.float32)
                    ratio_map_all = np.asarray(raw["ratio_map"], dtype=np.float32)
                    hpd_68_all = np.asarray(raw["ratio_hpd_68"], dtype=np.float32)
                    hpd_95_all = np.asarray(raw["ratio_hpd_95"], dtype=np.float32)

                n_loaded_total = int(sample_index.size)
                selected_positions, was_sampled = _select_raw_indices(n_loaded_total, max_points_per_config)
                n_total_points = int(
                    analysis_bundle.get("posterior_workload", {}).get(
                        "n_evaluated_parameter_settings",
                        n_loaded_total,
                    )
                )
                _append_raw_frames(
                    map_frames=map_frames,
                    interval_frames=interval_frames,
                    metadata_rows=metadata_rows,
                    config_id=config_id,
                    dimension=dimension,
                    prior=prior,
                    selected_indices=sample_index[selected_positions],
                    true_values=true_values_all[selected_positions],
                    ratio_map_subset=ratio_map_all[selected_positions],
                    hpd_68_subset=hpd_68_all[selected_positions],
                    hpd_95_subset=hpd_95_all[selected_positions],
                    n_total_points=n_total_points,
                    was_sampled=was_sampled or n_loaded_total < n_total_points,
                )
            continue

        first_prior_name = next(iter(analysis_bundle["bias_summary"]), None)
        if first_prior_name is None:
            _warn_skip(config_id, "analysis bundle does not contain bias_summary data")
            continue
        x_values = np.asarray(analysis_bundle["bias_summary"][first_prior_name]["x"], dtype=np.float32)
        if x_values.size == 0:
            _warn_skip(config_id, "analysis bundle does not contain any inference x-values")
            continue

        available_priors = [
            prior
            for prior in requested_priors
            if prior in analysis_bundle["artifacts"]["compact_store_files"]
        ]
        for prior in available_priors:
            ratio_map_path = _find_compact_store_artifact(
                analysis_bundle,
                bundle_dir=bundle_dir,
                prior=prior,
                suffix="ratio_map.npy",
            )
            hpd_68_path = _find_compact_store_artifact(
                analysis_bundle,
                bundle_dir=bundle_dir,
                prior=prior,
                suffix="ratio_hpd_68_intervals.npy",
            )
            hpd_95_path = _find_compact_store_artifact(
                analysis_bundle,
                bundle_dir=bundle_dir,
                prior=prior,
                suffix="ratio_hpd_95_intervals.npy",
            )
            missing_artifacts = [
                name
                for name, path in {
                    "ratio_map.npy": ratio_map_path,
                    "ratio_hpd_68_intervals.npy": hpd_68_path,
                    "ratio_hpd_95_intervals.npy": hpd_95_path,
                }.items()
                if path is None or not path.exists()
            ]
            if missing_artifacts:
                _warn_skip(
                    config_id,
                    f"raw compact-store artifacts for prior '{prior}' are missing: {', '.join(missing_artifacts)}",
                )
                continue

            ratio_map = np.load(ratio_map_path, mmap_mode="r")
            hpd_68 = np.load(hpd_68_path, mmap_mode="r")
            hpd_95 = np.load(hpd_95_path, mmap_mode="r")
            n_points = int(ratio_map.shape[0])
            selected_indices, was_sampled = _select_raw_indices(n_points, max_points_per_config)
            true_values = _true_parameter_values_from_indices(
                x_values=x_values,
                n_dimensions=dimension,
                flat_indices=selected_indices,
            )

            ratio_map_subset = np.asarray(ratio_map[selected_indices], dtype=np.float32)
            hpd_68_subset = np.asarray(hpd_68[selected_indices], dtype=np.float32)
            hpd_95_subset = np.asarray(hpd_95[selected_indices], dtype=np.float32)

            metadata_rows.append(
                {
                    "config_id": config_id,
                    "dimension": dimension,
                    "prior": prior,
                    "n_total_points": n_points,
                    "n_loaded_points": int(selected_indices.size),
                    "sampled": bool(was_sampled),
                }
            )

            for dim_index in range(dimension):
                true_column = true_values[:, dim_index]
                map_column = ratio_map_subset[:, dim_index]
                map_frames.append(
                    pd.DataFrame(
                        {
                            "config_id": config_id,
                            "dimension": dimension,
                            "prior": prior,
                            "sample_index": selected_indices,
                            "dim_index": dim_index,
                            "true_value": true_column,
                            "map_estimate": map_column,
                            "map_error": map_column - true_column,
                            "n_total_points": n_points,
                            "sampled": bool(was_sampled),
                        }
                    )
                )

                for level_name, interval_subset in (("68", hpd_68_subset), ("95", hpd_95_subset)):
                    low = interval_subset[:, dim_index, 0]
                    high = interval_subset[:, dim_index, 1]
                    interval_frames.append(
                        pd.DataFrame(
                            {
                                "config_id": config_id,
                                "dimension": dimension,
                                "prior": prior,
                                "interval_level": level_name,
                                "sample_index": selected_indices,
                                "dim_index": dim_index,
                                "true_value": true_column,
                                "low": low,
                                "high": high,
                                "width": high - low,
                                "contains_true": (true_column >= low) & (true_column <= high),
                                "n_total_points": n_points,
                                "sampled": bool(was_sampled),
                            }
                        )
                    )

    raw_map_df = (
        pd.concat(map_frames, ignore_index=True).sort_values(
            ["dimension", "prior", "sample_index", "dim_index"], ignore_index=True
        )
        if map_frames
        else pd.DataFrame(
            columns=[
                "config_id",
                "dimension",
                "prior",
                "sample_index",
                "dim_index",
                "true_value",
                "map_estimate",
                "map_error",
                "n_total_points",
                "sampled",
            ]
        )
    )
    raw_intervals_df = (
        pd.concat(interval_frames, ignore_index=True).sort_values(
            ["dimension", "prior", "interval_level", "sample_index", "dim_index"],
            ignore_index=True,
        )
        if interval_frames
        else pd.DataFrame(
            columns=[
                "config_id",
                "dimension",
                "prior",
                "interval_level",
                "sample_index",
                "dim_index",
                "true_value",
                "low",
                "high",
                "width",
                "contains_true",
                "n_total_points",
                "sampled",
            ]
        )
    )
    raw_metadata_df = pd.DataFrame(metadata_rows).sort_values(
        ["dimension", "prior", "config_id"], ignore_index=True
    )

    return {
        "raw_map_df": raw_map_df,
        "raw_intervals_df": raw_intervals_df,
        "raw_metadata_df": raw_metadata_df,
    }
