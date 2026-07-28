from __future__ import annotations

import concurrent.futures
import math
import multiprocessing as mp
import queue as queue_module
from dataclasses import dataclass
from pathlib import Path

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


def _normalize_hpd_entry(raw_entry, n_dims):
    intervals = raw_entry
    interval_combined = None

    if isinstance(raw_entry, dict):
        intervals = raw_entry.get("intervals", raw_entry.get("interval", None))
        interval_combined = raw_entry.get("interval_combined", raw_entry.get("combined", None))
    elif isinstance(raw_entry, (tuple, list)) and len(raw_entry) == 2:
        intervals, interval_combined = raw_entry

    intervals = np.asarray(intervals, dtype=float)
    if intervals.ndim == 1 and intervals.size == 2 and n_dims == 1:
        intervals = intervals.reshape(1, 2)
    if intervals.shape != (n_dims, 2):
        raise ValueError(f"Expected HPD intervals with shape ({n_dims}, 2), got {intervals.shape}.")

    if interval_combined is None:
        interval_combined = np.asarray(
            [float(np.mean(intervals[:, 0])), float(np.mean(intervals[:, 1]))],
            dtype=float,
        )
    else:
        interval_combined = np.asarray(interval_combined, dtype=float)
        if interval_combined.shape != (2,):
            raise ValueError(
                f"Expected combined HPD interval with shape (2,), got {interval_combined.shape}."
            )

    return intervals, interval_combined


def _array_ref(path, shape, dtype):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "shape": tuple(int(v) for v in shape),
        "dtype": np.dtype(dtype).name,
    }


def _is_array_ref(value):
    return isinstance(value, dict) and {"path", "shape", "dtype"} <= set(value)


def _load_array(value, mmap_mode="r"):
    if _is_array_ref(value):
        return np.load(value["path"], mmap_mode=mmap_mode)
    return np.asarray(value)


def _create_array_storage(path, shape, dtype):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=tuple(int(v) for v in shape))
    return arr, _array_ref(path, shape, dtype)


def _create_compact_map_store(n_points, n_dims, storage_dir=None, basename="posterior_map"):
    shape = (int(n_points), int(n_dims))
    if storage_dir is None:
        map_array = np.empty(shape, dtype=np.float32)
        map_ref = map_array
    else:
        map_array, map_ref = _create_array_storage(Path(storage_dir) / f"{basename}.npy", shape, np.float32)

    return (
        {
            "format": "compact_map_store",
            "n_points": int(n_points),
            "n_dims": int(n_dims),
            "dtype": "float32",
            "map": map_ref,
        },
        map_array,
    )


def _create_compact_hpd_store(n_points, n_dims, storage_dir=None, basename="posterior_hpd"):
    shape_intervals = (int(n_points), int(n_dims), 2)
    shape_combined = (int(n_points), 2)
    writable = {}
    store = {
        "format": "compact_hpd_store",
        "n_points": int(n_points),
        "n_dims": int(n_dims),
        "dtype": "float32",
    }

    for level in ("68", "95"):
        if storage_dir is None:
            intervals_array = np.empty(shape_intervals, dtype=np.float32)
            combined_array = np.empty(shape_combined, dtype=np.float32)
            intervals_ref = intervals_array
            combined_ref = combined_array
        else:
            intervals_array, intervals_ref = _create_array_storage(
                Path(storage_dir) / f"{basename}_{level}_intervals.npy",
                shape_intervals,
                np.float32,
            )
            combined_array, combined_ref = _create_array_storage(
                Path(storage_dir) / f"{basename}_{level}_combined.npy",
                shape_combined,
                np.float32,
            )

        writable[level] = {
            "intervals": intervals_array,
            "interval_combined": combined_array,
        }
        store[level] = {
            "intervals": intervals_ref,
            "interval_combined": combined_ref,
        }

    return store, writable


def _flush_arrays(arrays):
    for arr in arrays:
        if hasattr(arr, "flush"):
            arr.flush()


def _is_compact_map_store(value):
    return isinstance(value, dict) and value.get("format") == "compact_map_store" and "map" in value


def _is_compact_hpd_store(value):
    return isinstance(value, dict) and value.get("format") == "compact_hpd_store" and "68" in value and "95" in value


def _extract_map_array(raw_posteriors, parameters_post=None, n_dims=None):
    if _is_compact_map_store(raw_posteriors):
        map_array = _load_array(raw_posteriors["map"], mmap_mode="r")
        if map_array.ndim != 2:
            raise ValueError(f"Expected compact posterior map array with 2 dims, got {map_array.shape}.")
        if n_dims is not None and map_array.shape[1] != int(n_dims):
            raise ValueError(
                f"Expected compact posterior map array with {n_dims} dims, got {map_array.shape[1]}."
            )
        return map_array

    posts = raw_posteriors
    if n_dims is None:
        if parameters_post is None:
            raise ValueError("parameters_post is required to infer dimensions for legacy posterior entries.")
        n_dims = len(parameters_post)

    y_hat = np.zeros((len(posts), int(n_dims)), dtype=np.float32)
    dim_lengths = [len(r) for r in parameters_post] if parameters_post is not None else None

    for idx, posterior in enumerate(posts):
        if isinstance(posterior, dict) and "map" in posterior:
            y_hat[idx, :] = np.asarray(posterior["map"], dtype=np.float32)
            continue

        if dim_lengths is None or parameters_post is None:
            raise ValueError("parameters_post is required for legacy posterior grids without MAP entries.")

        arr = np.asarray(posterior)
        if arr.ndim == 1:
            imax = int(np.argmax(arr))
            multi = np.unravel_index(imax, dim_lengths)
        else:
            multi = np.unravel_index(int(np.argmax(arr)), arr.shape)
        for d in range(int(n_dims)):
            y_hat[idx, d] = parameters_post[d][multi[d]]

    return y_hat


def _extract_hpd_level(raw_hpds, level, n_dims):
    if _is_compact_hpd_store(raw_hpds):
        if level not in raw_hpds:
            raise KeyError(f"Missing HPD level '{level}' in compact HPD store.")
        intervals = _load_array(raw_hpds[level]["intervals"], mmap_mode="r")
        interval_combined = _load_array(raw_hpds[level]["interval_combined"], mmap_mode="r")
        expected_shape = (int(raw_hpds["n_points"]), int(n_dims), 2)
        if tuple(intervals.shape) != expected_shape:
            raise ValueError(
                f"Expected compact HPD intervals with shape {expected_shape}, got {intervals.shape}."
            )
        if tuple(interval_combined.shape) != (int(raw_hpds["n_points"]), 2):
            raise ValueError(
                f"Expected compact HPD combined intervals with shape {(int(raw_hpds['n_points']), 2)}, "
                f"got {interval_combined.shape}."
            )
        return intervals, interval_combined

    intervals = np.empty((len(raw_hpds), int(n_dims), 2), dtype=np.float32)
    interval_combined = np.empty((len(raw_hpds), 2), dtype=np.float32)
    for idx, entry in enumerate(raw_hpds):
        intervals[idx], interval_combined[idx] = _normalize_hpd_entry(entry[level], int(n_dims))
    return intervals, interval_combined


def _select_num_cuda_workers(device, max_gpus, n_priors):
    device = torch.device(device)
    if device.type != "cuda":
        return 1
    if max_gpus < 1:
        raise ValueError("max_gpus must be at least 1.")
    return max(1, min(int(max_gpus), torch.cuda.device_count(), n_priors))


def _state_dict_to_cpu(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _build_model_from_state(config, model_state, device):
    from .models import BinaryClassifier

    model = BinaryClassifier(config)
    model.load_state_dict(model_state)
    model.to(device).eval()
    return model


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
        result = np.trapz(result, x_range[i], axis=i)
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
    support_axis=None,
):
    if support_axis is not None:
        support_axis = np.asarray(support_axis, dtype=np.float32).ravel()
        if support_axis.size == 0:
            raise ValueError("support_axis must contain at least one lattice point.")

        parameter_min, parameter_max = parameters_min_max
        parameter_range = parameter_max - parameter_min
        low = parameter_min + parameter_range * margin
        high = parameter_max - parameter_range * margin
        eligible = support_axis[
            (support_axis >= low - 1e-6)
            & (support_axis <= high + 1e-6)
        ]
        if eligible.size < int(n_parameters_to_infer):
            raise ValueError(
                "Not enough interior grid points to satisfy n_parameters_to_infer_per_dim "
                f"for margin={margin}: need {n_parameters_to_infer}, found {eligible.size}."
            )
        if data_is_random:
            choice_idx = generator.choice(eligible.size, size=int(n_parameters_to_infer), replace=False)
            return np.sort(eligible[choice_idx].astype(np.float32))
        if int(n_parameters_to_infer) == 1:
            return np.asarray([eligible[eligible.size // 2]], dtype=np.float32)
        idx = np.rint(
            np.linspace(0, eligible.size - 1, int(n_parameters_to_infer), dtype=np.float64)
        ).astype(int)
        idx = np.clip(idx, 0, eligible.size - 1)
        if np.unique(idx).size != int(n_parameters_to_infer):
            raise ValueError(
                "Unable to choose enough unique lattice-aligned inference points; "
                "try reducing n_parameters_to_infer_per_dim or margin."
            )
        return eligible[idx].astype(np.float32)

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
    support_axes=None,
):
    n_dimensions = config["data"]["n_parameters"]

    if parameters_to_infer is None:
        if support_axes is not None and len(support_axes) != n_dimensions:
            raise ValueError(
                f"Expected support_axes for {n_dimensions} dimensions, got {len(support_axes)}."
            )
        parameters_to_infer = []
        for _ in range(n_dimensions):
            support_axis = None if support_axes is None else support_axes[_]
            parameters_to_infer.append(
                create_inference_parameters(
                    n_parameters_to_infer_per_dim,
                    parameters_min_max,
                    generator,
                    margin,
                    data_is_random,
                    support_axis=support_axis,
                )
            )

    parameter_mesh = np.meshgrid(*parameters_to_infer, indexing="ij")
    parameter_combinations = np.stack(
        [mesh.reshape(-1) for mesh in parameter_mesh],
        axis=1,
    ).astype(np.float32)

    repeated_parameters = np.repeat(
        parameter_combinations[:, None, :],
        int(n_repititions_per_parameter),
        axis=1,
    )
    sampled_data = draw_data(
        repeated_parameters.reshape(-1, n_dimensions),
        config,
        generator,
    ).reshape(-1, int(n_repititions_per_parameter), n_dimensions)

    return parameters_to_infer, parameter_combinations, sampled_data


def _parameter_combinations_from_flat_indices(parameters_to_infer, flat_indices):
    axes = [np.asarray(axis, dtype=np.float32).ravel() for axis in parameters_to_infer]
    if not axes:
        raise ValueError("parameters_to_infer must contain at least one axis.")
    flat_indices = np.asarray(flat_indices, dtype=np.int64).ravel()
    coordinates = np.unravel_index(flat_indices, tuple(len(axis) for axis in axes))
    return np.stack([axis[np.asarray(coord, dtype=np.int64)] for axis, coord in zip(axes, coordinates)], axis=1)


def _snap_batch_to_support_axes(points, support_axes, margin_bounds=None):
    points = np.asarray(points, dtype=np.float32)
    snapped = np.empty_like(points, dtype=np.float32)

    for dim, axis in enumerate(support_axes):
        axis = np.asarray(axis, dtype=np.float32).ravel()
        if axis.size == 0:
            raise ValueError("Each support axis must contain at least one point.")
        eligible = axis
        if margin_bounds is not None:
            low, high = margin_bounds
            eligible = axis[(axis >= low - 1e-6) & (axis <= high + 1e-6)]
            if eligible.size == 0:
                raise ValueError("No support-axis points remain inside the inference margin.")
        idx = np.abs(points[:, dim, None] - eligible[None, :]).argmin(axis=1)
        snapped[:, dim] = eligible[idx]

    return snapped


@dataclass(frozen=True)
class InferenceDesign:
    mode: str
    parameters_min_max: tuple[float, float]
    config: dict
    parameters_to_infer: tuple[np.ndarray, ...]
    n_repititions_per_parameter: int
    total_points: int
    conceptual_grid_size: int
    margin: float
    seed: int = 0
    support_axes: tuple[np.ndarray, ...] | None = None

    @property
    def n_dimensions(self):
        return int(self.config["data"]["n_parameters"])

    @property
    def grid_shape(self):
        return tuple(int(len(axis)) for axis in self.parameters_to_infer)

    def iter_batches(self, batch_size, generator):
        batch_size = max(1, int(batch_size))
        if self.mode == "grid":
            for start in range(0, self.total_points, batch_size):
                end = min(start + batch_size, self.total_points)
                flat_indices = np.arange(start, end, dtype=np.int64)
                true_params = _parameter_combinations_from_flat_indices(
                    self.parameters_to_infer,
                    flat_indices,
                ).astype(np.float32)
                repeated_parameters = np.repeat(
                    true_params[:, None, :],
                    int(self.n_repititions_per_parameter),
                    axis=1,
                )
                sampled_data = draw_data(
                    repeated_parameters.reshape(-1, self.n_dimensions),
                    self.config,
                    generator,
                ).reshape(-1, int(self.n_repititions_per_parameter), self.n_dimensions)
                yield flat_indices, true_params, sampled_data
            return

        if self.mode != "sobol":
            raise ValueError(f"Unsupported inference design mode: {self.mode!r}.")

        parameter_min, parameter_max = self.parameters_min_max
        parameter_range = parameter_max - parameter_min
        low = parameter_min + parameter_range * self.margin
        high = parameter_max - parameter_range * self.margin
        lows = torch.full((self.n_dimensions,), float(low), dtype=torch.float32)
        widths = torch.full((self.n_dimensions,), float(high - low), dtype=torch.float32)
        sobol = torch.quasirandom.SobolEngine(
            dimension=self.n_dimensions,
            scramble=True,
            seed=int(self.seed),
        )

        done = 0
        while done < self.total_points:
            n_batch = min(batch_size, self.total_points - done)
            flat_indices = np.arange(done, done + n_batch, dtype=np.int64)
            u = sobol.draw(n_batch).to(dtype=torch.float32)
            true_params = (lows[None, :] + u * widths[None, :]).cpu().numpy().astype(np.float32)
            if self.support_axes is not None:
                true_params = _snap_batch_to_support_axes(
                    true_params,
                    self.support_axes,
                    margin_bounds=(low, high),
                )

            repeated_parameters = np.repeat(
                true_params[:, None, :],
                int(self.n_repititions_per_parameter),
                axis=1,
            )
            sampled_data = draw_data(
                repeated_parameters.reshape(-1, self.n_dimensions),
                self.config,
                generator,
            ).reshape(-1, int(self.n_repititions_per_parameter), self.n_dimensions)
            yield flat_indices, true_params, sampled_data
            done += n_batch


def create_inference_design(
    parameters_min_max,
    config,
    generator,
    *,
    design="sobol",
    n_inference_points=65536,
    n_parameters_to_infer_per_dim=5,
    margin=1 / 4,
    data_is_random=False,
    n_repititions_per_parameter=1,
    parameters_to_infer=None,
    support_axes=None,
    seed=0,
):
    n_dimensions = int(config["data"]["n_parameters"])
    design = str(design)
    if design not in {"grid", "sobol"}:
        raise ValueError("design must be one of {'grid', 'sobol'}.")

    if parameters_to_infer is None:
        if support_axes is not None and len(support_axes) != n_dimensions:
            raise ValueError(
                f"Expected support_axes for {n_dimensions} dimensions, got {len(support_axes)}."
            )
        parameters_to_infer = []
        for dim in range(n_dimensions):
            support_axis = None if support_axes is None else support_axes[dim]
            parameters_to_infer.append(
                create_inference_parameters(
                    n_parameters_to_infer_per_dim,
                    parameters_min_max,
                    generator,
                    margin,
                    data_is_random,
                    support_axis=support_axis,
                )
            )

    parameters_to_infer = tuple(np.asarray(axis, dtype=np.float32).ravel() for axis in parameters_to_infer)
    conceptual_grid_size = int(math.prod(len(axis) for axis in parameters_to_infer))
    if conceptual_grid_size <= 0:
        raise ValueError("Inference grid must contain at least one point.")

    if design == "grid":
        total_points = conceptual_grid_size
    else:
        total_points = int(n_inference_points)
        if total_points <= 0:
            raise ValueError("n_inference_points must be positive for Sobol inference.")

    support_axes_tuple = None
    if support_axes is not None:
        support_axes_tuple = tuple(np.asarray(axis, dtype=np.float32).copy() for axis in support_axes)

    return InferenceDesign(
        mode=design,
        parameters_min_max=tuple(float(v) for v in parameters_min_max),
        config=config,
        parameters_to_infer=parameters_to_infer,
        n_repititions_per_parameter=int(n_repititions_per_parameter),
        total_points=int(total_points),
        conceptual_grid_size=int(conceptual_grid_size),
        margin=float(margin),
        seed=int(seed),
        support_axes=support_axes_tuple,
    )


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


def _enumerate_support_axes(support_axes):
    mesh = np.meshgrid(*support_axes, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1).astype(np.float32)


def _draw_lattice_theta_samples(support_axes, n_samples, seed=0, scramble=True):
    support_axes = tuple(np.asarray(axis, dtype=np.float32).ravel() for axis in support_axes)
    if not support_axes:
        raise ValueError("support_axes must contain at least one axis.")

    full_support_size = math.prod(len(axis) for axis in support_axes)
    target = min(int(n_samples), int(full_support_size))
    if target <= 0:
        raise ValueError("n_samples must be positive.")
    if full_support_size <= target:
        return _enumerate_support_axes(support_axes)

    axis_sizes = np.asarray([len(axis) for axis in support_axes], dtype=np.int64)
    sobol = torch.quasirandom.SobolEngine(
        dimension=len(support_axes),
        scramble=bool(scramble),
        seed=int(seed),
    )

    seen = set()
    ordered_indices = []
    while len(ordered_indices) < target:
        batch_size = max(1024, 2 * (target - len(ordered_indices)))
        u = sobol.draw(batch_size).cpu().numpy()
        idx_batch = np.floor(u * axis_sizes[None, :]).astype(np.int64)
        idx_batch = np.clip(idx_batch, 0, axis_sizes[None, :] - 1)
        for row in idx_batch:
            key = tuple(int(v) for v in row)
            if key in seen:
                continue
            seen.add(key)
            ordered_indices.append(key)
            if len(ordered_indices) >= target:
                break
        if len(seen) >= full_support_size:
            break

    theta = np.empty((len(ordered_indices), len(support_axes)), dtype=np.float32)
    for dim, axis in enumerate(support_axes):
        dim_idx = np.asarray([row[dim] for row in ordered_indices], dtype=np.int64)
        theta[:, dim] = axis[dim_idx]
    return theta


def _draw_prior_theta_samples(prior, all_parameters_in_range, n_samples, seed=0, scramble=True, dtype=torch.float32):
    support_axes = getattr(prior, "support_axes", None)
    if getattr(prior, "support_kind", None) == "discrete_grid" and support_axes is not None:
        return _draw_lattice_theta_samples(
            support_axes,
            n_samples=n_samples,
            seed=seed,
            scramble=scramble,
        )
    return _draw_qmc_theta_samples(
        all_parameters_in_range,
        n_samples=n_samples,
        seed=seed,
        scramble=scramble,
        dtype=dtype,
    )


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


def _get_combo_batch_size(n_repititions_per_parameter, eval_batch_size, n_qmc_samples):
    target_theta_chunk = max(1, min(int(n_qmc_samples), 1024))
    combo_batch_size = int(eval_batch_size) // max(1, int(n_repititions_per_parameter) * target_theta_chunk)
    return max(1, combo_batch_size)


def _get_eval_batch_plan(
    n_posterior_combinations,
    n_repititions_per_parameter,
    n_qmc_samples,
    eval_batch_size,
):
    combo_batch_size = _get_combo_batch_size(
        n_repititions_per_parameter,
        eval_batch_size,
        n_qmc_samples,
    )
    max_pairs = max(1, int(eval_batch_size))
    n_theta = int(n_qmc_samples)
    obs_per_combo_batch = max(1, min(int(n_posterior_combinations), combo_batch_size)) * int(
        n_repititions_per_parameter
    )
    target_obs_chunk = max(1, min(obs_per_combo_batch, max_pairs // max(1, min(n_theta, 1024))))
    theta_chunk_size = max(1, min(n_theta, max_pairs // max(1, target_obs_chunk)))
    return {
        "combo_batch_size": combo_batch_size,
        "obs_chunk_size": target_obs_chunk,
        "theta_chunk_size": theta_chunk_size,
        "forward_pairs": int(target_obs_chunk) * int(theta_chunk_size),
    }


def _evaluate_log_ratio_sums(model, sampled_batch, theta_samples_t, eval_batch_size=32768):
    model_device = next(model.parameters()).device
    sampled_batch_t = torch.as_tensor(sampled_batch, dtype=torch.float32, device=model_device)
    n_combos, n_repititions, n_dimensions = sampled_batch_t.shape
    n_theta = int(theta_samples_t.shape[0])

    flat_obs = sampled_batch_t.reshape(n_combos * n_repititions, n_dimensions)
    obs_to_combo = torch.arange(n_combos, device=model_device).repeat_interleave(n_repititions)
    log_ratio_sum = torch.zeros((n_combos, n_theta), device=model_device, dtype=torch.float64)

    max_pairs = max(1, int(eval_batch_size))
    target_obs_chunk = max(1, min(int(flat_obs.shape[0]), max_pairs // max(1, min(n_theta, 1024))))

    with torch.no_grad():
        for obs_start in range(0, int(flat_obs.shape[0]), target_obs_chunk):
            obs_end = min(obs_start + target_obs_chunk, int(flat_obs.shape[0]))
            obs_chunk = flat_obs[obs_start:obs_end]
            combo_idx_chunk = obs_to_combo[obs_start:obs_end]
            theta_chunk_size = max(1, min(n_theta, max_pairs // max(1, int(obs_chunk.shape[0]))))

            for theta_start in range(0, n_theta, theta_chunk_size):
                theta_end = min(theta_start + theta_chunk_size, n_theta)
                theta_chunk = theta_samples_t[theta_start:theta_end]
                theta_count = int(theta_chunk.shape[0])

                data_expand = obs_chunk[:, None, :].expand(int(obs_chunk.shape[0]), theta_count, n_dimensions)
                theta_expand = theta_chunk[None, :, :].expand(int(obs_chunk.shape[0]), theta_count, n_dimensions)
                inputs = torch.cat((data_expand, theta_expand), dim=2).reshape(
                    int(obs_chunk.shape[0]) * theta_count,
                    2 * n_dimensions,
                )
                outputs = model(inputs).reshape(int(obs_chunk.shape[0]), theta_count).clamp_(1e-9, 1 - 1e-9)
                contrib = (torch.log(outputs) - torch.log1p(-outputs)).to(torch.float64)

                combo_accum = torch.zeros((n_combos, theta_count), device=model_device, dtype=torch.float64)
                combo_accum.index_add_(0, combo_idx_chunk, contrib)
                log_ratio_sum[:, theta_start:theta_end] += combo_accum

    return log_ratio_sum


def _arrays_from_log_ratio_sums(theta_samples, log_ratio_sum_np, log_post_sum_np):
    n_rows = int(log_ratio_sum_np.shape[0])
    n_dimensions = int(theta_samples.shape[1])

    posterior_map = np.empty((n_rows, n_dimensions), dtype=np.float32)
    ratio_map = np.empty((n_rows, n_dimensions), dtype=np.float32)
    posterior_hpd_68 = np.empty((n_rows, n_dimensions, 2), dtype=np.float32)
    posterior_hpd_95 = np.empty((n_rows, n_dimensions, 2), dtype=np.float32)
    posterior_hpd_68_combined = np.empty((n_rows, 2), dtype=np.float32)
    posterior_hpd_95_combined = np.empty((n_rows, 2), dtype=np.float32)
    ratio_hpd_68 = np.empty((n_rows, n_dimensions, 2), dtype=np.float32)
    ratio_hpd_95 = np.empty((n_rows, n_dimensions, 2), dtype=np.float32)
    ratio_hpd_68_combined = np.empty((n_rows, 2), dtype=np.float32)
    ratio_hpd_95_combined = np.empty((n_rows, 2), dtype=np.float32)

    for row in range(n_rows):
        post_w = _normalize_log_weights(log_post_sum_np[row])
        ratio_w = _normalize_log_weights(log_ratio_sum_np[row])

        posterior_map[row, :] = theta_samples[int(np.argmax(log_post_sum_np[row]))]
        ratio_map[row, :] = theta_samples[int(np.argmax(log_ratio_sum_np[row]))]

        hpd_68_posterior = _weighted_hpd_from_samples(theta_samples, post_w, alpha=0.32)
        hpd_95_posterior = _weighted_hpd_from_samples(theta_samples, post_w, alpha=0.05)
        posterior_hpd_68[row, :, :] = np.asarray(hpd_68_posterior[0], dtype=np.float32)
        posterior_hpd_68_combined[row, :] = np.asarray(hpd_68_posterior[1], dtype=np.float32)
        posterior_hpd_95[row, :, :] = np.asarray(hpd_95_posterior[0], dtype=np.float32)
        posterior_hpd_95_combined[row, :] = np.asarray(hpd_95_posterior[1], dtype=np.float32)

        hpd_68_ratio = _weighted_hpd_from_samples(theta_samples, ratio_w, alpha=0.32)
        hpd_95_ratio = _weighted_hpd_from_samples(theta_samples, ratio_w, alpha=0.05)
        ratio_hpd_68[row, :, :] = np.asarray(hpd_68_ratio[0], dtype=np.float32)
        ratio_hpd_68_combined[row, :] = np.asarray(hpd_68_ratio[1], dtype=np.float32)
        ratio_hpd_95[row, :, :] = np.asarray(hpd_95_ratio[0], dtype=np.float32)
        ratio_hpd_95_combined[row, :] = np.asarray(hpd_95_ratio[1], dtype=np.float32)

    return {
        "posterior_map": posterior_map,
        "ratio_map": ratio_map,
        "posterior_hpd_68": posterior_hpd_68,
        "posterior_hpd_95": posterior_hpd_95,
        "posterior_hpd_68_combined": posterior_hpd_68_combined,
        "posterior_hpd_95_combined": posterior_hpd_95_combined,
        "ratio_hpd_68": ratio_hpd_68,
        "ratio_hpd_95": ratio_hpd_95,
        "ratio_hpd_68_combined": ratio_hpd_68_combined,
        "ratio_hpd_95_combined": ratio_hpd_95_combined,
    }


class _StreamingPriorSummary:
    def __init__(self, n_dims, parameter_range, curve_bins=64):
        self.n_dims = int(n_dims)
        self.parameter_range = tuple(float(v) for v in parameter_range)
        self.curve_bins = max(1, int(curve_bins))
        self.count = 0
        self._curve_count = np.zeros(self.curve_bins, dtype=np.float64)
        self._curve_bias_sum = np.zeros(self.curve_bins, dtype=np.float64)
        self._curve_width68_sum = np.zeros(self.curve_bins, dtype=np.float64)
        self._curve_width95_sum = np.zeros(self.curve_bins, dtype=np.float64)
        self._hpd = {
            kind: {
                level: {
                    "coverage_sum": np.zeros(self.n_dims, dtype=np.float64),
                    "width_sum": np.zeros(self.n_dims, dtype=np.float64),
                    "width_values": [],
                    "combined_width_values": [],
                }
                for level in ("68", "95")
            }
            for kind in ("posterior", "ratio")
        }

    def _bin_indices(self, values):
        low, high = self.parameter_range
        values = np.asarray(values, dtype=np.float64)
        if high <= low:
            return np.zeros(values.shape, dtype=np.int64)
        scaled = (values - low) / (high - low)
        return np.clip(np.floor(scaled * self.curve_bins).astype(np.int64), 0, self.curve_bins - 1)

    def _update_hpd(self, kind, level, true_params, intervals, combined):
        state = self._hpd[kind][level]
        widths = intervals[:, :, 1] - intervals[:, :, 0]
        contains_true = (true_params >= intervals[:, :, 0]) & (true_params <= intervals[:, :, 1])
        state["coverage_sum"] += np.sum(contains_true, axis=0)
        state["width_sum"] += np.sum(widths, axis=0)
        state["width_values"].append(np.asarray(widths, dtype=np.float32))
        state["combined_width_values"].append(np.asarray(combined[:, 1] - combined[:, 0], dtype=np.float32))

    def update(self, true_params, arrays):
        true_params = np.asarray(true_params, dtype=np.float32)
        n_rows = int(true_params.shape[0])
        if n_rows == 0:
            return

        self.count += n_rows
        self._update_hpd("posterior", "68", true_params, arrays["posterior_hpd_68"], arrays["posterior_hpd_68_combined"])
        self._update_hpd("posterior", "95", true_params, arrays["posterior_hpd_95"], arrays["posterior_hpd_95_combined"])
        self._update_hpd("ratio", "68", true_params, arrays["ratio_hpd_68"], arrays["ratio_hpd_68_combined"])
        self._update_hpd("ratio", "95", true_params, arrays["ratio_hpd_95"], arrays["ratio_hpd_95_combined"])

        ratio_bias = arrays["ratio_map"] - true_params
        width68 = arrays["ratio_hpd_68"][:, :, 1] - arrays["ratio_hpd_68"][:, :, 0]
        width95 = arrays["ratio_hpd_95"][:, :, 1] - arrays["ratio_hpd_95"][:, :, 0]
        bins = self._bin_indices(true_params.reshape(-1))
        np.add.at(self._curve_count, bins, 1.0)
        np.add.at(self._curve_bias_sum, bins, ratio_bias.reshape(-1))
        np.add.at(self._curve_width68_sum, bins, width68.reshape(-1))
        np.add.at(self._curve_width95_sum, bins, width95.reshape(-1))

    def _finalize_hpd_kind(self, kind):
        kind_summary = {}
        denom = max(1, int(self.count))
        for level in ("68", "95"):
            state = self._hpd[kind][level]
            width_values = (
                np.concatenate(state["width_values"], axis=0)
                if state["width_values"]
                else np.empty((0, self.n_dims), dtype=np.float32)
            )
            combined_values = (
                np.concatenate(state["combined_width_values"], axis=0)
                if state["combined_width_values"]
                else np.empty((0,), dtype=np.float32)
            )
            kind_summary[level] = {
                "n_points": int(self.count),
                "coverage_fraction_per_dim": state["coverage_sum"] / denom,
                "coverage_fraction_mean": float(np.sum(state["coverage_sum"]) / max(1, denom * self.n_dims)),
                "mean_interval_width_per_dim": state["width_sum"] / denom,
                "median_interval_width_per_dim": (
                    np.median(width_values, axis=0) if width_values.size else np.zeros(self.n_dims)
                ),
                "mean_interval_width": float(np.sum(state["width_sum"]) / max(1, denom * self.n_dims)),
                "median_interval_width": float(np.median(width_values)) if width_values.size else 0.0,
                "mean_combined_width": float(np.mean(combined_values)) if combined_values.size else 0.0,
                "median_combined_width": float(np.median(combined_values)) if combined_values.size else 0.0,
            }
        return kind_summary

    def finalize(self):
        mask = self._curve_count > 0
        low, high = self.parameter_range
        edges = np.linspace(low, high, self.curve_bins + 1, dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts = np.clip(self._curve_count[mask], 1.0, None)
        x = centers[mask]
        avg_bias = self._curve_bias_sum[mask] / counts
        avg_width_68 = self._curve_width68_sum[mask] / counts
        avg_width_95 = self._curve_width95_sum[mask] / counts
        return {
            "bias_summary": {
                "source": {
                    "map": "ratio",
                    "hpd": "ratio",
                },
                "x": x,
                "avg_bias": avg_bias,
                "avg_width_68": avg_width_68,
                "width_curves": {
                    "68": {
                        "x": x,
                        "avg_width": avg_width_68,
                    },
                    "95": {
                        "x": x,
                        "avg_width": avg_width_95,
                    },
                },
            },
            "hpd_summary": {
                "posterior": self._finalize_hpd_kind("posterior"),
                "ratio": self._finalize_hpd_kind("ratio"),
            },
        }


def _diagnostic_indices(n_points, max_points):
    n_points = int(n_points)
    max_points = int(max_points)
    if n_points <= 0 or max_points <= 0:
        return np.asarray([], dtype=np.int64)
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)
    return np.unique(np.linspace(0, n_points - 1, num=max_points, dtype=np.int64))


def _compact_map_store_from_array(map_array):
    map_array = np.asarray(map_array, dtype=np.float32)
    if map_array.ndim != 2:
        raise ValueError(f"Expected diagnostic map array with 2 dimensions, got {map_array.shape}.")
    return {
        "format": "compact_map_store",
        "n_points": int(map_array.shape[0]),
        "n_dims": int(map_array.shape[1]),
        "dtype": "float32",
        "map": map_array,
    }


def _combined_intervals(intervals):
    intervals = np.asarray(intervals, dtype=np.float32)
    return np.stack(
        [
            np.mean(intervals[:, :, 0], axis=1),
            np.mean(intervals[:, :, 1], axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def _compact_hpd_store_from_arrays(intervals68, intervals95):
    intervals68 = np.asarray(intervals68, dtype=np.float32)
    intervals95 = np.asarray(intervals95, dtype=np.float32)
    if intervals68.ndim != 3 or intervals95.ndim != 3:
        raise ValueError("Expected diagnostic HPD arrays with shape (n_points, n_dims, 2).")
    if intervals68.shape != intervals95.shape:
        raise ValueError("68% and 95% diagnostic HPD arrays must have matching shapes.")
    return {
        "format": "compact_hpd_store",
        "n_points": int(intervals68.shape[0]),
        "n_dims": int(intervals68.shape[1]),
        "dtype": "float32",
        "68": {
            "intervals": intervals68,
            "interval_combined": _combined_intervals(intervals68),
        },
        "95": {
            "intervals": intervals95,
            "interval_combined": _combined_intervals(intervals95),
        },
    }


def get_posteriors_and_errors_summary(
    inference_design,
    all_parameters_in_range,
    models,
    priors,
    device,
    *,
    generator,
    config=None,
    n_qmc_samples=2**16,
    eval_batch_size=32768,
    max_model_evals_per_prior=int(2e15),
    qmc_seed=2026,
    max_gpus=1,
    show_progress=True,
    diagnostic_dir=None,
    raw_diagnostic_sample_points=4096,
    curve_bins=64,
):
    if not isinstance(inference_design, InferenceDesign):
        raise TypeError("inference_design must be an InferenceDesign instance.")
    if generator is None:
        raise ValueError("generator is required for streaming inference data.")

    device = torch.device(device)
    n_posterior_combinations = int(inference_design.total_points)
    n_repititions_per_parameter = int(inference_design.n_repititions_per_parameter)
    n_dimensions = int(inference_design.n_dimensions)

    requested_n_qmc = int(n_qmc_samples)
    safe_n_qmc = max(
        1024,
        min(
            requested_n_qmc,
            int(max_model_evals_per_prior) // max(n_posterior_combinations * n_repititions_per_parameter, 1),
        ),
    )
    if safe_n_qmc < requested_n_qmc:
        print(
            f"[QMC] Reducing n_qmc_samples from {requested_n_qmc} to {safe_n_qmc} "
            f"to respect max_model_evals_per_prior={max_model_evals_per_prior}."
        )

    if max_gpus and int(max_gpus) > 1 and device.type == "cuda":
        print("[posterior] Summary-mode inference currently streams priors on one process.")

    batch_plan = _get_eval_batch_plan(
        n_posterior_combinations,
        n_repititions_per_parameter,
        safe_n_qmc,
        eval_batch_size,
    )
    combo_batch_size = batch_plan["combo_batch_size"]
    print(
        "[posterior-summary] "
        f"design={inference_design.mode}, "
        f"evaluated_points={n_posterior_combinations}, "
        f"conceptual_grid_size={inference_design.conceptual_grid_size}, "
        f"eval_batch_size={int(eval_batch_size)}, "
        f"combo_batch_size={combo_batch_size}, "
        f"obs_chunk_size={batch_plan['obs_chunk_size']}, "
        f"theta_chunk_size={batch_plan['theta_chunk_size']}, "
        f"forward_pairs={batch_plan['forward_pairs']}, "
        f"qmc_samples={safe_n_qmc}"
    )

    diagnostic_dir = None if diagnostic_dir is None else Path(diagnostic_dir).resolve()
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    diag_indices = _diagnostic_indices(n_posterior_combinations, raw_diagnostic_sample_points)
    diag_lookup = {int(value): pos for pos, value in enumerate(diag_indices.tolist())}
    n_diag = int(diag_indices.size)
    parameter_range = inference_design.config["data"]["parameter_range"]
    model_lookup = _get_model_lookup(models)

    runtimes = {}
    for prior in priors:
        prior_name = prior.__name__
        theta_samples = _draw_prior_theta_samples(
            prior,
            all_parameters_in_range,
            n_samples=safe_n_qmc,
            seed=qmc_seed,
            scramble=True,
            dtype=torch.float32,
        )
        theta_samples_t = torch.as_tensor(theta_samples, dtype=torch.float32, device=device)
        prior_vals = np.asarray(prior(theta_samples), dtype=np.float64)
        prior_vals = np.clip(prior_vals, 1e-300, None)
        log_prior_t = torch.as_tensor(np.log(prior_vals), dtype=torch.float64, device=device)[None, :]
        raw = {
            "sample_index": diag_indices.astype(np.int64),
            "true_params": np.empty((n_diag, n_dimensions), dtype=np.float32),
            "posterior_map": np.empty((n_diag, n_dimensions), dtype=np.float32),
            "ratio_map": np.empty((n_diag, n_dimensions), dtype=np.float32),
            "posterior_hpd_68": np.empty((n_diag, n_dimensions, 2), dtype=np.float32),
            "posterior_hpd_95": np.empty((n_diag, n_dimensions, 2), dtype=np.float32),
            "ratio_hpd_68": np.empty((n_diag, n_dimensions, 2), dtype=np.float32),
            "ratio_hpd_95": np.empty((n_diag, n_dimensions, 2), dtype=np.float32),
        }
        runtimes[prior_name] = {
            "prior": prior,
            "model": model_lookup[prior_name].to(device).eval(),
            "theta_samples": theta_samples,
            "theta_samples_t": theta_samples_t,
            "log_prior_t": log_prior_t,
            "summary": _StreamingPriorSummary(
                n_dimensions,
                parameter_range,
                curve_bins=curve_bins,
            ),
            "raw": raw,
            "diagnostic_path": None,
        }

    total_iterations = n_posterior_combinations * n_repititions_per_parameter * len(priors)
    progress = tqdm(
        total=total_iterations,
        desc="Calculating posteriors and errors (streaming summary)...",
        disable=not show_progress,
    )

    with progress as pbar:
        for flat_indices, true_params, sampled_batch in inference_design.iter_batches(combo_batch_size, generator):
            diag_mask = np.isin(flat_indices, diag_indices)
            diag_rows = np.nonzero(diag_mask)[0]
            diag_positions = [diag_lookup[int(value)] for value in flat_indices[diag_mask]]

            for prior in priors:
                prior_name = prior.__name__
                runtime = runtimes[prior_name]
                log_ratio_sum_batch = _evaluate_log_ratio_sums(
                    runtime["model"],
                    sampled_batch,
                    runtime["theta_samples_t"],
                    eval_batch_size=eval_batch_size,
                )
                log_post_sum_batch = log_ratio_sum_batch + runtime["log_prior_t"]
                arrays = _arrays_from_log_ratio_sums(
                    runtime["theta_samples"],
                    log_ratio_sum_batch.detach().cpu().numpy(),
                    log_post_sum_batch.detach().cpu().numpy(),
                )
                runtime["summary"].update(true_params, arrays)

                if diag_rows.size:
                    raw = runtime["raw"]
                    raw["true_params"][diag_positions, :] = true_params[diag_rows, :]
                    for key in (
                        "posterior_map",
                        "ratio_map",
                        "posterior_hpd_68",
                        "posterior_hpd_95",
                        "ratio_hpd_68",
                        "ratio_hpd_95",
                    ):
                        raw[key][diag_positions, ...] = arrays[key][diag_rows, ...]

                pbar.update(int(true_params.shape[0]) * n_repititions_per_parameter)

    all_posteriors = {}
    all_ratios = {}
    all_hpds_posterior = {}
    all_hpds_ratio = {}
    bias_summary = {}
    hpd_summary = {"posterior": {}, "ratio": {}}
    raw_diagnostic_files = {}

    for prior_name, runtime in runtimes.items():
        finalized = runtime["summary"].finalize()
        bias_summary[prior_name] = finalized["bias_summary"]
        hpd_summary["posterior"][prior_name] = finalized["hpd_summary"]["posterior"]
        hpd_summary["ratio"][prior_name] = finalized["hpd_summary"]["ratio"]

        raw = runtime["raw"]
        if diagnostic_dir is not None:
            diagnostic_path = diagnostic_dir / f"{prior_name}_raw_diagnostic_sample.npz"
            np.savez_compressed(diagnostic_path, **raw)
            runtime["diagnostic_path"] = diagnostic_path
            raw_diagnostic_files[prior_name] = str(diagnostic_path)
        else:
            raw_diagnostic_files[prior_name] = None

        all_posteriors[prior_name] = _compact_map_store_from_array(raw["posterior_map"])
        all_ratios[prior_name] = _compact_map_store_from_array(raw["ratio_map"])
        all_hpds_posterior[prior_name] = _compact_hpd_store_from_arrays(
            raw["posterior_hpd_68"],
            raw["posterior_hpd_95"],
        )
        all_hpds_ratio[prior_name] = _compact_hpd_store_from_arrays(
            raw["ratio_hpd_68"],
            raw["ratio_hpd_95"],
        )
        runtime["model"].to("cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    diagnostic_true_params = (
        next(iter(runtimes.values()))["raw"]["true_params"].copy()
        if runtimes
        else np.empty((0, n_dimensions), dtype=np.float32)
    )
    diagnostic_sampled_data = np.empty(
        (diagnostic_true_params.shape[0], n_repititions_per_parameter, n_dimensions),
        dtype=np.float32,
    )
    diagnostic_inference_data = (
        [axis.copy() for axis in inference_design.parameters_to_infer],
        diagnostic_true_params,
        diagnostic_sampled_data,
    )

    return {
        "all_posteriors": all_posteriors,
        "all_ratios": all_ratios,
        "all_hpds_posterior": all_hpds_posterior,
        "all_hpds_ratio": all_hpds_ratio,
        "diagnostic_inference_data": diagnostic_inference_data,
        "bias_summary": bias_summary,
        "hpd_summary": hpd_summary,
        "raw_diagnostic_files": raw_diagnostic_files,
        "diagnostic_sample_indices": diag_indices,
        "safe_n_qmc": int(safe_n_qmc),
        "batch_plan": batch_plan,
    }


def _compute_prior_results_qmc(
    *,
    prior_name,
    prior,
    model,
    sampled_data,
    all_parameters_in_range,
    device,
    n_qmc_samples,
    eval_batch_size,
    qmc_seed,
    progress=None,
    progress_queue=None,
    storage_dir=None,
    flush_every_batches=32,
):
    sampled_data = np.asarray(sampled_data, dtype=np.float32)
    n_posterior_combinations, n_repititions_per_parameter, n_dimensions = sampled_data.shape

    theta_samples = _draw_prior_theta_samples(
        prior,
        all_parameters_in_range,
        n_samples=int(n_qmc_samples),
        seed=qmc_seed,
        scramble=True,
        dtype=torch.float32,
    )
    theta_samples_t = torch.as_tensor(theta_samples, dtype=torch.float32, device=device)
    prior_vals = np.asarray(prior(theta_samples), dtype=np.float64)
    prior_vals = np.clip(prior_vals, 1e-300, None)
    log_prior_t = torch.as_tensor(np.log(prior_vals), dtype=torch.float64, device=device)[None, :]

    posteriors_grouped, posterior_map_array = _create_compact_map_store(
        n_posterior_combinations,
        n_dimensions,
        storage_dir=storage_dir,
        basename="posterior_map",
    )
    ratios_grouped, ratio_map_array = _create_compact_map_store(
        n_posterior_combinations,
        n_dimensions,
        storage_dir=storage_dir,
        basename="ratio_map",
    )
    errors_posterior, errors_posterior_arrays = _create_compact_hpd_store(
        n_posterior_combinations,
        n_dimensions,
        storage_dir=storage_dir,
        basename="posterior_hpd",
    )
    errors_ratio, errors_ratio_arrays = _create_compact_hpd_store(
        n_posterior_combinations,
        n_dimensions,
        storage_dir=storage_dir,
        basename="ratio_hpd",
    )
    flushables = [
        posterior_map_array,
        ratio_map_array,
        errors_posterior_arrays["68"]["intervals"],
        errors_posterior_arrays["68"]["interval_combined"],
        errors_posterior_arrays["95"]["intervals"],
        errors_posterior_arrays["95"]["interval_combined"],
        errors_ratio_arrays["68"]["intervals"],
        errors_ratio_arrays["68"]["interval_combined"],
        errors_ratio_arrays["95"]["intervals"],
        errors_ratio_arrays["95"]["interval_combined"],
    ]
    combo_batch_size = _get_combo_batch_size(n_repititions_per_parameter, eval_batch_size, n_qmc_samples)

    model = model.to(device).eval()
    for batch_idx, combo_start in enumerate(range(0, n_posterior_combinations, combo_batch_size)):
        combo_end = min(combo_start + combo_batch_size, n_posterior_combinations)
        sampled_batch = sampled_data[combo_start:combo_end]
        log_ratio_sum_batch = _evaluate_log_ratio_sums(
            model,
            sampled_batch,
            theta_samples_t,
            eval_batch_size=eval_batch_size,
        )
        # For a shared parameter inferred from repeated observations, the prior
        # enters the joint posterior once:
        #   p(theta | x_1, ..., x_n) propto p(theta) * prod_i r(x_i, theta)
        log_post_sum_batch = log_ratio_sum_batch + log_prior_t

        log_ratio_sum_np = log_ratio_sum_batch.detach().cpu().numpy()
        log_post_sum_np = log_post_sum_batch.detach().cpu().numpy()

        for row in range(combo_end - combo_start):
            row_idx = combo_start + row
            post_w = _normalize_log_weights(log_post_sum_np[row])
            ratio_w = _normalize_log_weights(log_ratio_sum_np[row])

            posterior_map_array[row_idx, :] = theta_samples[int(np.argmax(log_post_sum_np[row]))]
            ratio_map_array[row_idx, :] = theta_samples[int(np.argmax(log_ratio_sum_np[row]))]

            hpd_68_posterior = _weighted_hpd_from_samples(theta_samples, post_w, alpha=0.32)
            hpd_95_posterior = _weighted_hpd_from_samples(theta_samples, post_w, alpha=0.05)
            errors_posterior_arrays["68"]["intervals"][row_idx, :, :] = np.asarray(
                hpd_68_posterior[0],
                dtype=np.float32,
            )
            errors_posterior_arrays["68"]["interval_combined"][row_idx, :] = np.asarray(
                hpd_68_posterior[1],
                dtype=np.float32,
            )
            errors_posterior_arrays["95"]["intervals"][row_idx, :, :] = np.asarray(
                hpd_95_posterior[0],
                dtype=np.float32,
            )
            errors_posterior_arrays["95"]["interval_combined"][row_idx, :] = np.asarray(
                hpd_95_posterior[1],
                dtype=np.float32,
            )

            hpd_68_ratio = _weighted_hpd_from_samples(theta_samples, ratio_w, alpha=0.32)
            hpd_95_ratio = _weighted_hpd_from_samples(theta_samples, ratio_w, alpha=0.05)
            errors_ratio_arrays["68"]["intervals"][row_idx, :, :] = np.asarray(
                hpd_68_ratio[0],
                dtype=np.float32,
            )
            errors_ratio_arrays["68"]["interval_combined"][row_idx, :] = np.asarray(
                hpd_68_ratio[1],
                dtype=np.float32,
            )
            errors_ratio_arrays["95"]["intervals"][row_idx, :, :] = np.asarray(
                hpd_95_ratio[0],
                dtype=np.float32,
            )
            errors_ratio_arrays["95"]["interval_combined"][row_idx, :] = np.asarray(
                hpd_95_ratio[1],
                dtype=np.float32,
            )

        if progress is not None:
            progress.update((combo_end - combo_start) * n_repititions_per_parameter)
        if progress_queue is not None:
            progress_queue.put(
                {
                    "prior_name": prior_name,
                    "delta": (combo_end - combo_start) * n_repititions_per_parameter,
                    "completed_combos": combo_end,
                    "total_combos": n_posterior_combinations,
                }
            )
        if storage_dir is not None and flush_every_batches > 0 and (batch_idx + 1) % int(flush_every_batches) == 0:
            _flush_arrays(flushables)

    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if storage_dir is not None:
        _flush_arrays(flushables)

    return prior_name, posteriors_grouped, ratios_grouped, errors_posterior, errors_ratio


def _posterior_prior_worker(task):
    from .priors import build_priors

    config = task["config"]
    device = torch.device(task["device"])
    model = _build_model_from_state(config, task["model_state"], device)
    priors, _, _, _ = build_priors(config, np.random.default_rng(0))
    prior_lookup = {prior.__name__: prior for prior in priors}

    return _compute_prior_results_qmc(
        prior_name=task["prior_name"],
        prior=prior_lookup[task["prior_name"]],
        model=model,
        sampled_data=task["sampled_data"],
        all_parameters_in_range=task["all_parameters_in_range"],
        device=device,
        n_qmc_samples=task["n_qmc_samples"],
        eval_batch_size=task["eval_batch_size"],
        qmc_seed=task["qmc_seed"],
        progress=None,
        progress_queue=task.get("progress_queue"),
        storage_dir=task.get("storage_dir"),
    )


def _drain_worker_progress_queue(progress_queue, progress, log_state, log_every_fraction=0.05):
    if progress_queue is None:
        return

    while True:
        try:
            update = progress_queue.get_nowait()
        except queue_module.Empty:
            break

        progress.update(int(update["delta"]))

        prior_name = update["prior_name"]
        completed_combos = int(update["completed_combos"])
        total_combos = max(1, int(update["total_combos"]))
        fraction_done = completed_combos / total_combos

        next_fraction = log_state.setdefault(prior_name, log_every_fraction)
        if fraction_done >= next_fraction or completed_combos == total_combos:
            tqdm.write(
                f"[posterior:{prior_name}] "
                f"{completed_combos}/{total_combos} combinations "
                f"({fraction_done:.1%})"
            )
            while fraction_done >= next_fraction:
                next_fraction += log_every_fraction
            log_state[prior_name] = next_fraction


def get_posteriors_and_errors_qmc(
    inference_data,
    all_parameters_in_range,
    models,
    priors,
    device,
    config=None,
    n_qmc_samples=2**16,
    eval_batch_size=32768,
    max_model_evals_per_prior=int(2e15),
    qmc_seed=2026,
    max_gpus=1,
    show_progress=True,
    storage_dir=None,
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
    if storage_dir is not None:
        storage_dir = Path(storage_dir).resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)
        print(f"[posterior] Using disk-backed posterior storage under {storage_dir}")
    batch_plan = _get_eval_batch_plan(
        n_posterior_combinations,
        n_repititions_per_parameter,
        safe_n_qmc,
        eval_batch_size,
    )
    print(
        "[posterior] "
        f"eval_batch_size={int(eval_batch_size)}, "
        f"combo_batch_size={batch_plan['combo_batch_size']}, "
        f"obs_chunk_size={batch_plan['obs_chunk_size']}, "
        f"theta_chunk_size={batch_plan['theta_chunk_size']}, "
        f"forward_pairs={batch_plan['forward_pairs']}, "
        f"qmc_samples={safe_n_qmc}"
    )

    model_lookup = _get_model_lookup(models)
    n_cuda_workers = _select_num_cuda_workers(device, max_gpus, len(priors))

    if n_cuda_workers <= 1:
        total_iterations = n_posterior_combinations * n_repititions_per_parameter * len(priors)
        progress = tqdm(
            total=total_iterations,
            desc="Calculating posteriors and errors (QMC samples)...",
            disable=not show_progress,
        )

        with progress as pbar:
            for prior in priors:
                prior_name, posteriors_grouped, ratios_grouped, errors_posterior, errors_ratio = (
                    _compute_prior_results_qmc(
                        prior_name=prior.__name__,
                        prior=prior,
                        model=model_lookup[prior.__name__],
                        sampled_data=sampled_data,
                        all_parameters_in_range=all_parameters_in_range,
                        device=torch.device(device),
                        n_qmc_samples=safe_n_qmc,
                        eval_batch_size=eval_batch_size,
                        qmc_seed=qmc_seed,
                        progress=pbar,
                        storage_dir=None if storage_dir is None else storage_dir / prior.__name__,
                    )
                )

                all_hpds_posterior[prior_name] = errors_posterior
                all_hpds_ratio[prior_name] = errors_ratio
                all_posteriors[prior_name] = posteriors_grouped
                all_ratios[prior_name] = ratios_grouped
    else:
        if config is None:
            raise ValueError("config is required when max_gpus > 1 for posterior parallelism.")

        device_names = [f"cuda:{idx}" for idx in range(n_cuda_workers)]
        total_iterations = n_posterior_combinations * n_repititions_per_parameter * len(priors)
        progress = tqdm(
            total=total_iterations,
            desc="Calculating posteriors and errors (QMC workers)...",
            disable=not show_progress,
        )
        ctx = mp.get_context("spawn")
        sampled_data_for_workers = np.asarray(sampled_data, dtype=np.float32)
        manager = mp.Manager()
        progress_queue = manager.Queue()
        progress_log_state = {}
        try:
            tasks = []
            for idx, prior in enumerate(priors):
                tasks.append(
                    {
                        "prior_name": prior.__name__,
                        "model_state": _state_dict_to_cpu(model_lookup[prior.__name__]),
                        "config": config,
                        "sampled_data": sampled_data_for_workers,
                        "all_parameters_in_range": all_parameters_in_range,
                        "n_qmc_samples": safe_n_qmc,
                        "eval_batch_size": eval_batch_size,
                        "qmc_seed": qmc_seed,
                        "device": device_names[idx % n_cuda_workers],
                        "progress_queue": progress_queue,
                        "storage_dir": None if storage_dir is None else storage_dir / prior.__name__,
                    }
                )

            with progress, concurrent.futures.ProcessPoolExecutor(
                max_workers=n_cuda_workers,
                mp_context=ctx,
            ) as executor:
                future_map = {executor.submit(_posterior_prior_worker, task): task["prior_name"] for task in tasks}
                pending = set(future_map)

                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=1.0,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    _drain_worker_progress_queue(progress_queue, progress, progress_log_state)

                    for future in done:
                        prior_name, posteriors_grouped, ratios_grouped, errors_posterior, errors_ratio = (
                            future.result()
                        )
                        all_hpds_posterior[prior_name] = errors_posterior
                        all_hpds_ratio[prior_name] = errors_ratio
                        all_posteriors[prior_name] = posteriors_grouped
                        all_ratios[prior_name] = ratios_grouped

                _drain_worker_progress_queue(progress_queue, progress, progress_log_state)
        finally:
            manager.shutdown()

    return all_posteriors, all_ratios, all_hpds_posterior, all_hpds_ratio


def get_posteriors_and_errors(
    inference_data,
    all_parameters_in_range,
    models,
    priors,
    device,
    config=None,
    n_qmc_samples=2**16,
    eval_batch_size=32768,
    max_model_evals_per_prior=int(2e15),
    qmc_seed=2026,
    max_gpus=1,
    show_progress=True,
    storage_dir=None,
):
    return get_posteriors_and_errors_qmc(
        inference_data,
        all_parameters_in_range,
        models=models,
        priors=priors,
        device=device,
        config=config,
        n_qmc_samples=n_qmc_samples,
        eval_batch_size=eval_batch_size,
        max_model_evals_per_prior=max_model_evals_per_prior,
        qmc_seed=qmc_seed,
        max_gpus=max_gpus,
        show_progress=show_progress,
        storage_dir=storage_dir,
    )


def get_first_column_scatter_data(inference_data, all_posteriors, parameters_post, hpds):
    _, parameter_combinations, _ = inference_data
    true_params = np.asarray(parameter_combinations, dtype=np.float32)
    n_dims = len(parameters_post)

    if true_params.ndim != 2 or true_params.shape[1] != n_dims:
        raise ValueError(
            f"Expected true parameters with shape (N, {n_dims}), got {true_params.shape}."
        )

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

        y_hat = _extract_map_array(
            all_posteriors[prior_name],
            parameters_post=parameters_post,
            n_dims=n_dims,
        )
        intervals_68, _ = _extract_hpd_level(errors, level="68", n_dims=n_dims)

        n_points = int(intervals_68.shape[0])
        if n_points != int(y_hat.shape[0]):
            raise ValueError(
                f"For prior '{prior_name}', HPD count ({n_points}) does not match "
                f"posterior count ({y_hat.shape[0]})."
            )
        if n_points != true_params.shape[0]:
            raise ValueError(
                f"For prior '{prior_name}', HPD count ({n_points}) does not match "
                f"true points ({true_params.shape[0]})."
            )

        width_dims = intervals_68[:, :, 1] - intervals_68[:, :, 0]
        bias_dims = y_hat - true_params

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
