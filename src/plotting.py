from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def plot_prior_contours(priors, parameter_range, n_parameters, n_points=200, fixed_value=None):
    theta = np.linspace(parameter_range[0], parameter_range[1], n_points)
    x, y = np.meshgrid(theta, theta, indexing="ij")
    grid = np.full(x.shape + (n_parameters,), fixed_value, dtype=float)
    if fixed_value is None:
        grid.fill(0.5 * (parameter_range[0] + parameter_range[1]))
    grid[..., 0] = x
    grid[..., 1] = y

    n_priors = len(priors)
    ncols = min(2, n_priors)
    nrows = (n_priors + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    for i, prior in enumerate(priors):
        z = prior(grid)
        cf = axes[i].contourf(x, y, z, levels=30, cmap="viridis")
        axes[i].contour(x, y, z, levels=10, colors="k", linewidths=0.5, alpha=0.4)
        axes[i].set_title(prior.__name__)
        axes[i].set_xlabel(r"$\theta_0$")
        axes[i].set_ylabel(r"$\theta_1$")
        axes[i].grid(alpha=0.3)
        plt.colorbar(cf, ax=axes[i], fraction=0.046, pad=0.04)

    for k in range(n_priors, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    return fig, axes


def plot_prior_sample_histograms(prior_samplers, config, n_bins=100, factor_n=1_000):
    n_priors = len(prior_samplers)
    ncols = min(2, n_priors)
    nrows = (n_priors + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    param_min, param_max = config["data"]["parameter_range"]
    n_samples = factor_n * config["data"]["n_train"]

    for i, prior_sampler in enumerate(prior_samplers):
        samples = prior_sampler((n_samples,))
        axes[i].hist2d(
            samples[:, 0],
            samples[:, 1],
            bins=n_bins,
            range=[[param_min, param_max], [param_min, param_max]],
            cmap="viridis",
        )
        axes[i].set_title(f"Samples from {prior_sampler.__name__} prior")
        axes[i].set_xlabel(r"$\theta_0$")
        axes[i].set_ylabel(r"$\theta_1$")
        axes[i].grid(alpha=0.3)

    for k in range(n_priors, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    return fig, axes


def plot_training_tests(training_results, test_results, prior_name, filename=None):
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax[0].plot(training_results["train_loss"], label="Train Loss")
    ax[0].plot(training_results["val_loss"], label="Validation Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_ylim(0.2, 0.7)
    ax[0].set_title(f"Training Results for {prior_name} prior")
    ax[0].legend()

    ax[1].plot(training_results["val_accuracy"], label="Validation Accuracy")
    ax[1].plot(training_results["val_auc"], label="Validation AUC")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Metric")
    ax[1].set_ylim(0.5, 1)
    ax[1].set_title(f"Training Results for {prior_name} prior")
    ax[1].legend()

    test_loss, test_accuracy, test_auc, test_roc = test_results
    roc_fpr, roc_tpr, _ = test_roc
    ax[2].plot(roc_fpr, roc_tpr, label=f"AUC = {test_auc:.4f}")
    ax[2].plot([0, 1], [0, 1], "k--", label="Random Guess")
    ax[2].set_xlabel("False Positive Rate")
    ax[2].set_ylabel("True Positive Rate")
    ax[2].set_title(f"Test ROC Curve for {prior_name} prior")
    ax[2].legend()
    ax[2].text(
        0.66,
        0.22,
        f"Test Loss: {test_loss:.3f}\nTest Acc.: {test_accuracy:.3f}",
        transform=ax[2].transAxes,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="none"),
    )

    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    return fig, ax


def plot_all_rocs(test_results, filename=None):
    def _iter_results(tr):
        if isinstance(tr, dict):
            for key, value in tr.items():
                yield key, value
        else:
            for result_dict in tr:
                for key, value in result_dict.items():
                    yield key, value

    curves = []
    for name, res in _iter_results(test_results):
        if res is None or len(res) < 4:
            continue
        try:
            auc = float(res[2])
            fpr, tpr, _ = res[3]
            curves.append((auc, fpr, tpr, name))
        except Exception:
            continue

    if not curves:
        print("No valid ROC data found in test_results.")
        return None

    curves.sort(key=lambda x: x[0], reverse=True)

    fig = plt.figure(figsize=(8, 6))
    for auc, fpr, tpr, name in curves:
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.5, alpha=0.9)

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curves across prior combinations")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)
    return fig


def plot_errorbars(
    hpds: dict,
    inference_data,
    all_posteriors,
    parameters_post,
    show_annotations=True,
    filename=None,
):
    _, parameter_combinations, _ = inference_data
    true_params = np.asarray(parameter_combinations, dtype=float)

    n_dims = len(parameters_post)
    n_posteriors = len(hpds)

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(
        nrows=n_posteriors,
        ncols=n_dims,
        figsize=(5 * n_dims, 3.8 * n_posteriors),
        squeeze=False,
    )

    dim_lengths = [len(r) for r in parameters_post]

    def _get_bounds(interval, y_hat):
        try:
            arr = np.asarray(interval, dtype=float)
        except Exception:
            return y_hat, y_hat
        if arr.ndim == 1 and arr.size == 2:
            return float(arr[0]), float(arr[1])
        if arr.ndim == 2 and arr.shape[1] == 2:
            mask = (arr[:, 0] <= y_hat) & (y_hat <= arr[:, 1])
            if np.any(mask):
                row = arr[np.where(mask)[0][0]]
            else:
                mids = 0.5 * (arr[:, 0] + arr[:, 1])
                row = arr[np.argmin(np.abs(mids - y_hat))]
            return float(row[0]), float(row[1])
        return y_hat, y_hat

    def _aggregate(values_x, values_y, lower68, upper68, lower95, upper95, widths68):
        keys = np.round(np.asarray(values_x, dtype=float), decimals=12)
        uniq, inv = np.unique(keys, return_inverse=True)
        grouped = {
            "x": uniq,
            "y": np.zeros(len(uniq), dtype=float),
            "l68": np.zeros(len(uniq), dtype=float),
            "u68": np.zeros(len(uniq), dtype=float),
            "l95": np.zeros(len(uniq), dtype=float),
            "u95": np.zeros(len(uniq), dtype=float),
            "w68": np.zeros(len(uniq), dtype=float),
        }
        for i in range(len(uniq)):
            mask = inv == i
            grouped["y"][i] = np.mean(np.asarray(values_y)[mask])
            grouped["l68"][i] = np.mean(np.asarray(lower68)[mask])
            grouped["u68"][i] = np.mean(np.asarray(upper68)[mask])
            grouped["l95"][i] = np.mean(np.asarray(lower95)[mask])
            grouped["u95"][i] = np.mean(np.asarray(upper95)[mask])
            grouped["w68"][i] = np.mean(np.asarray(widths68)[mask])
        return grouped

    for i, (prior_name, errors) in enumerate(hpds.items()):
        posts = all_posteriors[prior_name]
        n_points = len(posts)
        y_hat = np.zeros((n_points, n_dims), dtype=float)

        for idx, posterior in enumerate(posts):
            if isinstance(posterior, dict) and "map" in posterior:
                y_hat[idx, :] = np.asarray(posterior["map"], dtype=float)
                continue

            arr = np.asarray(posterior)
            if arr.ndim == 1:
                imax = int(np.argmax(arr))
                multi = np.unravel_index(imax, dim_lengths)
            else:
                multi = np.unravel_index(int(np.argmax(arr)), arr.shape)
            for d in range(n_dims):
                y_hat[idx, d] = parameters_post[d][multi[d]]

        for d in range(n_dims):
            x_true = true_params[:, d]
            y_est = y_hat[:, d]

            yerr68_lower, yerr68_upper = [], []
            yerr95_lower, yerr95_upper = [], []
            widths68 = []

            for idx, entry in enumerate(errors):
                y_estimate = y_est[idx]
                low68, high68 = _get_bounds(entry["68"][d], y_estimate)
                low95, high95 = _get_bounds(entry["95"][d], y_estimate)
                yerr68_lower.append(abs(y_estimate - low68))
                yerr68_upper.append(abs(high68 - y_estimate))
                yerr95_lower.append(abs(y_estimate - low95))
                yerr95_upper.append(abs(high95 - y_estimate))
                widths68.append(high68 - low68)

            grouped = _aggregate(
                x_true,
                y_est,
                yerr68_lower,
                yerr68_upper,
                yerr95_lower,
                yerr95_upper,
                widths68,
            )

            a = ax[i, d]
            yerr95 = np.vstack([grouped["l95"], grouped["u95"]])
            yerr68 = np.vstack([grouped["l68"], grouped["u68"]])
            a.errorbar(
                grouped["x"],
                grouped["y"],
                yerr=yerr95,
                fmt="o",
                color="C0",
                ecolor="C0",
                alpha=0.35,
                capsize=5,
                label="95% HPD",
            )
            a.errorbar(
                grouped["x"],
                grouped["y"],
                yerr=yerr68,
                fmt="o",
                color="C0",
                ecolor="C0",
                capsize=5,
                label="68% HPD",
            )

            abs_errors = grouped["y"] - grouped["x"]
            if show_annotations:
                for x_val, y_val, width, err_abs in zip(
                    grouped["x"],
                    grouped["y"],
                    grouped["w68"],
                    abs_errors,
                ):
                    a.annotate(
                        rf"$\delta$={err_abs:.2f}" + "\n" + rf"$\Delta$={width:.2f}",
                        xy=(x_val, y_val),
                        xytext=(6, -22),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=10,
                        color="C0",
                    )

            tmin, tmax = parameters_post[d][0], parameters_post[d][-1]
            a.plot([tmin, tmax], [tmin, tmax], "k--", label="Ideal")
            a.set_title(f"{prior_name} prior (θ_{d})")
            a.set_xlabel(f"True $\\theta_{d}$")
            a.set_ylabel(f"Estimated $\\theta_{d}$")
            a.grid()
            a.legend()

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    return fig, ax


def plot_error_and_hpd_width(hpds: dict, inference_data, all_posteriors, parameters_post, filename=None):
    _, parameter_combinations, _ = inference_data
    true_params = np.asarray(parameter_combinations, dtype=float)

    n_dims = len(parameters_post)
    n_posteriors = len(hpds)

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(
        nrows=n_posteriors,
        ncols=3,
        figsize=(19, 3.8 * n_posteriors),
        squeeze=False,
    )

    dim_lengths = [len(r) for r in parameters_post]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    blues = plt.cm.Blues(np.linspace(0.45, 0.9, max(n_dims, 2)))
    reds = plt.cm.Reds(np.linspace(0.45, 0.9, max(n_dims, 2)))

    def _parse_hpd_entry(entry, level="68"):
        raw = entry[level]

        intervals = None
        interval_combined = None

        if isinstance(raw, dict):
            intervals = raw.get("intervals", raw.get("interval", None))
            interval_combined = raw.get("interval_combined", raw.get("combined", None))
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            intervals, interval_combined = raw[0], raw[1]
        else:
            intervals = raw

        intervals = np.asarray(intervals, dtype=float)
        if interval_combined is None:
            if intervals.ndim == 2 and intervals.shape[1] == 2:
                low = float(np.mean(intervals[:, 0]))
                high = float(np.mean(intervals[:, 1]))
                interval_combined = np.array([low, high], dtype=float)
            else:
                interval_combined = np.array([np.nan, np.nan], dtype=float)
        else:
            interval_combined = np.asarray(interval_combined, dtype=float)

        return intervals, interval_combined

    def _aggregate_curve_by_x(x, y, decimals=12):
        x_key = np.round(np.asarray(x, dtype=float), decimals=decimals)
        y = np.asarray(y, dtype=float)
        x_unique, inv = np.unique(x_key, return_inverse=True)
        y_avg = np.zeros(len(x_unique), dtype=float)
        for i in range(len(x_unique)):
            y_avg[i] = np.mean(y[inv == i])
        return x_unique, y_avg

    def _aggregate_per_dim_by_true_value(true_params_local, width_dims_local, bias_dims_local):
        width_curves = []
        bias_curves = []
        for d in range(n_dims):
            x_d = true_params_local[:, d]
            xw, yw = _aggregate_curve_by_x(x_d, width_dims_local[:, d])
            xb, yb = _aggregate_curve_by_x(x_d, bias_dims_local[:, d])
            width_curves.append((xw, yw))
            bias_curves.append((xb, yb))
        return width_curves, bias_curves

    def _aggregate_combined_from_dim_curves(curves_by_dim):
        x_all, y_all = [], []
        for x_d, y_d in curves_by_dim:
            x_all.extend(x_d.tolist())
            y_all.extend(y_d.tolist())
        if len(x_all) == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        return _aggregate_curve_by_x(np.asarray(x_all), np.asarray(y_all))

    panel_data = {}
    all_bias_comb, all_width_comb = [], []
    all_width_dims, all_bias_dims = [], []

    for prior_name, errors in hpds.items():
        posts = all_posteriors[prior_name]
        n_points = len(posts)

        y_hat = np.zeros((n_points, n_dims), dtype=float)
        for idx, posterior in enumerate(posts):
            if isinstance(posterior, dict) and "map" in posterior:
                y_hat[idx, :] = np.asarray(posterior["map"], dtype=float)
                continue

            arr = np.asarray(posterior)
            if arr.ndim == 1:
                imax = int(np.argmax(arr))
                multi = np.unravel_index(imax, dim_lengths)
            else:
                multi = np.unravel_index(int(np.argmax(arr)), arr.shape)
            for d in range(n_dims):
                y_hat[idx, d] = parameters_post[d][multi[d]]

        width_dims = np.zeros((n_points, n_dims), dtype=float)
        bias_dims = np.zeros((n_points, n_dims), dtype=float)

        for i_e, entry in enumerate(errors):
            intervals, _ = _parse_hpd_entry(entry, level="68")
            for d in range(n_dims):
                low, high = float(intervals[d, 0]), float(intervals[d, 1])
                width_dims[i_e, d] = high - low
                bias_dims[i_e, d] = 0.5 * (low + high) - true_params[i_e, d]

        width_curves, bias_curves = _aggregate_per_dim_by_true_value(
            true_params,
            width_dims,
            bias_dims,
        )

        x_comb, bias_comb = _aggregate_combined_from_dim_curves(bias_curves)
        _, width_comb = _aggregate_combined_from_dim_curves(width_curves)

        panel_data[prior_name] = {
            "x_comb": x_comb,
            "bias_comb": bias_comb,
            "width_comb": width_comb,
            "width_curves": width_curves,
            "bias_curves": bias_curves,
        }

        if len(bias_comb):
            all_bias_comb.extend(bias_comb.tolist())
        if len(width_comb):
            all_width_comb.extend(width_comb.tolist())
        for _, y_d in width_curves:
            if len(y_d):
                all_width_dims.extend(y_d.tolist())
        for _, y_d in bias_curves:
            if len(y_d):
                all_bias_dims.extend(y_d.tolist())

    def _lims(values, floor0=False):
        if len(values) == 0:
            return (0.0, 1.0)
        vmin, vmax = float(np.min(values)), float(np.max(values))
        pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.1
        lo = vmin - pad
        hi = vmax + pad
        if floor0:
            lo = max(0.0, lo)
        return lo, hi

    bias_comb_ylim = _lims(all_bias_comb, floor0=False)
    width_comb_ylim = _lims(all_width_comb, floor0=True)
    width_dims_ylim = _lims(all_width_dims, floor0=True)
    bias_dims_ylim = _lims(all_bias_dims, floor0=False)

    for r, prior_name in enumerate(hpds.keys()):
        pdata = panel_data[prior_name]
        x_comb = pdata["x_comb"]
        bias_comb = pdata["bias_comb"]
        width_comb = pdata["width_comb"]
        width_curves = pdata["width_curves"]
        bias_curves = pdata["bias_curves"]

        a1 = ax[r, 0]
        a1b = a1.twinx()
        l1 = a1.scatter(x_comb, bias_comb, color="C0", s=20, marker="o", label="Avg dim bias")
        l2 = a1b.scatter(
            x_comb,
            width_comb,
            color="C1",
            s=20,
            marker="s",
            label="Avg dim 68% width",
        )

        a1.set_ylim(*bias_comb_ylim)
        a1b.set_ylim(*width_comb_ylim)
        a1.set_xlabel("True inference value")
        a1.set_ylabel("Bias", color="C0")
        a1b.set_ylabel("68% width", color="C1")
        a1.tick_params(axis="y", labelcolor="C0")
        a1b.tick_params(axis="y", labelcolor="C1")
        a1.grid(True, alpha=0.3)
        a1.set_title(f"{prior_name}: averages across dimensions")
        a1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="best", fontsize=9)

        a2 = ax[r, 1]
        for d in range(n_dims):
            x_d, y_d = width_curves[d]
            a2.scatter(
                x_d,
                y_d,
                s=18,
                color=blues[d],
                marker=markers[d % len(markers)],
                label=f"dim {d}",
            )
        a2.set_ylim(*width_dims_ylim)
        a2.set_xlabel("True value in that dimension")
        a2.set_ylabel("68% width")
        a2.grid(True, alpha=0.3)
        a2.set_title(f"{prior_name}: width per dimension")
        a2.legend(loc="best", fontsize=9, ncol=min(n_dims, 3))

        a3 = ax[r, 2]
        for d in range(n_dims):
            x_d, y_d = bias_curves[d]
            a3.scatter(
                x_d,
                y_d,
                s=18,
                color=reds[d],
                marker=markers[d % len(markers)],
                label=f"dim {d}",
            )
            a3.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
        a3.set_ylim(*bias_dims_ylim)
        a3.set_xlabel("True value in that dimension")
        a3.set_ylabel("Avg bias (interval center - true)")
        a3.grid(True, alpha=0.3)
        a3.set_title(f"{prior_name}: avg bias per dimension")
        a3.legend(loc="best", fontsize=9, ncol=min(n_dims, 3))

        a1.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
        a1b.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
        a2.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    return fig, ax


def _enforce_markers(fig, filename=None):
    for axis in fig.axes:
        title = axis.get_title().lower()
        if "width per dimension" in title:
            for line in axis.lines:
                line.set_marker("s")
                line.set_markersize(4)
        elif "avg bias per dimension" in title:
            for line in axis.lines:
                line.set_marker("o")
                line.set_markersize(4)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    return fig


def plot_ratio_violins(all_ratios, test_parameters, filename=None):
    plt.rcParams.update({"font.size": 10})
    n_priors = len(all_ratios)
    ncols = 3
    nrows = (n_priors + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    positions = np.arange(len(test_parameters))
    labels = []
    for tp in test_parameters:
        if np.isscalar(tp):
            labels.append(f"{float(tp):.2f}")
        else:
            labels.append(np.asarray(tp).round(2).tolist())

    for i, (prior_name, ratio_distributions) in enumerate(all_ratios.items()):
        axis = axes[i]
        axis.violinplot(
            ratio_distributions,
            positions=positions,
            showmeans=True,
            showextrema=True,
            widths=0.8,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_title(f"{prior_name} prior")
        axis.set_xlabel(r"parameter $\theta_0$")
        axis.set_ylabel(r"$r(x|\theta_0)=p(x|\theta_0)/p(x)$")
        axis.set_ylim(0, 5)
        axis.grid(alpha=0.3)

        means = [np.mean(d) if len(d) else np.nan for d in ratio_distributions]
        for pos, mean in enumerate(means):
            if np.isfinite(mean):
                y_top = axis.get_ylim()[1]
                y_text = min(mean * 1.05, 0.95 * y_top)
                axis.text(pos, y_text, f"μ={mean:.3g}", ha="center", va="bottom", fontsize=9)

    for k in range(n_priors, nrows * ncols):
        fig.delaxes(axes[k])

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    return fig, axes


def plot_reweighted_distributions(
    priors,
    data_x,
    reweighted_distributions,
    test_parameters,
    n_bins,
    filename=None,
):
    x_all = np.concatenate([np.asarray(data_x[0]).ravel(), np.asarray(data_x[1]).ravel()])
    x_min, x_max = x_all.min(), x_all.max()

    ncols = 3
    nrows = (len(priors) + ncols - 1) // ncols
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    for i, prior in enumerate(priors):
        axis = axes[i]
        axis.hist(
            data_x[0],
            bins=n_bins,
            density=False,
            alpha=0.45,
            label=f"x|θ={test_parameters[0]}",
            color="C0",
        )
        axis.hist(
            data_x[1],
            bins=n_bins,
            density=False,
            alpha=0.45,
            label=f"x|θ={test_parameters[1]}",
            color="C1",
        )

        centers, rw_density = reweighted_distributions[prior.__name__]
        axis.plot(
            centers,
            rw_density,
            drawstyle="steps-mid",
            lw=1,
            color="C3",
            label=f"reweighted θ={test_parameters[0]}→{test_parameters[1]}",
        )

        axis.set_title(f"{prior.__name__} prior")
        axis.set_xlabel("x")
        axis.set_ylabel("Density")
        axis.set_xlim(x_min, x_max)
        axis.grid(alpha=0.3)
        axis.legend()

    for k in range(len(priors), nrows * ncols):
        fig.delaxes(axes[k])

    fig.tight_layout()
    if filename:
        fig.savefig(filename)
    return fig, axes

