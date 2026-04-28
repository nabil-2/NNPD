from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .posterior import _extract_hpd_level, _extract_map_array
from .priors import snap_to_support_axis


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
        support_axes = getattr(prior, "support_axes", None)
        if getattr(prior, "support_kind", None) == "discrete_grid" and support_axes is not None:
            x_axis = np.asarray(support_axes[0], dtype=float)
            y_axis = np.asarray(support_axes[1], dtype=float)
            x_disc, y_disc = np.meshgrid(x_axis, y_axis, indexing="ij")
            grid_disc = np.empty(x_disc.shape + (n_parameters,), dtype=float)
            for dim in range(n_parameters):
                axis = np.asarray(support_axes[dim], dtype=float)
                fill_value = axis[axis.size // 2] if fixed_value is None else snap_to_support_axis(fixed_value, axis)
                grid_disc[..., dim] = float(fill_value)
            grid_disc[..., 0] = x_disc
            grid_disc[..., 1] = y_disc
            z = prior(grid_disc)
            cf = axes[i].scatter(
                x_disc.ravel(),
                y_disc.ravel(),
                c=np.asarray(z, dtype=float).ravel(),
                cmap="viridis",
                marker="s",
                s=18,
                linewidths=0.0,
            )
        else:
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
    layout="vertical",
):
    _, parameter_combinations, _ = inference_data
    true_params = np.asarray(parameter_combinations, dtype=np.float32)

    n_dims = len(parameters_post)
    n_posteriors = len(hpds)

    if layout == "horizontal":
        plt.rcParams.update({"font.size": 20})
        title_fontsize = 24
        label_fontsize = 22
        tick_fontsize = 18
        legend_fontsize = 18
        panel_height = 5.2
    else:
        plt.rcParams.update({"font.size": 16})
        title_fontsize = 18
        label_fontsize = 16
        tick_fontsize = 14
        legend_fontsize = 14
        panel_height = 3.8

    if layout == "vertical":
        nrows, ncols = n_posteriors, n_dims

        def _axis_at(prior_idx, dim_idx):
            return ax[prior_idx, dim_idx]

    elif layout == "horizontal":
        nrows, ncols = n_dims, n_posteriors

        def _axis_at(prior_idx, dim_idx):
            return ax[dim_idx, prior_idx]

    elif layout == "grid":
        n_panels = n_posteriors * n_dims
        ncols = int(np.ceil(np.sqrt(n_panels)))
        nrows = (n_panels + ncols - 1) // ncols

        def _axis_at(prior_idx, dim_idx):
            return ax.flat[prior_idx * n_dims + dim_idx]

    else:
        raise ValueError("layout must be 'vertical', 'horizontal', or 'grid'")

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, panel_height * nrows),
        squeeze=False,
    )

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
        y_hat = _extract_map_array(
            all_posteriors[prior_name],
            parameters_post=parameters_post,
            n_dims=n_dims,
        )
        intervals68, _ = _extract_hpd_level(errors, level="68", n_dims=n_dims)
        intervals95, _ = _extract_hpd_level(errors, level="95", n_dims=n_dims)
        n_points = int(y_hat.shape[0])
        if intervals68.shape[0] != n_points or intervals95.shape[0] != n_points:
            raise ValueError(
                f"For prior '{prior_name}', HLD count does not match posterior count ({n_points})."
            )

        for d in range(n_dims):
            x_true = true_params[:, d]
            y_est = y_hat[:, d]

            low68 = intervals68[:, d, 0]
            high68 = intervals68[:, d, 1]
            low95 = intervals95[:, d, 0]
            high95 = intervals95[:, d, 1]
            yerr68_lower = np.abs(y_est - low68)
            yerr68_upper = np.abs(high68 - y_est)
            yerr95_lower = np.abs(y_est - low95)
            yerr95_upper = np.abs(high95 - y_est)
            widths68 = high68 - low68

            grouped = _aggregate(
                x_true,
                y_est,
                yerr68_lower,
                yerr68_upper,
                yerr95_lower,
                yerr95_upper,
                widths68,
            )

            a = _axis_at(i, d)
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
                label="95% HLD",
            )
            a.errorbar(
                grouped["x"],
                grouped["y"],
                yerr=yerr68,
                fmt="o",
                color="C0",
                ecolor="C0",
                capsize=5,
                label="68% HLD",
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
            a.set_title(rf"{prior_name} prior ($\theta_{{{d}}}$)", fontsize=title_fontsize)
            a.set_xlabel(rf"True $\theta_{{{d}}}$", fontsize=label_fontsize)
            a.set_ylabel(rf"Estimated $\theta_{{{d}}}$", fontsize=label_fontsize)
            a.tick_params(axis="both", labelsize=tick_fontsize)
            a.grid()
            a.legend(fontsize=legend_fontsize)

    if layout == "grid":
        for idx in range(n_posteriors * n_dims, nrows * ncols):
            fig.delaxes(ax.flat[idx])

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    return fig, ax


def plot_error_and_hld_width(hpds: dict, inference_data, all_posteriors, parameters_post, filename=None):
    _, parameter_combinations, _ = inference_data
    true_params = np.asarray(parameter_combinations, dtype=np.float32)

    n_dims = len(parameters_post)
    n_posteriors = len(hpds)

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(
        nrows=n_posteriors,
        ncols=3,
        figsize=(19, 3.8 * n_posteriors),
        squeeze=False,
    )

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    blues = plt.cm.Blues(np.linspace(0.45, 0.9, max(n_dims, 2)))
    reds = plt.cm.Reds(np.linspace(0.45, 0.9, max(n_dims, 2)))

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
        y_hat = _extract_map_array(
            all_posteriors[prior_name],
            parameters_post=parameters_post,
            n_dims=n_dims,
        )
        intervals_68, _ = _extract_hpd_level(errors, level="68", n_dims=n_dims)
        if intervals_68.shape[0] != y_hat.shape[0]:
            raise ValueError(
                f"For prior '{prior_name}', HLD count ({intervals_68.shape[0]}) does not match "
                f"posterior count ({y_hat.shape[0]})."
            )

        width_dims = intervals_68[:, :, 1] - intervals_68[:, :, 0]
        bias_dims = y_hat - true_params

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
            label="Avg dim 68% HLD width",
        )

        a1.set_ylim(*bias_comb_ylim)
        a1b.set_ylim(*width_comb_ylim)
        a1.set_xlabel("True inference value")
        a1.set_ylabel("Bias", color="C0")
        a1b.set_ylabel("68% HLD width", color="C1")
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
        a2.set_ylabel("68% HLD width")
        a2.grid(True, alpha=0.3)
        a2.set_title(f"{prior_name}: HLD width per dimension")
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
        a3.set_ylabel("Avg bias (MAP - true)")
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


# Backward-compatible alias for older scripts/notebooks that still use the old HPD name.
plot_error_and_hpd_width = plot_error_and_hld_width


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
    ncols = _verification_grid_columns(n_priors)
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
    reweighting_distributions=None,
):
    x_all = np.concatenate([np.asarray(data_x[0]).ravel(), np.asarray(data_x[1]).ravel()])
    x_min, x_max = x_all.min(), x_all.max()
    shared_bins = np.histogram_bin_edges(x_all, bins=n_bins)

    ncols = _verification_grid_columns(len(priors))
    nrows = (len(priors) + ncols - 1) // ncols
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    for i, prior in enumerate(priors):
        axis = axes[i]
        axis.hist(
            data_x[0],
            bins=shared_bins,
            density=True,
            alpha=0.45,
            label=f"x|θ={test_parameters[0]}",
            color="C0",
        )
        axis.hist(
            data_x[1],
            bins=shared_bins,
            density=True,
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

        title = f"{prior.__name__} prior"
        if reweighting_distributions is not None and prior.__name__ in reweighting_distributions:
            weights = np.asarray(reweighting_distributions[prior.__name__], dtype=float).ravel()
            n_samples = int(weights.size)
            weight_sum = np.sum(weights)
            weight_sq_sum = np.sum(np.square(weights))
            if (
                n_samples > 0
                and np.isfinite(weight_sum)
                and np.isfinite(weight_sq_sum)
                and weight_sq_sum > 0
            ):
                ess = (weight_sum**2) / weight_sq_sum
            else:
                ess = 0.0
            title = f"{title}\nN={n_samples:,} | ESS={int(np.rint(ess)):,}"
        axis.set_title(title)
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


def _binned_median_curve(x, y, n_bins=8):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.asarray([]), np.asarray([])
    if np.allclose(x, x[0]):
        return np.asarray([x[0]], dtype=float), np.asarray([np.median(y)], dtype=float)

    edges = np.linspace(np.min(x), np.max(x), int(n_bins) + 1)
    centers = []
    medians = []
    for idx in range(len(edges) - 1):
        if idx == len(edges) - 2:
            mask = (x >= edges[idx]) & (x <= edges[idx + 1])
        else:
            mask = (x >= edges[idx]) & (x < edges[idx + 1])
        if np.any(mask):
            centers.append(0.5 * (edges[idx] + edges[idx + 1]))
            medians.append(np.median(y[mask]))
    return np.asarray(centers, dtype=float), np.asarray(medians, dtype=float)


def _format_metric(value, digits=3):
    value = float(value)
    if np.isnan(value):
        return "nan"
    return f"{value:.{digits}g}"


def _verification_grid_columns(n_priors):
    n_priors = int(n_priors)
    return max(1, min(2, n_priors))


def plot_log_ratio_error_vs_distance(model_quality_results, filename=None, n_bins=8):
    plt.rcParams.update({"font.size": 10})
    n_priors = len(model_quality_results)
    ncols = _verification_grid_columns(n_priors)
    nrows = (n_priors + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
    rmse_arrays = [
        np.asarray(result["pair_metrics"]["rmse_log_r10"], dtype=float).ravel()
        for result in model_quality_results.values()
    ]
    rmse_max = max((float(np.max(values)) for values in rmse_arrays if values.size), default=1.0)
    rmse_ylim = (0.0, rmse_max * 1.05 if rmse_max > 0 else 1.0)
    legend_handles = [
        Line2D([], [], linestyle="", marker="o", color="tab:blue", markersize=6, label="parameter pair"),
        Line2D([], [], linestyle="-", marker="o", color="black", linewidth=1.5, markersize=4, label="binned median"),
    ]

    for idx, (prior_name, result) in enumerate(model_quality_results.items()):
        axis = axes[idx]
        pair_metrics = result["pair_metrics"]
        summary = result["summary"]
        distance = np.asarray(pair_metrics["distance"], dtype=float)
        rmse = np.asarray(pair_metrics["rmse_log_r10"], dtype=float)

        axis.scatter(distance, rmse, alpha=0.7, s=25, color="tab:blue")
        line_x, line_y = _binned_median_curve(distance, rmse, n_bins=n_bins)
        if line_x.size:
            axis.plot(line_x, line_y, color="black", linewidth=1.5, marker="o", markersize=4)

        title = (
            f"{prior_name} prior\n"
            f"median={_format_metric(summary['median_rmse_log_r10'])} | "
            f"worst={_format_metric(summary['worst_rmse_log_r10'])}"
        )
        axis.set_title(title)
        axis.set_xlabel(r"$||\theta_1 - \theta_0||_2$")
        axis.set_ylabel(r"RMSE of $\log \hat{R}_{10}(x)$")
        axis.set_ylim(*rmse_ylim)
        axis.grid(alpha=0.3)
        axis.legend(handles=legend_handles, loc="upper right", fontsize=9)

        spearman = summary.get("spearman_rmse_vs_distance", float("nan"))
        axis.text(
            0.03,
            0.97,
            rf"Spearman $\rho$={_format_metric(spearman)}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="0.8"),
        )

    for idx in range(n_priors, nrows * ncols):
        fig.delaxes(axes[idx])

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    return fig, axes


def plot_log_ratio_exact_vs_predicted(model_quality_results, filename=None):
    plt.rcParams.update({"font.size": 10})
    n_priors = len(model_quality_results)
    ncols = _verification_grid_columns(n_priors)
    nrows = (n_priors + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    for idx, (prior_name, result) in enumerate(model_quality_results.items()):
        axis = axes[idx]
        pooled = result["pooled"]
        summary = result["summary"]
        exact = np.asarray(pooled["exact_log_r10"], dtype=float)
        predicted = np.asarray(pooled["predicted_log_r10"], dtype=float)

        hexbin = axis.hexbin(exact, predicted, gridsize=35, mincnt=1, cmap="viridis")
        min_edge = float(np.min(np.concatenate([exact, predicted])))
        max_edge = float(np.max(np.concatenate([exact, predicted])))
        axis.plot(
            [min_edge, max_edge],
            [min_edge, max_edge],
            linestyle="--",
            color="black",
            linewidth=1,
            label="ideal agreement",
        )
        title = (
            f"{prior_name} prior\n"
            f"pooled RMSE={_format_metric(summary['pooled_rmse_log_r10'])} | "
            f"pair median={_format_metric(summary['median_rmse_log_r10'])} | "
            f"pair worst={_format_metric(summary['worst_rmse_log_r10'])}"
        )
        axis.set_title(title)
        axis.set_xlabel(r"exact $\log R_{10}(x)$")
        axis.set_ylabel(r"predicted $\log \hat{R}_{10}(x)$")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper left", fontsize=9)
        fig.colorbar(hexbin, ax=axis, fraction=0.046, pad=0.04, label="samples per hexbin")

    for idx in range(n_priors, nrows * ncols):
        fig.delaxes(axes[idx])

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    return fig, axes


def plot_reweighting_swd_vs_distance(reweighting_results, filename=None, n_bins=8):
    plt.rcParams.update({"font.size": 10})
    n_priors = len(reweighting_results)
    ncols = _verification_grid_columns(n_priors)
    nrows = (n_priors + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
    swd_arrays = [
        np.asarray(result["pair_metrics"]["swd"], dtype=float).ravel()
        for result in reweighting_results.values()
    ]
    swd_max = max((float(np.max(values)) for values in swd_arrays if values.size), default=1.0)
    swd_ylim = (0.0, swd_max * 1.05 if swd_max > 0 else 1.0)
    legend_handles = [
        Line2D([], [], linestyle="", marker="o", color="tab:blue", markersize=6, label="parameter pair"),
        Line2D([], [], linestyle="-", marker="o", color="black", linewidth=1.5, markersize=4, label="binned median"),
    ]

    for idx, (prior_name, result) in enumerate(reweighting_results.items()):
        axis = axes[idx]
        pair_metrics = result["pair_metrics"]
        summary = result["summary"]
        distance = np.asarray(pair_metrics["distance"], dtype=float)
        swd = np.asarray(pair_metrics["swd"], dtype=float)
        ess_fraction = np.asarray(pair_metrics["ess_fraction"], dtype=float)

        scatter = axis.scatter(distance, swd, c=ess_fraction, cmap="viridis", s=30, alpha=0.9)
        line_x, line_y = _binned_median_curve(distance, swd, n_bins=n_bins)
        if line_x.size:
            axis.plot(line_x, line_y, color="black", linewidth=1.5, marker="o", markersize=4)

        title = (
            f"{prior_name} prior\n"
            f"median={_format_metric(summary['median_swd'])} | "
            f"worst={_format_metric(summary['worst_swd'])}"
        )
        axis.set_title(title)
        axis.set_xlabel(r"$||\theta_1 - \theta_0||_2$")
        axis.set_ylabel(r"$\mathrm{SW}_1$")
        axis.set_ylim(*swd_ylim)
        axis.grid(alpha=0.3)
        axis.legend(handles=legend_handles, loc="upper right", fontsize=9)
        fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04, label=r"$\mathrm{ESS}/N$")

        spearman = summary.get("spearman_swd_vs_distance", float("nan"))
        axis.text(
            0.03,
            0.97,
            rf"Spearman $\rho$={_format_metric(spearman)}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="0.8"),
        )

    for idx in range(n_priors, nrows * ncols):
        fig.delaxes(axes[idx])

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    return fig, axes


def plot_reweighting_summary(reweighting_results, filename=None):
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    prior_names = list(reweighting_results.keys())
    swd_values = [np.asarray(result["pair_metrics"]["swd"], dtype=float) for result in reweighting_results.values()]
    ess_fraction_values = [
        np.asarray(result["pair_metrics"]["ess_fraction"], dtype=float) for result in reweighting_results.values()
    ]
    positions = np.arange(1, len(prior_names) + 1)

    axes[0].violinplot(swd_values, positions=positions, showmeans=True, showextrema=True, widths=0.8)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(prior_names)
    axes[0].set_title(r"$\mathrm{SW}_1$ by prior")
    axes[0].set_ylabel(r"$\mathrm{SW}_1$")
    axes[0].grid(alpha=0.3)

    axes[1].violinplot(ess_fraction_values, positions=positions, showmeans=True, showextrema=True, widths=0.8)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(prior_names)
    axes[1].set_title(r"$\mathrm{ESS}/N$ by prior")
    axes[1].set_ylabel(r"$\mathrm{ESS}/N$")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    return fig, axes
