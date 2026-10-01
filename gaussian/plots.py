"""Figures across the priors of one configuration. Ordinary Matplotlib, reading only saved data.

Each hook receives one context and one dependency mapping per cohort member and returns
{filename_stem: figure}; the runner saves them as PDFs in runs/<configuration>/plots/<hook>/.
"""
import functools

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import trapezoid
from scipy.special import logsumexp

from nnpd.nre.training import classification
from .metrics import _association

RED, BLUE, GREEN, PURPLE = plt.colormaps["Set1"].colors[:4]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
MEDIAN_BINS = 8  # bins of the binned-median curves in the verification plots


def _font(size, ticks=None):
    """Build the figure with this base font size; tick labels, made only when saving, get `ticks`."""
    def decorate(build):
        @functools.wraps(build)
        def styled(*args):
            with plt.rc_context({"font.size": size}):
                fig = build(*args)
                for ax in fig.axes:
                    ax.tick_params(labelsize=ticks or size)
                fig.tight_layout()
            return fig
        return styled
    return decorate


def _label(context):
    """The prior's name, with the replica when the cohort trains several."""
    replicas = context.training_settings["cohort"]["replicas"]
    return context.member["prior"] if replicas == 1 else context.member["name"]


def _theta(j):
    return rf"$\theta_{{{j}}}$"


def _point(theta, digits=2):
    """One number when every coordinate is equal, as on the diagonal, otherwise the vector."""
    theta = np.round(np.asarray(theta, dtype=float), digits)
    return f"{theta[0]:.{digits}f}" if np.all(theta == theta[0]) else str(theta.tolist())


def _panels(count, width, height):
    """One panel per cohort member, two per row; unused panels are removed."""
    columns = min(2, count)
    rows = -(-count // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False)
    for ax in axes.flat[count:]:
        fig.delaxes(ax)
    return fig, axes.flat[:count]


def _color_violins(parts, color):
    for body in parts["bodies"]:
        body.set(facecolor=color, edgecolor=color, alpha=0.3)
    for name in ("cmeans", "cmins", "cmaxes", "cbars"):
        parts[name].set_color(color)


def _binned_median(x, y):
    """Median of y in equal-width bins of x, at the bin centers; empty bins are skipped."""
    if np.ptp(x) == 0:
        return x[:1], np.median(y, keepdims=True)
    edges = np.linspace(x.min(), x.max(), MEDIAN_BINS + 1)
    index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, MEDIAN_BINS - 1)
    found = np.unique(index)
    return (edges[found] + edges[found + 1]) / 2, np.array([np.median(y[index == i]) for i in found])


def _spearman(ax, x, y):
    rho = _association(x, y)
    ax.text(0.03, 0.97, rf"Spearman $\rho$={'nan' if rho is None else f'{rho:.3g}'}", transform=ax.transAxes,
            ha="left", va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="0.8"))


# Priors -------------------------------------------------------------------------------------------

def _prior_slice(context, points=200):
    """The prior over the first one or two parameters, the others fixed at their middle.

    Normalized over the shown parameters: for an independent prior this is their marginal,
    a density along continuous axes and a probability mass along discrete ones.
    """
    parameters, supports = context.problem.parameters, context.prior.support_axes
    shown = min(2, len(parameters))
    axes = [np.linspace(p.low, p.high, points) if support is None else support
            for p, support in zip(parameters[:shown], supports)]
    theta = np.array([(p.low + p.high) / 2 if support is None else support[len(support) // 2]
                      for p, support in zip(parameters, supports)])
    theta = np.tile(theta, (*(len(axis) for axis in axes), 1))
    for j, values in enumerate(np.meshgrid(*axes, indexing="ij")):
        theta[..., j] = values
    density = np.exp(context.prior.log_prob(theta))
    total = density
    for j in reversed(range(shown)):
        total = total.sum(axis=-1) if supports[j] is not None else trapezoid(total, axes[j], axis=-1)
    return axes, density / total


@_font(10)
def _prior_densities(contexts):
    fig, axes = _panels(len(contexts), 5, 4)
    for ax, context in zip(axes, contexts):
        (x, *y), density = _prior_slice(context)
        discrete = [axis is not None for axis in context.prior.support_axes[:2]]
        unit = "probability mass" if all(discrete) else "density"
        if not y:
            if discrete[0]:
                ax.vlines(x, 0, density, color=RED)
            else:
                ax.plot(x, density, color=RED)
            ax.set(xlabel=_theta(0), ylabel=f"Prior {unit}", ylim=(0, None))
        else:
            grid = np.meshgrid(x, y[0], indexing="ij")
            constant = np.ptp(density) <= 1e-9 * density.max()
            if any(discrete):
                image = ax.scatter(grid[0].ravel(), grid[1].ravel(), c=density.ravel(), cmap="viridis",
                                   marker="s", s=18, linewidths=0)
            else:
                levels = density.max() * np.array([0.5, 1.5]) if constant else 30
                image = ax.contourf(*grid, density, levels=levels, cmap="viridis")
                if not constant:
                    ax.contour(*grid, density, levels=10, colors="k", linewidths=0.5, alpha=0.4)
            bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=f"Prior {unit}")
            if constant:
                bar.set_ticks([density.max()], labels=[f"{density.max():.3g}"])
            ax.set(xlabel=_theta(0), ylabel=_theta(1))
        ax.set_title(_label(context))
        ax.grid(alpha=0.3)
    return fig


@_font(10)
def _prior_samples(contexts):
    settings = contexts[0].analysis_settings["plotting"]
    fig, axes = _panels(len(contexts), 5, 4)
    for ax, context in zip(axes, contexts):
        samples = context.prior.sample(settings["prior_points"], context.rng("prior-plot"))
        bounds = [(p.low, p.high) for p in context.problem.parameters[:2]]
        if samples.shape[1] == 1:
            ax.hist(samples[:, 0], bins=settings["bins"], range=bounds[0], density=True, color=RED, alpha=0.6)
            ax.set(xlabel=_theta(0), ylabel="Density")
        else:
            ax.hist2d(samples[:, 0], samples[:, 1], bins=settings["bins"], range=bounds, cmap="viridis")
            ax.set(xlabel=_theta(0), ylabel=_theta(1))
        ax.set_title(f"Samples from {_label(context)} prior")
        ax.grid(alpha=0.3)
    return fig


def priors(contexts, dependencies):
    """Each training prior over the first two parameters (or the only one), and samples from it."""
    densities = "prior_contours" if len(contexts[0].problem.parameters) > 1 else "prior_densities"
    return {densities: _prior_densities(contexts), "prior_sample_histograms": _prior_samples(contexts)}


# Training ------------------------------------------------------------------------------------------

def _roc(labels, logits):
    """False and true positive rates at every distinct threshold, from (0, 0) to (1, 1)."""
    order = np.argsort(-logits, kind="stable")
    positive = np.asarray(labels)[order] == 1
    last = np.r_[np.flatnonzero(np.diff(logits[order])), len(order) - 1]
    true, false = np.cumsum(positive)[last], np.cumsum(~positive)[last]
    return np.r_[0, false / false[-1]], np.r_[0, true / true[-1]]


@_font(8)
def _training(context, dependencies):
    history, data = dependencies["model"].json("history"), dependencies["test_predictions"]
    labels, logits = data.array("labels"), data.array("logits")
    test, name = classification(labels, logits), _label(context)
    validation = history["validation"]
    epochs = np.arange(1, len(validation) + 1)
    fig, (loss, scores, roc) = plt.subplots(1, 3, figsize=(12, 3))
    loss.plot(epochs, history["train_loss"], color=RED, label="Train loss")
    loss.plot(epochs, [row["bce"] for row in validation], color=BLUE, label="Validation loss")
    loss.set(xlabel="Epoch", ylabel="Loss", title=f"Training results for {name} prior")
    accuracy, auc = [row["accuracy"] for row in validation], [row["auc"] for row in validation]
    scores.plot(epochs, accuracy, color=RED, label="Validation accuracy")
    scores.plot(epochs, auc, color=BLUE, label="Validation AUC")
    scores.set(xlabel="Epoch", ylabel="Metric", ylim=(min(0.5, *accuracy, *auc), 1),
               title=f"Training results for {name} prior")
    roc.plot(*_roc(labels, logits), color=RED, label=f"AUC = {test['auc']:.4f}")
    roc.plot([0, 1], [0, 1], "k--", label="Random guess")
    roc.set(xlabel="False positive rate", ylabel="True positive rate", title=f"Test ROC curve for {name} prior")
    roc.text(0.66, 0.22, f"Test loss: {test['bce']:.3f}\nTest acc.: {test['accuracy']:.3f}", transform=roc.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="none"))
    for ax in (loss, scores, roc):
        ax.legend()
    return fig


@_font(8)
def _all_rocs(contexts, dependencies):
    curves = []
    for context, data in zip(contexts, dependencies):
        labels, logits = data["test_predictions"].array("labels"), data["test_predictions"].array("logits")
        curves.append((classification(labels, logits)["auc"], _label(context), *_roc(labels, logits)))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_prop_cycle(color=plt.colormaps["Set1"].colors)
    for auc, name, false, true in sorted(curves, key=lambda curve: -curve[0]):
        ax.plot(false, true, linewidth=1.5, alpha=0.9, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC curves across priors")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def training(contexts, dependencies):
    """Each model's loss, validation scores and test ROC curve, and all test ROC curves together."""
    figures = {f"training_{_label(context)}_prior": _training(context, data)
               for context, data in zip(contexts, dependencies)}
    figures["all_roc_curves"] = _all_rocs(contexts, dependencies)
    return figures


# Posterior errors ----------------------------------------------------------------------------------

def _groups(values, count):
    """Group the truths along one parameter: by distinct value, or in `count` equal bins if there are more."""
    distinct, index = np.unique(np.round(values, 9), return_inverse=True)
    if len(distinct) > count:
        edges = np.linspace(values.min(), values.max(), count + 1)
        _, index = np.unique(np.clip(np.searchsorted(edges, values, side="right") - 1, 0, count - 1),
                             return_inverse=True)
    return index


def _mean(values, index):
    return np.bincount(index, weights=values) / np.bincount(index)


def _limits(values, floor=None):
    low, high = float(np.min(values)), float(np.max(values))
    pad = 0.05 * (high - low) if high > low else 0.1
    return (low - pad if floor is None else max(floor, low - pad)), high + pad


@_font(16, ticks=14)
def _errorbars(contexts, dependencies, target):
    parameters = contexts[0].problem.parameters
    count = contexts[0].analysis_settings["plotting"]["truth_bins"]
    fig, axes = plt.subplots(len(contexts), len(parameters), figsize=(5 * len(parameters), 3.8 * len(contexts)),
                             squeeze=False)
    for row, context, data in zip(axes, contexts, dependencies):
        inference, truth = data["inference"], data["observations"].array("truth")
        mode, lower, upper = (inference.array(f"{target}_{name}") for name in ("mode", "lower", "upper"))
        levels = inference.metadata["levels"]
        for j, (ax, parameter) in enumerate(zip(row, parameters)):
            index = _groups(truth[:, j], count)
            x, y = _mean(truth[:, j], index), _mean(mode[:, j], index)
            for k in np.argsort(levels)[::-1]:  # the widest region first and faintest
                below = np.maximum(y - _mean(lower[:, k, j], index), 0)
                above = np.maximum(_mean(upper[:, k, j], index) - y, 0)
                ax.errorbar(x, y, yerr=[below, above], fmt="o", color=RED, ecolor=RED, capsize=5,
                            alpha=1 if levels[k] == min(levels) else 0.35, label=f"{100 * levels[k]:g}% HLD")
            ax.plot([parameter.low, parameter.high], [parameter.low, parameter.high], "k--", label="Ideal")
            ax.set_title(f"{_label(context)} prior ({_theta(j)})", fontsize=18)
            ax.set(xlabel=f"True {_theta(j)}", ylabel=f"Estimated {_theta(j)}")
            ax.grid()
            ax.legend(fontsize=14)
    return fig


@_font(12)
def _error_and_width(contexts, dependencies, target):
    p = len(contexts[0].problem.parameters)
    count = contexts[0].analysis_settings["plotting"]["truth_bins"]
    blues, reds = (plt.colormaps[name](np.linspace(0.45, 0.9, max(p, 2))) for name in ("Blues", "Reds"))
    rows = []
    for data in dependencies:
        inference, truth = data["inference"], data["observations"].array("truth")
        levels = inference.metadata["levels"]
        level = int(np.argmin(levels))
        bias = inference.array(f"{target}_mode") - truth
        width = inference.array(f"{target}_upper")[:, level] - inference.array(f"{target}_lower")[:, level]
        curves = []  # per parameter: true values, mean bias, mean width
        for j in range(p):
            index = _groups(truth[:, j], count)
            curves.append([_mean(values[:, j], index) for values in (truth, bias, width)])
        pooled = [np.concatenate(values) for values in zip(*curves)]
        index = _groups(pooled[0], count)
        rows.append((curves, [_mean(values, index) for values in pooled]))
    name = f"{100 * levels[level]:g}% HLD width"
    bias_limits = _limits(np.concatenate([combined[1] for _, combined in rows]))
    width_limits = _limits(np.concatenate([combined[2] for _, combined in rows]), floor=0)
    bias_axis_limits = _limits(np.concatenate([curve[1] for curves, _ in rows for curve in curves]))
    width_axis_limits = _limits(np.concatenate([curve[2] for curves, _ in rows for curve in curves]), floor=0)
    fig, axes = plt.subplots(len(contexts), 3, figsize=(19, 3.8 * len(contexts)), squeeze=False)
    for (averages, widths, biases), context, (curves, (x, bias, width)) in zip(axes, contexts, rows):
        label = _label(context)
        twin = averages.twinx()
        shown = [averages.scatter(x, bias, color=RED, s=20, marker="o", label="Avg dim bias"),
                 twin.scatter(x, width, color=BLUE, s=20, marker="s", label=f"Avg dim {name}")]
        averages.set(xlabel="True inference value", ylim=bias_limits, title=f"{label}: averages across dimensions")
        averages.set_ylabel("Bias", color=RED)
        twin.set_ylim(*width_limits)
        twin.set_ylabel(name, color=BLUE)
        averages.tick_params(axis="y", labelcolor=RED)
        twin.tick_params(axis="y", labelcolor=BLUE)
        averages.legend(shown, [item.get_label() for item in shown], loc="best", fontsize=9)
        for j, (truth, mean_bias, mean_width) in enumerate(curves):
            marker = MARKERS[j % len(MARKERS)]
            widths.scatter(truth, mean_width, s=18, color=blues[j], marker=marker, label=_theta(j))
            biases.scatter(truth, mean_bias, s=18, color=reds[j], marker=marker, label=_theta(j))
        widths.set(xlabel="True value in that dimension", ylabel=name, ylim=width_axis_limits,
                   title=f"{label}: HLD width per dimension")
        biases.set(xlabel="True value in that dimension", ylabel="Avg bias (mode - true)", ylim=bias_axis_limits,
                   title=f"{label}: avg bias per dimension")
        for ax in (averages, biases):
            ax.axhline(0, color="0.35", linestyle="--", linewidth=1, alpha=0.8, zorder=0)
        for ax in (averages, widths, biases):
            ax.grid(True, alpha=0.3)
        for ax in (widths, biases):
            ax.legend(loc="best", fontsize=9, ncol=min(p, 3))
    return fig


def posterior_errors(contexts, dependencies):
    """Estimates against truths with their HLD regions, and bias and HLD width against the truth.

    Truths are averaged by true value along each parameter, as described in _groups. The
    "_ratios" figures use the normalized likelihood ratio, the others the posterior.
    """
    figures = {}
    targets = dependencies[0]["inference"].metadata["targets"]
    for target, suffix in (("posterior", ""), ("ratio", "_ratios")):
        if target in targets:
            figures[f"errorbars{suffix}"] = _errorbars(contexts, dependencies, target)
            figures[f"error_and_hld_width{suffix}"] = _error_and_width(contexts, dependencies, target)
    return figures


# Verifications -------------------------------------------------------------------------------------

@_font(10)
def _ratio_violins(contexts, dependencies):
    fig, axes = _panels(len(contexts), 6, 4)
    for ax, context, data in zip(axes, contexts, dependencies):
        theta = data["normalization_observations"].array("theta")
        ratios = np.exp(data["normalization_predictions"].array("log_ratio"))
        positions = np.arange(len(theta))
        _color_violins(ax.violinplot(list(ratios), positions=positions, showmeans=True, showextrema=True,
                                     widths=0.8), RED)
        ax.set_xticks(positions, [_point(point) for point in theta])
        ax.set(title=f"{_label(context)} prior", xlabel=r"parameter $\theta$",
               ylabel=r"$r(x|\theta)=p(x|\theta)/p(x)$", ylim=(0, 5))
        ax.grid(alpha=0.3)
        for position, values in zip(positions, ratios):
            ax.text(position, min(values.mean() * 1.05, 0.95 * 5), f"μ={values.mean():.3g}", ha="center",
                    va="bottom", fontsize=9)
    return fig


@_font(10)
def _reweighted(contexts, dependencies):
    fig, axes = _panels(len(contexts), 6, 4)
    bins = contexts[0].analysis_settings["verification"]["showcase"]["bins"]
    for ax, context, data in zip(axes, contexts, dependencies):
        samples = data["showcase_observations"]
        start, end = (_point(point, 1) for point in samples.array("theta"))
        source, target = samples.array("source")[:, 0], samples.array("target")[:, 0]
        score = data["showcase_predictions"].array("log_ratio")
        weights = np.exp(score - logsumexp(score))
        edges = np.histogram_bin_edges(np.concatenate((source, target)), bins=bins)
        ax.hist(source, bins=edges, density=True, histtype="stepfilled", alpha=0.45, color=RED,
                label=f"x|θ={start}")
        ax.hist(target, bins=edges, density=True, histtype="stepfilled", alpha=0.45, color=BLUE, label=f"x|θ={end}")
        ax.hist(source, bins=edges, weights=weights, density=True, histtype="step", linewidth=1, color=PURPLE,
                label=f"reweighted θ={start}→{end}")
        ax.set(title=f"{_label(context)} prior\nN={len(source):,} | ESS={1 / np.sum(weights ** 2):,.0f}",
               xlabel="x" if context.problem.observation_dim == 1 else r"$x_0$", ylabel="Density",
               xlim=(edges[0], edges[-1]))
        ax.grid(alpha=0.3)
        ax.legend()
    return fig


def _pairs(data):
    """Endpoint distance, per-pair RMSE of the learned log ratio, and all learned and exact log ratios."""
    predictions = data["verification_predictions"]
    learned, exact = predictions.array("probe_log_ratio"), predictions.array("probe_exact_log_ratio")
    pairs = data["verification_observations"].array("pairs")
    rmse = np.sqrt(np.mean((learned - exact) ** 2, axis=1))
    return np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1), rmse, learned, exact


def _point_legend():
    return [Line2D([], [], linestyle="", marker="o", color="tab:blue", markersize=6, label="parameter pair"),
            Line2D([], [], color="black", marker="o", linewidth=1.5, markersize=4, label="binned median")]


@_font(10)
def _error_vs_distance(contexts, dependencies):
    fig, axes = _panels(len(contexts), 6, 4)
    results = [_pairs(data) for data in dependencies]
    top = 1.05 * max(rmse.max() for _, rmse, _, _ in results) or 1
    for ax, context, (distance, rmse, _, _) in zip(axes, contexts, results):
        ax.scatter(distance, rmse, alpha=0.7, s=25, color="tab:blue")
        ax.plot(*_binned_median(distance, rmse), color="black", linewidth=1.5, marker="o", markersize=4)
        ax.set(title=f"{_label(context)} prior\nmedian={np.median(rmse):.3g} | worst={rmse.max():.3g}",
               xlabel=r"$||\theta_1 - \theta_0||_2$", ylabel=r"RMSE of $\log \hat{R}_{10}(x)$", ylim=(0, top))
        ax.grid(alpha=0.3)
        ax.legend(handles=_point_legend(), loc="upper right", fontsize=9)
        _spearman(ax, distance, rmse)
    return fig


@_font(10)
def _exact_vs_predicted(contexts, dependencies):
    fig, axes = _panels(len(contexts), 6, 4)
    for ax, context, data in zip(axes, contexts, dependencies):
        _, rmse, learned, exact = _pairs(data)
        learned, exact = learned.ravel(), exact.ravel()
        cells = ax.hexbin(exact, learned, gridsize=35, mincnt=1, cmap="viridis")
        ends = [min(exact.min(), learned.min()), max(exact.max(), learned.max())]
        ax.plot(ends, ends, linestyle="--", color="black", linewidth=1, label="ideal agreement")
        pooled = np.sqrt(np.mean((learned - exact) ** 2))
        ax.set(title=f"{_label(context)} prior\npooled RMSE={pooled:.3g} | pair median={np.median(rmse):.3g} | "
                     f"pair worst={rmse.max():.3g}",
               xlabel=r"exact $\log R_{10}(x)$", ylabel=r"predicted $\log \hat{R}_{10}(x)$")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        fig.colorbar(cells, ax=ax, fraction=0.046, pad=0.04, label="samples per hexbin")
    return fig


def _reweighting(data):
    """Endpoint distance, learned sliced W1 and ESS/N of every reweighting pair."""
    distances = data["reweighting_distances"]
    pairs = data["verification_observations"].array("pairs")
    return (np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1), distances.array("learned_w1"),
            distances.array("ess") / distances.metadata["source_samples_per_pair"])


@_font(10)
def _swd_vs_distance(contexts, dependencies):
    fig, axes = _panels(len(contexts), 6, 4)
    results = [_reweighting(data) for data in dependencies]
    top = 1.05 * max(swd.max() for _, swd, _ in results) or 1
    for ax, context, (distance, swd, ess) in zip(axes, contexts, results):
        points = ax.scatter(distance, swd, c=ess, cmap="viridis", s=30, alpha=0.9)
        ax.plot(*_binned_median(distance, swd), color="black", linewidth=1.5, marker="o", markersize=4)
        ax.set(title=f"{_label(context)} prior\nmedian={np.median(swd):.3g} | worst={swd.max():.3g}",
               xlabel=r"$||\theta_1 - \theta_0||_2$", ylabel=r"$\mathrm{SW}_1$", ylim=(0, top))
        ax.grid(alpha=0.3)
        ax.legend(handles=_point_legend(), loc="upper right", fontsize=9)
        fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04, label=r"$\mathrm{ESS}/N$")
        _spearman(ax, distance, swd)
    return fig


@_font(10)
def _reweighting_summary(contexts, dependencies):
    results = [_reweighting(data) for data in dependencies]
    positions = np.arange(1, len(contexts) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column, name in zip(axes, (1, 2), (r"$\mathrm{SW}_1$", r"$\mathrm{ESS}/N$")):
        _color_violins(ax.violinplot([result[column] for result in results], positions=positions, showmeans=True,
                                     showextrema=True, widths=0.8), RED)
        ax.set_xticks(positions, [_label(context) for context in contexts])
        ax.set(title=f"{name} by prior", ylabel=name)
        ax.grid(alpha=0.3)
    return fig


def verifications(contexts, dependencies):
    """Mean ratio under each prior's evidence, fixed-parameter reweighting, and pairwise
    log-ratio and reweighting errors against the endpoint distance."""
    start, end = (_point(point, 1) for point in dependencies[0]["showcase_observations"].array("theta"))
    reweighted = "reweighted_distributions" + ("" if "[" in start + end else f"_{start}_{end}")
    return {"ratio_violins": _ratio_violins(contexts, dependencies),
            reweighted: _reweighted(contexts, dependencies),
            "log_ratio_error_vs_distance": _error_vs_distance(contexts, dependencies),
            "log_ratio_exact_vs_predicted": _exact_vs_predicted(contexts, dependencies),
            "reweighting_swd_vs_distance": _swd_vs_distance(contexts, dependencies),
            "reweighting_summary": _reweighting_summary(contexts, dependencies)}
