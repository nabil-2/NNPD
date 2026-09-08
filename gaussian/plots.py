"""Ordinary Matplotlib functions with unrestricted access to context and saved data."""
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from nnpd.nre.inference import normalized_mass


def training(context, dependencies):
    history = dependencies["model"].json("history")
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], marker="o", label="Training")
    ax.plot(epochs, [row["bce"] for row in history["validation"]], marker="o", label="Validation")
    ax.set(xlabel="Epoch", ylabel="Binary cross entropy", title=context.member["prior"])
    ax.legend()
    return {"loss": fig}


def prior(context, dependencies):
    figures = {}
    samples = context.prior.sample(context.config["plotting"]["prior_points"], context.rng("prior-plot"))
    for column, parameter in enumerate(context.problem.parameters):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(samples[:, column], bins=context.config["plotting"]["bins"], density=True, alpha=0.6)
        ax.set(xlabel=parameter.name, ylabel="Empirical density", title=f"Training prior: {context.member['prior']}")
        figures[parameter.name.replace(":", "-")] = fig
    return figures


def inference(context, dependencies):
    data, observations, candidates = (dependencies[key] for key in ("inference", "observations", "candidates"))
    truths, modes = observations.array("truth"), data.array("ratio_mode")
    lower, upper = data.array("ratio_lower"), data.array("ratio_upper")
    figures = {}
    chosen = np.linspace(0, len(truths) - 1, min(len(truths), context.config["plotting"]["max_inference_cases"]), dtype=int)
    for axis, name in enumerate(context.problem.names):
        order = chosen[np.argsort(truths[chosen, axis], kind="stable")]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for level in reversed(range(len(data.metadata["levels"]))):
            # Bounds are joint-HLD projections; use vlines, not potentially negative yerr at nonnested modes.
            ax.vlines(truths[order, axis], lower[order, level, axis], upper[order, level, axis],
                      alpha=0.4, linewidth=3 + level * 2,
                      label=f"{100 * data.metadata['levels'][level]:g}% projected HLD")
        ax.scatter(truths[order, axis], modes[order, axis], s=16, label="Ratio estimate")
        bounds = context.problem.parameters[axis]
        ax.plot([bounds.low, bounds.high], [bounds.low, bounds.high], linestyle="--", label="Ideal")
        ax.set(xlabel=f"True {name}", ylabel=f"Estimated {name}", title=context.member["prior"])
        ax.legend(fontsize="small")
        figures[f"estimates-{name.replace(':', '-')}"] = fig
    theta, measure = candidates.array("theta"), candidates.array("log_measure")
    for saved, case in enumerate(data.array("retained_cases")[:context.config["inference"]["diagnostic_cases"]]):
        for axis, name in enumerate(context.problem.names):
            fig, ax = plt.subplots(figsize=(6, 4))
            for target in data.metadata["targets"]:
                mass = normalized_mass(data.array(f"{target}_log_score")[saved], measure)
                ax.hist(theta[:, axis], bins=context.config["plotting"]["bins"], weights=mass,
                        histtype="step", label=target)
            ax.axvline(truths[case, axis], linestyle="--", label="True")
            ax.set(xlabel=name, ylabel="Marginal probability mass per bin",
                   title=f"{context.member['prior']}, case {case}")
            ax.legend()
            figures[f"marginal-{case}-{name.replace(':', '-')}"] = fig
        # Save each two-dimensional projection separately; no opaque corner-plot dependency.
        for row in range(1, theta.shape[1]):
            for column in range(row):
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                mass = normalized_mass(data.array("ratio_log_score")[saved], measure)
                image = ax.hist2d(theta[:, column], theta[:, row], weights=mass,
                                  bins=context.config["plotting"]["bins"])
                fig.colorbar(image[3], ax=ax, label="Marginal probability mass per bin")
                ax.scatter([truths[case, column]], [truths[case, row]], marker="*", s=90, label="True")
                ax.set(xlabel=context.problem.names[column], ylabel=context.problem.names[row],
                       title=f"Normalized ratio, case {case}")
                ax.legend()
                figures[f"projection-{case}-{column}-{row}"] = fig
    return figures


def pairwise(context, dependencies):
    data = dependencies["verification_predictions"]
    exact, predicted = data.array("probe_exact_log_ratio"), data.array("probe_log_ratio")
    error = np.sqrt(np.mean((predicted - exact) ** 2, axis=1))
    figures = {}
    for name, pair in (("median", int(np.argsort(error)[len(error) // 2])), ("worst", int(np.argmax(error)))):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(exact[pair], predicted[pair], s=8, alpha=0.5)
        bounds = [min(exact[pair].min(), predicted[pair].min()), max(exact[pair].max(), predicted[pair].max())]
        ax.plot(bounds, bounds, linestyle="--")
        ax.set(xlabel="Exact log likelihood ratio", ylabel="Estimated log likelihood ratio",
               title=f"{context.member['prior']}: {name} pair (RMSE={error[pair]:.3g})")
        figures[name] = fig
    return figures


def reweighting(context, dependencies):
    samples, predictions = dependencies["verification_observations"], dependencies["verification_predictions"]
    distances = dependencies["reweighting_distances"].array("learned_w1")
    index = int(np.argmax(distances))
    source, target = samples.array("source_x")[index], samples.array("target_x")[index]
    score = predictions.array("source_log_ratio")[index]
    weights = np.exp(score - logsumexp(score))
    figures = {}
    for axis in range(source.shape[1]):
        fig, ax = plt.subplots(figsize=(6, 4))
        edges = np.histogram_bin_edges(np.concatenate((source[:, axis], target[:, axis])),
                                       bins=context.config["plotting"]["bins"])
        ax.hist(source[:, axis], bins=edges, density=True, histtype="step", label="Source")
        ax.hist(target[:, axis], bins=edges, density=True, histtype="step", label="Target")
        ax.hist(source[:, axis], weights=weights, bins=edges, density=True, histtype="step", label="Reweighted")
        ax.set(xlabel=f"x:{axis}", ylabel="Density", title=f"Worst sliced-W1 pair: {index}")
        ax.legend()
        figures[f"axis-{axis}"] = fig
    return figures


def showcase(context, dependencies):
    samples, predictions = dependencies["showcase_observations"], dependencies["showcase_predictions"]
    score = predictions.array("log_ratio")
    weights = np.exp(score - logsumexp(score))
    source, target = samples.array("source"), samples.array("target")
    figures = {}
    for axis in range(source.shape[1]):
        fig, ax = plt.subplots(figsize=(6, 4))
        edges = np.histogram_bin_edges(np.concatenate((source[:, axis], target[:, axis])),
                                       bins=context.config["verification"]["showcase"]["bins"])
        ax.hist(source[:, axis], bins=edges, density=True, histtype="step", label="Source")
        ax.hist(target[:, axis], bins=edges, density=True, histtype="step", label="Target")
        ax.hist(source[:, axis], weights=weights, bins=edges, density=True, histtype="step", label="Reweighted")
        ax.set(xlabel=f"x:{axis}", ylabel="Density", title="Fixed-parameter reweighting showcase")
        ax.legend()
        figures[f"axis-{axis}"] = fig
    return figures
