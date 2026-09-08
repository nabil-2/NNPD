"""Small, independent metrics. None trains a model or regenerates observations."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
from nnpd.nre.training import classification


def classifier(context, dependencies):
    data = dependencies["test_predictions"]
    return classification(data.array("labels"), data.array("logits"))


def _bias(data, truths, target):
    error = data.array(f"{target}_mode") - truths
    return {"mean_signed": float(error.mean()), "mean_absolute": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "per_axis_signed": error.mean(axis=0).tolist(),
            "per_axis_absolute": np.abs(error).mean(axis=0).tolist()}


def _coverage(data, truths, target):
    inside = ((truths[:, None, :] >= data.array(f"{target}_lower")) &
              (truths[:, None, :] <= data.array(f"{target}_upper")))
    return {"levels": data.metadata["levels"], "projected_mean": inside.mean(axis=(0, 2)).tolist(),
            "projected_per_axis": inside.mean(axis=0).tolist(),
            "joint_density": data.array(f"{target}_joint_contains_truth").mean(axis=0).tolist(),
            "cases": len(truths), "interpretation": "empirical true-parameter membership on supplied truths"}


def _width(data, target):
    width = data.array(f"{target}_upper") - data.array(f"{target}_lower")
    return {"levels": data.metadata["levels"], "mean": width.mean(axis=(0, 2)).tolist(),
            "per_axis": width.mean(axis=0).tolist()}


def bias(context, dependencies):
    return _bias(dependencies["inference"], dependencies["observations"].array("truth"), "ratio")


def coverage(context, dependencies):
    return _coverage(dependencies["inference"], dependencies["observations"].array("truth"), "ratio")


def width(context, dependencies):
    return _width(dependencies["inference"], "ratio")


def resolution(context, dependencies):
    data = dependencies["inference"]
    ess = data.array("ratio_ess")
    count = data.metadata["candidate_count"]
    return {"candidate_count": count, "candidate_ess_min": float(ess.min()),
            "candidate_ess_median": float(np.median(ess)), "candidate_ess_fraction_median": float(np.median(ess / count)),
            "mean_enclosed_mass": data.array("ratio_enclosed_mass").mean(axis=0).tolist(),
            "base_measure": data.metadata["base_measure"],
            "warning": "Small candidate ESS can signal inadequate resolution; increase candidates and compare exact_reference."}


def posterior(context, dependencies):
    data = dependencies["inference"]
    if "posterior" not in data.metadata["targets"]:
        return {"enabled": False}
    truths = dependencies["observations"].array("truth")
    return {"enabled": True, "bias": _bias(data, truths, "posterior"),
            "coverage": _coverage(data, truths, "posterior"), "width": _width(data, "posterior")}


def exact_reference(context, dependencies):
    data = dependencies["inference"]
    if "exact" not in data.metadata["targets"]:
        return {"enabled": False}
    truths = dependencies["observations"].array("truth")
    return {"enabled": True, "bias": _bias(data, truths, "exact"),
            "coverage": _coverage(data, truths, "exact"), "width": _width(data, "exact"),
            "learned_vs_exact_mode_rmse": float(np.sqrt(np.mean((data.array("ratio_mode") -
                                                                  data.array("exact_mode")) ** 2))),
            "caveat": "Exact likelihood, but same numerical candidate approximation; not an analytic continuous HLD."}


def _association(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def pairwise(context, dependencies):
    data = dependencies["verification_predictions"]
    error = data.array("probe_log_ratio") - data.array("probe_exact_log_ratio")
    rmse = np.sqrt(np.mean(error ** 2, axis=1))
    distance = np.linalg.norm(np.diff(dependencies["verification_observations"].array("pairs"), axis=1)[:, 0], axis=1)
    return {"rmse_pooled": float(np.sqrt(np.mean(error ** 2))), "mae_pooled": float(np.abs(error).mean()),
            "max_absolute": float(np.abs(error).max()), "per_pair_rmse": rmse.tolist(),
            "median_pair_rmse": float(np.median(rmse)), "worst_pair_rmse": float(rmse.max()),
            "spearman_distance_rmse": _association(distance, rmse),
            "distance_definition": "Euclidean distance in raw inferred parameter coordinates"}


def reweighting(context, dependencies):
    data = dependencies["reweighting_distances"]
    pairs = dependencies["verification_observations"].array("pairs")
    distance = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1)
    learned = data.array("learned_w1")
    return {"median_sliced_w1": float(np.median(learned)), "worst_sliced_w1": float(learned.max()),
            "median_exact_sliced_w1": float(np.median(data.array("exact_w1"))),
            "median_unweighted_sliced_w1": float(np.median(data.array("unweighted_w1"))),
            "median_ess": float(np.median(data.array("ess"))),
            "median_ess_fraction": float(np.median(data.array("ess")) / data.metadata["source_samples_per_pair"]),
            "spearman_distance_w1": _association(distance, learned)}


def normalization(context, dependencies):
    scores = dependencies["normalization_predictions"].array("log_ratio")
    means = logsumexp(scores, axis=1) - np.log(scores.shape[1])
    return {"log_mean_ratio_per_theta": means.tolist(), "expected_log_mean": 0.0,
            "mean_absolute_log_error": float(np.mean(np.abs(means))),
            "caveat": "Monte Carlo check under each model's own training-prior evidence; finite-sample error remains."}
