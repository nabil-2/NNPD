"""Register analysis here. Adding metrics or plots does NOT invalidate training."""
from nnpd import Product, Metric, Plot
from nnpd.nre.inference import ensemble_log_ratio, summarize_density
from nnpd.nre.training import predict_logits
from nnpd.nre.distributions import IndependentPrior, grid_axis
from .problem import GaussianProblem
from . import evaluation as ev, diagnostics as dg, metrics as mt, plots as pl


def products():
    prediction_sources = (predict_logits, GaussianProblem)
    return {
        "observations": Product(ev.inference_observations, settings=ev.truth_settings,
                                sources=(GaussianProblem, grid_axis)),
        "candidates": Product(ev.candidates, settings=ev.candidate_settings, sources=(grid_axis,)),
        "inference": Product(ev.inference, ("model", "observations", "candidates"), ev.inference_settings,
                             (ensemble_log_ratio, summarize_density, IndependentPrior, *prediction_sources)),
        "test_predictions": Product(dg.test_predictions, ("model", "training"), dg.model_evaluation_settings,
                                    prediction_sources),
        "verification_observations": Product(dg.verification_observations, settings=dg.verification_settings,
                                             sources=(GaussianProblem, ev.unit_sobol, grid_axis)),
        "verification_predictions": Product(dg.verification_predictions, ("model", "verification_observations"),
                                            dg.model_evaluation_settings, prediction_sources),
        "reweighting_distances": Product(dg.reweighting_distances,
                                        ("verification_observations", "verification_predictions"), settings=lambda c: {}),
        "normalization_observations": Product(dg.normalization_observations,
                                              settings=lambda c: {**dg.verification_settings(c), "member": c.member,
                                                                  "prior": c.config["priors"][c.member["prior"]]},
                                              sources=(IndependentPrior, GaussianProblem)),
        "normalization_predictions": Product(dg.normalization_predictions, ("model", "normalization_observations"),
                                             dg.model_evaluation_settings, prediction_sources),
        "showcase_observations": Product(dg.showcase_observations, settings=dg.verification_settings,
                                         sources=(GaussianProblem,)),
        "showcase_predictions": Product(dg.showcase_predictions, ("model", "showcase_observations"),
                                        dg.model_evaluation_settings, prediction_sources),
    }


def metrics():
    inference_dependencies = ("inference", "observations")
    return {
        "classification": Metric(mt.classifier, ("test_predictions",)),
        "bias": Metric(mt.bias, inference_dependencies),
        "coverage": Metric(mt.coverage, inference_dependencies),
        "width": Metric(mt.width, ("inference",)),
        "resolution": Metric(mt.resolution, ("inference",)),
        "posterior": Metric(mt.posterior, inference_dependencies),
        "exact_reference": Metric(mt.exact_reference, inference_dependencies),
        "pairwise": Metric(mt.pairwise, ("verification_predictions", "verification_observations")),
        "reweighting": Metric(mt.reweighting, ("reweighting_distances", "verification_observations")),
        "normalization": Metric(mt.normalization, ("normalization_predictions",)),
    }


def plots():
    return {
        "training": Plot(pl.training, ("model",)),
        "prior": Plot(pl.prior),
        "inference": Plot(pl.inference, ("inference", "observations", "candidates")),
        "pairwise": Plot(pl.pairwise, ("verification_predictions",)),
        "reweighting": Plot(pl.reweighting, ("verification_observations", "verification_predictions", "reweighting_distances")),
        "showcase": Plot(pl.showcase, ("showcase_observations", "showcase_predictions")),
    }
