"""What is computed from the trained Gaussian models. Changing it never invalidates training."""
from nnpd import Analysis, Metric, Plot, Product
from nnpd.nre.inference import ensemble_log_ratio, summarize_density
from nnpd.nre.training import predict_logits
from nnpd.nre.distributions import IndependentPrior, grid_axis
from .problem import GaussianProblem
from .experiment import _finite, _positive_integer
from . import evaluation as ev, diagnostics as dg, metrics as mt, plots as pl


class GaussianAnalysis(Analysis):
    name = "gaussian-prior-dependence"

    def products(self):
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
                                            ("verification_observations", "verification_predictions"),
                                            settings=lambda context: {}),
            "normalization_observations": Product(dg.normalization_observations, settings=dg.normalization_settings,
                                                  sources=(IndependentPrior, GaussianProblem)),
            "normalization_predictions": Product(dg.normalization_predictions, ("model", "normalization_observations"),
                                                 dg.model_evaluation_settings, prediction_sources),
            "showcase_observations": Product(dg.showcase_observations, settings=dg.verification_settings,
                                             sources=(GaussianProblem,)),
            "showcase_predictions": Product(dg.showcase_predictions, ("model", "showcase_observations"),
                                            dg.model_evaluation_settings, prediction_sources),
        }

    def metrics(self):
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

    def plots(self):
        return {
            "training": Plot(pl.training, ("model",)),
            "prior": Plot(pl.prior),
            "inference": Plot(pl.inference, ("inference", "observations", "candidates")),
            "pairwise": Plot(pl.pairwise, ("verification_predictions",)),
            "reweighting": Plot(pl.reweighting, ("verification_observations", "verification_predictions",
                                                 "reweighting_distances")),
            "showcase": Plot(pl.showcase, ("showcase_observations", "showcase_predictions")),
        }

    def validate(self, settings):
        inference = settings["inference"]
        for name in ("observations", "repeats", "truth_points_per_axis", "max_grid_truths", "sobol_truths",
                     "candidate_count", "candidate_points_per_axis", "max_model_evaluations",
                     "max_saved_bytes", "max_candidate_points"):
            _positive_integer(inference[name], f"inference.{name}")
        _positive_integer(inference["candidate_points_per_axis"], "candidate_points_per_axis", 2)
        for name in ("seed", "diagnostic_cases"):
            _positive_integer(inference[name], f"inference.{name}", 0)
        _finite(inference["truth_margin_fraction"], "inference.truth_margin_fraction", 0)
        if inference["truth_margin_fraction"] > 0.5:
            raise ValueError("truth_margin_fraction removes the parameter domain.")
        if inference["truth_design"] not in {"sobol", "grid", "size_adaptive"}:
            raise ValueError("inference.truth_design must be sobol, grid, or size_adaptive.")
        if inference["candidate_design"] not in {"grid", "sobol"}:
            raise ValueError("inference.candidate_design must be grid or sobol.")
        targets = inference["targets"]
        if not targets or len(set(targets)) != len(targets) or set(targets) - {"ratio", "posterior", "exact"}:
            raise ValueError("Select unique targets from ratio, posterior, exact.")
        ratio_metrics = {"bias", "coverage", "width", "resolution", "exact_reference"}
        if "ratio" not in targets and (ratio_metrics & set(settings["metrics"]) or "inference" in settings["plots"]):
            raise ValueError("Selected ratio metrics/plots require the 'ratio' inference target.")
        if inference["retain"] not in {"diagnostic", "full"}:
            raise ValueError("inference.retain must be diagnostic or full.")
        levels = inference["levels"]
        if not levels or len(set(levels)) != len(levels) or any(not 0 < level <= 1 for level in levels):
            raise ValueError("Choose unique enclosed HLD masses in (0,1].")
        verification = settings["verification"]
        if verification["pair_design"] not in {"grid", "uniform", "sobol"}:
            raise ValueError("verification.pair_design must be grid, uniform, or sobol.")
        if verification["normalization_design"] not in {"diagonal", "prior"}:
            raise ValueError("normalization_design must be diagonal or prior.")
        for name in ("pairs", "samples_per_endpoint", "source_samples", "target_samples", "directions",
                     "marginal_samples", "normalization_thetas"):
            _positive_integer(verification[name], f"verification.{name}")
        _positive_integer(verification["seed"], "verification.seed", 0)
        showcase = verification["showcase"]
        for name in ("source_fraction", "target_fraction"):
            if not 0 <= showcase[name] <= 1:
                raise ValueError("Showcase fractions must be in [0,1].")
        for name in ("samples", "bins"):
            _positive_integer(showcase[name], f"showcase.{name}")
        for name, value in settings["plotting"].items():
            _positive_integer(value, f"plotting.{name}")

    def _needed(self, settings):
        needed = set()
        registry = self.products()
        def visit(name):
            if name in needed:
                return
            needed.add(name)
            if name in registry:
                for dependency in registry[name].needs:
                    visit(dependency)
        for category in ("metrics", "plots"):
            hooks = getattr(self, category)()
            for name in settings[category]:
                if name in hooks:
                    for dependency in hooks[name].needs:
                        visit(dependency)
        return needed

    def estimate(self, settings, problem, prior):
        inference, errors, warnings = settings["inference"], [], []
        result = {"errors": errors, "warnings": warnings}
        if "inference" not in self._needed(settings):
            return result
        p, d = len(problem.parameters), problem.observation_dim
        design, truths = ev.truth_layout(problem, inference)
        _, count, candidate_design = ev.candidate_layout(problem, prior, inference)
        cases = truths * inference["repeats"]
        targets = inference["targets"]
        evaluations = (count + 1) * cases * inference["observations"] if any(t != "exact" for t in targets) else 0
        retained = cases if inference["retain"] == "full" else min(cases, inference["diagnostic_cases"])
        target_bytes = cases * (p + len(inference["levels"]) * (2 * p + 3) + 2) * 8 + retained * count * 8
        saved_bytes = (target_bytes * len(targets) + cases * (p + inference["observations"] * d) * 4
                       + count * (p * 4 + 8))
        result.update(truth_design=design, inference_cases=cases, candidate_design=candidate_design,
                      candidate_count_upper_bound=count, inference_model_evaluations=evaluations,
                      estimated_inference_bytes=saved_bytes)
        if count > inference["max_candidate_points"]:
            errors.append(f"{count:,} candidates exceed max_candidate_points ({inference['max_candidate_points']:,}); "
                          "reduce candidate_count or candidate_points_per_axis")
        if evaluations > inference["max_model_evaluations"]:
            errors.append(f"{evaluations:,} inference model evaluations exceed max_model_evaluations "
                          f"({inference['max_model_evaluations']:,}); reduce the truths ({design} design), "
                          "repeats, observations or candidate_count")
        if saved_bytes > inference["max_saved_bytes"]:
            errors.append(f"{saved_bytes:,} saved inference bytes exceed max_saved_bytes "
                          f"({inference['max_saved_bytes']:,}); reduce the truths, repeats, observations "
                          "or retained scores")
        if (any(axis is not None for axis in prior.support_axes) and not inference["native_grid_prior"]
                and "posterior" in targets):
            errors.append("a discrete posterior needs native_grid_prior=True")
        if inference["truth_design"] == "size_adaptive" and design == "sobol":
            warnings.append(f"size_adaptive selects {truths} Sobol truths because the "
                            f"{inference['truth_points_per_axis']}**{p} grid exceeds max_grid_truths")
        if inference["align_truths_to_grid"]:
            warnings.append("Truths explicitly snapped to each parameter's grid_step; duplicates are retained and reported")
        if count < 100 ** p:
            warnings.append("Candidate resolution may be inadequate; compare exact_reference and run a candidate-count sweep")
        return result
