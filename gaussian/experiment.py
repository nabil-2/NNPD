"""Scientific wiring and early validation; the core runner knows none of these details."""
from __future__ import annotations

import math
from nnpd import Experiment
from nnpd.core.config import select
from nnpd.nre.distributions import IndependentPrior
from nnpd.nre.data import balanced_pairs
from nnpd.nre.models import build_network, NETWORKS
from nnpd.nre.training import fit_binary, predict_logits
from .problem import GaussianProblem
from .sampling import training_data, data_settings
from .evaluation import truth_layout, candidate_layout


def _positive_integer(value, name, minimum=1):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _finite(value, name, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric.")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {minimum}.")


class GaussianExperiment(Experiment):
    name = "gaussian-prior-dependence"
    version = "1"

    def make_problem(self, config):
        return GaussianProblem(config["problem"])

    def make_prior(self, config, member, problem):
        return IndependentPrior(problem.parameters, config["priors"][member["prior"]])

    def members(self, config):
        return [{"name": f"{prior}-r{replica}", "prior": prior, "replica": replica}
                for prior in config["cohort"]["priors"] for replica in range(config["cohort"]["replicas"])]

    def sample_training(self, context, writer):
        return training_data(context, writer)

    def build_model(self, context):
        inputs = context.problem.observation_dim + len(context.problem.parameters)
        return build_network(inputs, context.config["model"])

    def train(self, context, model):
        return fit_binary(context, model)

    def signature(self, stage, context):
        if stage == "data":
            return data_settings(context)
        return {**select(context.config, "model", "training", "seed"), "member": context.member}

    def sources(self, stage):
        if stage == "data":
            return (self.sample_training, self.make_problem, self.make_prior, training_data,
                    GaussianProblem, IndependentPrior, balanced_pairs)
        return (self.build_model, self.train, build_network, fit_binary, predict_logits)

    def products(self):
        from .hooks import products
        return products()

    def metrics(self):
        from .hooks import metrics
        return metrics()

    def plots(self):
        from .hooks import plots
        return plots()

    def validate(self, config):
        _positive_integer(config["seed"], "seed", 0)
        problem = self.make_problem(config)
        cohort = config["cohort"]
        _positive_integer(cohort["replicas"], "cohort.replicas")
        if not cohort["priors"] or len(set(cohort["priors"])) != len(cohort["priors"]):
            raise ValueError("The prior cohort must be nonempty without duplicates.")
        for name in cohort["priors"]:
            if name not in config["priors"]:
                raise ValueError(f"Unknown prior {name!r}.")
            self.make_prior(config, {"prior": name}, problem)
        for name, value in config["data"].items():
            _positive_integer(value, f"data.{name}", 2)
            if value % 2:
                raise ValueError(f"data.{name} must be even for exactly balanced classes.")
        model, training = config["model"], config["training"]
        if model["kind"] not in NETWORKS or model["activation"] not in {"relu", "gelu", "tanh"}:
            raise ValueError("Unknown model kind or activation.")
        if not isinstance(model["hidden"], list) or not model["hidden"]:
            raise ValueError("model.hidden must be a nonempty literal list of widths.")
        for width in [*model["hidden"], model["width"], model["blocks"]]:
            _positive_integer(width, "model width/block count")
        _finite(model["dropout"], "model.dropout", 0)
        if model["dropout"] >= 1:
            raise ValueError("Dropout must be below 1.")
        for name in ("epochs", "batch_size", "evaluation_batch_size"):
            _positive_integer(training[name], f"training.{name}")
        for name in ("learning_rate", "epsilon"):
            _finite(training[name], f"training.{name}", 0, strict=True)
        _finite(training["weight_decay"], "training.weight_decay", 0)
        if training["gradient_clip"] is not None:
            _finite(training["gradient_clip"], "training.gradient_clip", 0, strict=True)
        if len(training["betas"]) != 2 or any(not 0 <= value < 1 for value in training["betas"]):
            raise ValueError("Adam betas must contain two values in [0,1).")
        if training["optimizer"] not in {"adam", "adamw"} or training["checkpoint"] not in {"last", "best"}:
            raise ValueError("Unknown optimizer/checkpoint policy.")
        if training["precision"] not in {"off", "float16", "bfloat16"}:
            raise ValueError("Unknown training precision.")
        cfg = config["inference"]
        for name in ("observations", "repeats", "truth_points_per_axis", "max_grid_truths", "sobol_truths",
                     "candidate_count", "candidate_points_per_axis", "max_model_evaluations",
                     "max_saved_bytes", "max_candidate_points"):
            _positive_integer(cfg[name], f"inference.{name}")
        _positive_integer(cfg["candidate_points_per_axis"], "candidate_points_per_axis", 2)
        for name in ("seed", "diagnostic_cases"):
            _positive_integer(cfg[name], f"inference.{name}", 0)
        _finite(cfg["truth_margin_fraction"], "inference.truth_margin_fraction", 0)
        if cfg["truth_margin_fraction"] > 0.5:
            raise ValueError("truth_margin_fraction removes the parameter domain.")
        if cfg["truth_design"] not in {"auto", "grid", "sobol"} or cfg["candidate_design"] not in {"grid", "sobol"}:
            raise ValueError("Unknown inference design.")
        if not cfg["targets"] or len(set(cfg["targets"])) != len(cfg["targets"]) or set(cfg["targets"]) - {"ratio", "posterior", "exact"}:
            raise ValueError("Select unique targets from ratio, posterior, exact.")
        ratio_metrics = {"bias", "coverage", "width", "resolution", "exact_reference"}
        if "ratio" not in cfg["targets"] and (ratio_metrics & set(config["metrics"]) or "inference" in config["plots"]):
            raise ValueError("Selected ratio metrics/plots require the 'ratio' inference target.")
        if cfg["retain"] not in {"diagnostic", "full"}:
            raise ValueError("inference.retain must be diagnostic or full.")
        if not cfg["levels"] or len(set(cfg["levels"])) != len(cfg["levels"]) or any(not 0 < level <= 1 for level in cfg["levels"]):
            raise ValueError("Choose unique enclosed HLD masses in (0,1].")
        verification = config["verification"]
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
        for name, value in config["plotting"].items():
            _positive_integer(value, f"plotting.{name}")
        for name in ("cpu_threads", "timeout_minutes"):
            _positive_integer(config["runtime"][name], f"runtime.{name}")

    def _needed(self, config):
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
            for name in config[category]:
                if name in hooks:
                    for dependency in hooks[name].needs:
                        visit(dependency)
        return needed

    def estimate(self, config, member):
        problem, cfg = self.make_problem(config), config["inference"]
        prior = self.make_prior(config, member, problem)
        needed, errors, warnings = self._needed(config), [], []
        p, d = len(problem.parameters), problem.observation_dim
        result = {"observation_dimension": d, "inferred_parameters": problem.names,
                  "network_inputs": p + d, "training_rows": config["data"]["train_size"],
                  "errors": errors, "warnings": warnings}
        if "inference" not in needed:
            return result
        design, truths = truth_layout(problem, cfg)
        _, count, candidate_design = candidate_layout(problem, prior, cfg)
        cases = truths * cfg["repeats"]
        evaluations = (count + 1) * cases * cfg["observations"] if any(t != "exact" for t in cfg["targets"]) else 0
        retained = cases if cfg["retain"] == "full" else min(cases, cfg["diagnostic_cases"])
        target_bytes = cases * (p + len(cfg["levels"]) * (2 * p + 3) + 2) * 8 + retained * count * 8
        saved_bytes = target_bytes * len(cfg["targets"]) + cases * (p + cfg["observations"] * d) * 4 + count * (p * 4 + 8)
        result.update(truth_design=design, inference_cases=cases, candidate_design=candidate_design,
                      candidate_count_upper_bound=count, inference_model_evaluations=evaluations,
                      estimated_inference_bytes=saved_bytes)
        if count > cfg["max_candidate_points"]:
            errors.append("candidate_count exceeds max_candidate_points")
        if evaluations > cfg["max_model_evaluations"]:
            errors.append("inference evaluations exceed max_model_evaluations")
        if saved_bytes > cfg["max_saved_bytes"]:
            errors.append("inference/observation storage estimate exceeds max_saved_bytes")
        if any(axis is not None for axis in prior.support_axes) and not cfg["native_grid_prior"] and "posterior" in cfg["targets"]:
            errors.append("a discrete posterior needs native_grid_prior=True")
        if config["inference"]["truth_design"] == "auto" and design == "sobol":
            warnings.append(f"Explicit auto rule selects {truths} Sobol truths, NOT 25**d exhaustive truths")
        if cfg["align_truths_to_grid"]:
            warnings.append("Truths explicitly snapped to each parameter's grid_step; duplicates are retained and reported")
        if count < 100 ** p:
            warnings.append("Candidate resolution may be inadequate; compare exact_reference and run a candidate-count sweep")
        return result
