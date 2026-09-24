"""What is trained in the Gaussian study, and early validation of the training settings."""
from __future__ import annotations

import math

from nnpd import Experiment
from nnpd.core.settings import select
from nnpd.nre.distributions import IndependentPrior
from nnpd.nre.data import balanced_pairs
from nnpd.nre.models import build_network, NETWORKS
from nnpd.nre.training import fit_binary, predict_logits
from .problem import GaussianProblem
from .sampling import training_data, data_settings


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

    def make_problem(self, settings):
        return GaussianProblem(settings["problem"])

    def make_prior(self, settings, member, problem):
        return IndependentPrior(problem.parameters, settings["priors"][member["prior"]])

    def members(self, settings):
        return [{"name": f"{prior}-r{replica}", "prior": prior, "replica": replica}
                for prior in settings["cohort"]["priors"] for replica in range(settings["cohort"]["replicas"])]

    def sample_training(self, context, writer):
        return training_data(context, writer)

    def build_model(self, context):
        inputs = context.problem.observation_dim + len(context.problem.parameters)
        return build_network(inputs, context.training_settings["model"])

    def train(self, context, model):
        return fit_binary(context, model)

    def signature(self, stage, context):
        if stage == "data":
            return data_settings(context)
        return {**select(context.training_settings, "model", "training", "seed"), "member": context.member}

    def sources(self, stage):
        if stage == "data":
            return (self.sample_training, self.make_problem, self.make_prior, training_data,
                    GaussianProblem, IndependentPrior, balanced_pairs)
        return (self.build_model, self.train, build_network, fit_binary, predict_logits)

    def validate(self, settings):
        _positive_integer(settings["seed"], "seed", 0)
        problem = self.make_problem(settings)
        cohort = settings["cohort"]
        _positive_integer(cohort["replicas"], "cohort.replicas")
        if not cohort["priors"] or len(set(cohort["priors"])) != len(cohort["priors"]):
            raise ValueError("The prior cohort must be nonempty without duplicates.")
        for name in cohort["priors"]:
            if name not in settings["priors"]:
                raise ValueError(f"Unknown prior {name!r}.")
            self.make_prior(settings, {"prior": name}, problem)
        for name, value in settings["data"].items():
            _positive_integer(value, f"data.{name}", 2)
            if value % 2:
                raise ValueError(f"data.{name} must be even for exactly balanced classes.")
        model, training = settings["model"], settings["training"]
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
        for name in ("cpu_threads", "timeout_minutes"):
            _positive_integer(settings["runtime"][name], f"runtime.{name}")

    def estimate(self, settings, member):
        problem = self.make_problem(settings)
        return {"observation_dimension": problem.observation_dim, "inferred_parameters": problem.names,
                "network_inputs": len(problem.parameters) + problem.observation_dim,
                "training_rows": settings["data"]["train_size"]}
