"""A complete extension example. The framework, sampler and trainer are unchanged."""
import numpy as np
from matplotlib import pyplot as plt
from nnpd import Metric, Plot
from .experiment import GaussianExperiment


def median_absolute_bias(context, dependencies):
    estimates = dependencies["inference"].array("ratio_mode")
    truth = dependencies["observations"].array("truth")
    return {"median_absolute": float(np.median(np.abs(estimates - truth)))}


def parameter_count(context, dependencies):
    return {"trainable_parameters": sum(p.numel() for p in context.model.parameters() if p.requires_grad)}


def observation_histogram(context, dependencies):
    observations = dependencies["observations"].array("observations")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(observations[0, :, 0], bins=context.config["plotting"]["bins"])
    ax.set(xlabel="x:0", ylabel="Count", title=f"{type(context.model).__name__}; {context.prior.measure} prior")
    return {"first-ensemble": fig}


class ExtendedGaussian(GaussianExperiment):
    def metrics(self):
        return {**super().metrics(),
                "median_absolute_bias": Metric(median_absolute_bias, ("inference", "observations")),
                "parameter_count": Metric(parameter_count, ("model",))}

    def plots(self):
        return {**super().plots(),
                "observation_histogram": Plot(observation_histogram, ("observations", "model"))}
