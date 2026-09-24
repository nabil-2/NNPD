"""Training settings: what is trained. Analysis choices live in settings_analysis.py.

Choice([...]) means baseline + one-factor alternatives, NEVER a Cartesian product.
An ordinary list is one value (hidden layer widths, parameter names, cohorts, etc.).
The four training priors are an explicitly repeated comparison cohort in every run.
"""
from copy import deepcopy
from nnpd import Choice

DEFAULT_PROFILE = "default"
PROFILES = ("default", "paper", "smoke")  # settings_analysis.py defines the same profiles

SETTINGS = {
    "experiment": "gaussian:GaussianExperiment",
    "output": "outputs/default",
    "seed": 0,
    "problem": {
        "dimension": Choice([1, 2, 3, 4, 5, 6]),
        "infer": ["mean:*"],       # e.g. Choice([["mean:*"], ["mean:*", "std:1"]])
        "fixed_mean": 5.0,
        "fixed_std": 2.0,
        "fixed": {},                # e.g. {"mean:0": 4.0}, only for uninferred parameters
        "parameters": {
            "mean": {"low": 0.0, "high": 10.0, "normal_mean": 5.0, "normal_std": 4.0,
                     "exponential_rate": 0.1, "grid_step": 0.2},
            "std": {"low": 0.2, "high": 4.0, "normal_mean": 2.0, "normal_std": 0.7,
                    "exponential_rate": 0.1, "grid_step": 0.2},
        },
        "overrides": {},            # e.g. {"mean:1": {"low": 1.0, "high": 9.0}}
    },
    "cohort": {
        "priors": ["uniform", "normal", "exponential", "grid"],
        "replicas": 1,              # independent training replicas; not the 15 inference observations
    },
    "priors": {
        "uniform": {"family": "uniform"},
        "normal": {"family": "normal"},
        "exponential": {"family": "exponential"},
        "grid": {"family": "grid"},
        # A family may be overridden by inferred name:
        # "mixed": {"family": "uniform", "by_parameter": {"mean:0": "grid"}},
    },
    "data": {"train_size": 70_000, "validation_size": 15_000, "test_size": 15_000},
    "model": {
        "kind": "mlp",             # Choice(["mlp", "residual"])
        "hidden": [64, 64, 64, 64],
        "width": 64, "blocks": 3,  # residual model only
        "activation": "relu", "dropout": 0.0,
    },
    "training": {
        "epochs": 5, "batch_size": 128,  # GLOBAL batch size in DDP too
        "optimizer": "adam", "learning_rate": 0.001,
        "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.0,
        "precision": "off",         # off, float16 (CUDA), bfloat16
        "gradient_clip": None, "checkpoint": "last", "verbose": True,
        "evaluation_batch_size": 65_536,  # rows per model evaluation, also during analysis
    },
    "runtime": {                   # also used by analyze; cannot vary within one launch
        "device": "auto",          # auto, cpu, cuda, cuda:0; torchrun picks LOCAL_RANK
        "parallel": "jobs",        # jobs: parallel full runs; ddp: train each model together
        "cpu_threads": 1, "deterministic": True, "timeout_minutes": 60,
    },
}


def make_settings(profile: str = DEFAULT_PROFILE) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {PROFILES}.")
    settings = deepcopy(SETTINGS)
    settings["output"] = f"outputs/{profile}"
    if profile == "paper":
        settings["problem"]["parameters"]["mean"]["exponential_rate"] = -0.1
    if profile == "smoke":
        settings["problem"]["dimension"] = 1
        settings["data"] = {"train_size": 1024, "validation_size": 256, "test_size": 256}
        settings["model"]["hidden"] = [32, 32]
        settings["training"].update(epochs=2, evaluation_batch_size=2048)
    return settings
