"""Edit all experiment choices here. No other file needs editing for ordinary runs.

Choice([...]) means baseline + one-factor alternatives, NEVER a Cartesian product.
An ordinary list is one value (hidden layer widths, parameter names, cohorts, etc.).
The four training priors are an explicitly repeated comparison cohort in every run.
"""
from copy import deepcopy
from nnpd import Choice

DEFAULT_PROFILE = "legacy"

SETTINGS = {
    "application": "gaussian:GaussianExperiment",
    "output": "outputs/legacy",
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
        "evaluation_batch_size": 65_536,
    },
    "inference": {
        "seed": 2026,
        "observations": 15,         # one common theta for these independent observations
        "repeats": 1,               # independently repeated observation ensembles per truth
        "truth_design": "auto",    # grid, sobol, auto; auto rule is recorded in metadata
        "truth_points_per_axis": 25, "truth_margin_fraction": 0.1,
        "max_grid_truths": 1_000_000, "sobol_truths": 65_536,
        "align_truths_to_grid": True,
        "candidate_design": "sobol",  # sobol or grid (grid can become enormous)
        "candidate_count": 16_384,
        "candidate_points_per_axis": 200,  # used only for candidate_design='grid'
        "native_grid_prior": True,  # discrete prior uses counting measure on its support
        "levels": [0.68, 0.95],     # ENCLOSED mass, not tail probabilities
        "targets": ["ratio", "posterior", "exact"],
        "retain": "diagnostic",    # diagnostic: a few cases; full: all candidate scores
        "diagnostic_cases": 3,
        "max_model_evaluations": 2_000_000_000_000,
        "max_saved_bytes": 8_000_000_000,
        "max_candidate_points": 2_000_000,
    },
    "verification": {
        "seed": 2027, "pairs": 256, "pair_design": "grid", "samples_per_endpoint": 512,
        "source_samples": 2048, "target_samples": 2048, "directions": 32,
        "marginal_samples": 1000, "normalization_thetas": 5,
        "normalization_design": "diagonal", "normalization_on_grid": True,
        "showcase": {"source_fraction": 0.3, "target_fraction": 0.7,
                     "samples": 1_000_000, "bins": 1000},
    },
    "metrics": ["classification", "bias", "coverage", "width", "resolution",
                "posterior", "exact_reference", "pairwise", "reweighting", "normalization"],
    "plots": ["training", "prior", "inference", "pairwise", "reweighting", "showcase"],
    "plotting": {"dpi": 140, "bins": 100, "prior_points": 200_000, "max_inference_cases": 200},
    "runtime": {
        "device": "auto",          # auto, cpu, cuda, cuda:0; torchrun picks LOCAL_RANK
        "parallel": "jobs",        # jobs: parallel full runs; ddp: train each model together
        "cpu_threads": 1, "deterministic": True, "timeout_minutes": 60,
    },
}

# Presets are here, not hidden inside the framework. Scientific differences are documented.
PROFILES = ("legacy", "paper", "smoke", "mixed", "extended", "smoke_ddp")


def make_config(profile: str = DEFAULT_PROFILE) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {PROFILES}.")
    cfg = deepcopy(SETTINGS)
    cfg["output"] = f"outputs/{profile}"
    if profile == "paper":
        cfg["problem"]["parameters"]["mean"]["exponential_rate"] = -0.1
        cfg["inference"]["truth_margin_fraction"] = 0.0
        cfg["inference"]["align_truths_to_grid"] = False
        cfg["inference"]["truth_design"] = "grid"  # no silent replacement of 25**d truths
        cfg["inference"]["targets"] = ["ratio", "exact"]
        cfg["inference"]["native_grid_prior"] = False  # continuous domain in paper Eq. 4
    if profile in {"smoke", "mixed", "extended", "smoke_ddp"}:
        cfg["problem"]["dimension"] = 1
        cfg["data"] = {"train_size": 1024, "validation_size": 256, "test_size": 256}
        cfg["model"]["hidden"] = [32, 32]
        cfg["training"].update(epochs=2, evaluation_batch_size=2048)
        cfg["inference"].update(truth_design="grid", truth_points_per_axis=7,
                                candidate_count=256, diagnostic_cases=2)
        cfg["verification"].update(pairs=4, samples_per_endpoint=32, source_samples=64,
                                   target_samples=64, directions=4, marginal_samples=128,
                                   normalization_thetas=4)
        cfg["verification"]["showcase"].update(samples=256, bins=30)
        cfg["plotting"].update(prior_points=1000, bins=30)
    if profile == "mixed":
        cfg["problem"]["dimension"] = 2
        cfg["problem"]["infer"] = Choice([["mean:*"], ["mean:*", "std:1"]])
        cfg["model"]["kind"] = Choice(["mlp", "residual"])
        cfg["training"]["learning_rate"] = Choice([0.001, 0.01])
        cfg["cohort"]["priors"] = ["uniform", "normal"]
        cfg["inference"].update(truth_points_per_axis=3, candidate_count=512)
    if profile == "extended":
        cfg["output"] = "outputs/smoke"  # same cache: demonstrate adding metrics without retraining
        cfg["application"] = "gaussian.extensions:ExtendedGaussian"
        cfg["metrics"] += ["median_absolute_bias", "parameter_count"]
        cfg["plots"] += ["observation_histogram"]
    if profile == "smoke_ddp":
        cfg["runtime"]["parallel"] = "ddp"
        cfg["cohort"]["priors"] = ["uniform"]
    return cfg
