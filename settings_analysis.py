"""Analysis settings: what is computed from the trained models. Analysis never trains.

A profile here analyzes the models trained with the same profile in settings_training.py.
Change anything and run `analyze` again: the saved models are reused, and the latest
analysis replaces the previous one. Choice is not allowed here; sweeps are for training.
"""
from copy import deepcopy

PROFILES = ("default", "paper", "smoke")  # the same profiles as settings_training.py

SETTINGS = {
    "analysis": "gaussian:GaussianAnalysis",
    "inference": {
        "seed": 2026,
        "observations": 15,         # one common theta for these independent observations
        "repeats": 1,               # independently repeated observation ensembles per truth
        "truth_design": "sobol",   # sobol; grid: truth_points_per_axis**p; size_adaptive: grid up to max_grid_truths, else sobol
        "sobol_truths": 1_024,      # Sobol truths at p=1 inferred parameter; doubles with each additional one
        "truth_points_per_axis": 25, "max_grid_truths": 1_000_000,  # grid and size_adaptive only
        "truth_margin_fraction": 0.1,
        "align_truths_to_grid": True,
        "candidate_design": "sobol",  # sobol or grid (grid can become enormous)
        "candidate_count": 16_384,
        "candidate_points_per_axis": 200,  # used only for candidate_design='grid'
        "native_grid_prior": True,  # discrete prior uses counting measure on its support
        "levels": [0.68, 0.95],     # ENCLOSED mass, not tail probabilities
        "targets": ["ratio", "posterior", "exact"],
        "retain": "diagnostic",    # diagnostic: a few cases; full: all candidate scores
        "diagnostic_cases": 3,
        # Analysis limits: run/analyze refuse up front if any trained model exceeds them.
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
}


def make_settings(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {PROFILES}.")
    settings = deepcopy(SETTINGS)
    if profile == "paper":
        settings["inference"]["truth_margin_fraction"] = 0.0
        settings["inference"]["align_truths_to_grid"] = False
        settings["inference"]["truth_design"] = "size_adaptive"  # exact 25**p grid while feasible (p <= 4)
        settings["inference"]["targets"] = ["ratio", "exact"]
        settings["inference"]["native_grid_prior"] = False  # continuous domain in paper Eq. 4
    if profile == "smoke":
        settings["inference"].update(truth_design="grid", truth_points_per_axis=7,
                                     candidate_count=256, diagnostic_cases=2)
        settings["verification"].update(pairs=4, samples_per_endpoint=32, source_samples=64,
                                        target_samples=64, directions=4, marginal_samples=128,
                                        normalization_thetas=4)
        settings["verification"]["showcase"].update(samples=256, bins=30)
        settings["plotting"].update(prior_points=1000, bins=30)
    return settings
