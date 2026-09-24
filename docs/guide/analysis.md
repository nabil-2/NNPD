# Products, metrics, and plots

The Gaussian application registers its analysis in `gaussian/analysis.py`, the
`GaussianAnalysis` class named by `"analysis"` in `settings_analysis.py`. Analysis
never trains models. A selected hook declares named products; the runner resolves
and caches those dependencies first. Which metrics and plots are computed, and
with which inference and verification settings, is set in `settings_analysis.py`.

## Shared inference

```text
observations ---+                +--> bias
               |                +--> coverage
candidates ----+--> inference ---+--> width
               |                +--> resolution
model ---------+                +--> posterior / exact_reference
                                +--> inference plots
```

`observations` saves truths and their observation ensembles. `candidates` saves
parameter points and their integration weights. `inference` saves modes,
projected joint-HLD bounds, effective sample sizes, attained masses, truth
membership, and selected raw score rows for each enabled target.

Adding a consumer does not require another copy of this shared inference dataset.
Changing its dependencies or relevant settings can create a different artifact.
For example, HLD summaries and retained scores form one inference product;
changing the HLD levels rebuilds that product rather than recomputing thresholds
from previously retained scores.

## Built-in metrics

| Registry name | Main inputs | Reported quantities |
|---|---|---|
| `classification` | `test_predictions` | Held-out classification measurements. |
| `bias` | `inference`, `observations` | Signed and absolute ratio-mode errors, RMSE, per-axis errors. |
| `coverage` | `inference`, `observations` | Projected-bound and joint-density truth membership. |
| `width` | `inference` | Mean and per-axis projected joint-HLD widths. |
| `resolution` | `inference` | Candidate count, ESS, attained mass, and reference measure. |
| `posterior` | `inference`, `observations` | Posterior bias, coverage, and width when enabled. |
| `exact_reference` | `inference`, `observations` | Analytic-likelihood comparisons on the same numerical candidates. |
| `pairwise` | Verification observations/predictions | Learned versus exact log-ratio errors across endpoint pairs. |
| `reweighting` | Reweighting distances and verification observations | Sliced Wasserstein distances, baselines, importance-weight ESS. |
| `normalization` | `normalization_predictions` | Monte Carlo mean-ratio check under each training prior's evidence. |

`posterior` and `exact_reference` return `{"enabled": False}` when their target
is disabled. The other ratio metrics require compatible inference targets.
Undefined association values use `None`, not NaN. The source API documents the
functions in `gaussian.metrics`; metric names are the registry keys above.

## Plot hooks

The built-in names are `training`, `prior`, `inference`, `pairwise`, `reweighting`,
and `showcase`. A hook may return multiple figures. In particular, inference
plots include parameter estimates and diagnostic marginal/pairwise projections.
These are examples, not a fixed publication style.

## Add analysis after training

`gaussian/extensions.py` defines `ExtendedAnalysis` with three more hooks:
`median_absolute_bias` shares inference data, `parameter_count` uses the model
directly, and `observation_histogram` accesses the observation ensemble, model
class, and prior measure. To add them to already trained models, change
`settings_analysis.py`:

```text
"analysis": "gaussian.extensions:ExtendedAnalysis",
"metrics": [..., "median_absolute_bias", "parameter_count"],
"plots": [..., "observation_histogram"],
```

and analyze again:

```bash
python run.py analyze --profile smoke
```

`analyze` never trains. It reuses the saved models and every shared product whose
settings did not change, and replaces each run's metrics and plots. The same works
for any trained profile. `notebooks/03_extensions.ipynb` analyzes again and also
computes these metrics directly on a saved run, without changing any file.

## Complete registration and extension examples

These files are included directly from the application sources:

```{literalinclude} ../../gaussian/analysis.py
:language: python
:caption: gaussian/analysis.py
```

```{literalinclude} ../../gaussian/extensions.py
:language: python
:caption: gaussian/extensions.py
```

The [extension guide](../EXTENDING.md) shows how to add a product consumed by two
new metrics, replace a prior or training function, and declare accurate cache
settings/source dependencies.
