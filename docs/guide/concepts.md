# How the pieces fit together

## Framework versus application

`nnpd/core` knows how to resolve settings, execute jobs, cache artifacts, and save
results. It does not import the Gaussian application or the settings files.
`nnpd/nre` supplies optional neural-ratio-estimation building blocks.
`gaussian` implements the particular simulator and analysis.

## Training versus analysis

Everything is split into what is trained and what is computed from the trained
models. Each side has a settings file and an implementation class:

| | Settings file | Class | Owns |
|---|---|---|---|
| Training | `settings_training.py` | `Experiment`, named by `"experiment"` | Problem, prior, training data, model, training procedure |
| Analysis | `settings_analysis.py` | `Analysis`, named by `"analysis"` | Products, metrics, plots |

`train` uses only the training side. `analyze` uses the analysis side on the saved
models of the latest training and never trains. Changing the analysis therefore
never invalidates a model, and an analysis can be repeated or changed at any time.

| Location | Read it to understand |
|---|---|
| `settings_training.py`, `settings_analysis.py` | Presets and tunable choices. |
| `nnpd/core/api.py` | The `Experiment`, `Analysis`, and hook contracts. |
| `nnpd/core/runner.py` | Planning, training, analysis, and restoration. |
| `gaussian/experiment.py` | How the Gaussian application is trained. |
| `gaussian/analysis.py` | Which products, metrics, and plots it computes, and which products each requires. |
| `gaussian/extensions.py` | A complete additional-metric and plot example. |

## Configuration, cohort, and job

A **configuration** is a resolved training settings dictionary: every `Choice`
has become one ordinary value. A **cohort member** is a deliberate comparison within that
configuration. In the example, it identifies the training prior and training
replica. A **job** combines a configuration and cohort member.

Five one-factor-at-a-time configurations with four priors and one replica produce
twenty model jobs. This cohort multiplication does not turn the scientific sweep
into a Cartesian search over independent knobs.

## Lifecycle

```text
settings_training.py + Experiment          settings_analysis.py + Analysis
        |                                            |
        v                                            |
plan: resolve alternatives, validate, estimate the analysis workload
        |                                            |
        v                                            |
train: training artifact ----> model artifact        |
                                     |               |
                                     v               v
                  analyze: requested shared products (never trains)
                                     |
                                     v
                             metrics and plots
```

The two reserved root artifacts are named `training` and `model`. `train` creates
them; `analyze` opens them from the saved runs. Other products are built lazily when a
selected metric, plot, or another product requests them. The runner rejects
unknown dependencies and cycles.

## Context and dependencies

A hook receives a `Context` and a mapping from declared dependency names to
`Artifact` objects. Builders additionally receive a `Writer`. Context supplies
`training_settings`, `analysis_settings`, `experiment`, `analysis`, `member`,
`problem`, `prior`, `model`, `runtime`, `run_dir`, `store`, `rng(...)`, and
`require(...)`.

Use the dependency mapping for data that determines a cached product, and declare
that data in `Product.needs`. Access through context is flexible, but it does not
replace an accurate cache recipe. Changing an undeclared external input will not
magically invalidate a product.

Metrics return small JSON-compatible values. Products hold reusable numeric
arrays and structured metadata. Plot hooks return a mapping of filename stems to
Matplotlib figures; the runner writes PNGs and closes them.

The full [extension contract](../EXTENDING.md), [storage schema](../STORAGE.md),
and [API reference](../reference/api.md) provide implementation details.
