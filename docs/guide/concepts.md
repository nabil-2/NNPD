# How the pieces fit together

## Framework versus application

`nnpd/core` knows how to resolve settings, execute jobs, cache artifacts, and save
results. It does not import the Gaussian application or `settings.py`.
`nnpd/nre` supplies optional neural-ratio-estimation building blocks.
`gaussian` implements the particular simulator and analysis.

| Location | Read it to understand |
|---|---|
| `settings.py` | Presets and tunable choices. |
| `nnpd/core/api.py` | The abstract experiment and hook contracts. |
| `nnpd/core/runner.py` | Planning, execution, analysis, and restoration. |
| `gaussian/experiment.py` | How the Gaussian application implements the contracts. |
| `gaussian/hooks.py` | Which products each metric and plot requires. |
| `gaussian/extensions.py` | A complete additional-metric and plot example. |

## Configuration, cohort, and job

A **configuration** is a resolved settings dictionary: every `Choice` has become
one ordinary value. A **cohort member** is a deliberate comparison within that
configuration. In the example, it identifies the training prior and training
replica. A **job** combines a configuration and cohort member.

Five one-factor-at-a-time configurations with four priors and one replica produce
twenty model jobs. This cohort multiplication does not turn the scientific sweep
into a Cartesian search over independent knobs.

## Lifecycle

```text
settings + Experiment
        |
        v
plan: resolve alternatives, validate hooks/settings, estimate workload
        |
        v
training artifact ----> model artifact
                              |
                              v
                   requested shared products
                              |
                              v
                       metrics and plots
```

The two reserved root artifacts are named `training` and `model`. The runner
creates or restores them before analysis. Other products are built lazily when a
selected metric, plot, or another product requests them. The runner rejects
unknown dependencies and cycles.

## Context and dependencies

A hook receives a `Context` and a mapping from declared dependency names to
`Artifact` objects. Builders additionally receive a `Writer`. Context supplies
`config`, `member`, `problem`, `prior`, `model`, `runtime`, `run_dir`, `store`,
`rng(...)`, and `require(...)`.

Use the dependency mapping for data that determines a cached product, and declare
that data in `Product.needs`. Access through context is flexible, but it does not
replace an accurate cache recipe. Changing an undeclared external input will not
magically invalidate a product.

Metrics return small JSON-compatible values. Products hold reusable numeric
arrays and structured metadata. Plot hooks return a mapping of filename stems to
Matplotlib figures; the runner writes PNGs and closes them.

The full [extension contract](../EXTENDING.md), [storage schema](../STORAGE.md),
and [API reference](../reference/api.md) provide implementation details.
