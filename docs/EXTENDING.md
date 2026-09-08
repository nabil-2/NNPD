# Extending the framework

## The smallest useful abstraction

There is one abstract experiment, plus three small hook types. The runner knows
about dependencies and files, not Gaussian means, a specific network, or metric
names. Subclass `nnpd.Experiment` and implement:

| Method | Responsibility |
|---|---|
| `make_problem(config)` | Construct the simulator/problem object. |
| `make_prior(config, member, problem)` | Construct the prior and its sampler. |
| `sample_training(context, writer)` | Write the model's required training data; return JSON metadata. |
| `build_model(context)` | Return a new untrained `torch.nn.Module`. |
| `train(context, model)` | Train that model in place; return JSON-compatible history. |

Optional methods supply validation, workload estimates, cohort members, cache
signatures/source dependencies, and the product/metric/plot registries. A fixed
comparison cohort defaults to one empty member. The core framework does not
require using the bundled NRE trainer or even using a likelihood simulator class.
`tests/test_pipeline.py::IndependentExample` demonstrates a non-Gaussian
implementation with its own data schema and training function.

For a new application, create an importable Python package and set, for example,
`"application": "my_application:MyExperiment"` in `settings.py`. The class must
have a no-argument constructor; scientific choices belong in the supplied config.
The common runner settings are `output`, `runtime`, `metrics`, `plots` and
`plotting.dpi`; other settings are owned by the application. The generic seed
helper uses `seed` when present, otherwise zero.

## A new simulator or prior

For an NRE application, `Simulator.sample(theta, rng)` returns one observation per
parameter row, preserving leading dimensions. `parameters` describes the ordered
inferred coordinates; `observation_dim` is the number of observation features.
The optional `log_likelihood(x, theta)` is only needed for analytic diagnostic
products, not for classification training.

`Prior.sample(size, rng)` and `Prior.log_prob(theta)` are separate replaceable
methods. The bundled `IndependentPrior` defines both exactly for its four bounded
families. A correlated/non-factorized prior should subclass `Prior` and implement
its own density and sampler instead of pretending to be independent. Its
`parameters` must match the simulator's ordered inferred coordinates.

`support_axes` is a tuple: `None` for a continuous coordinate, an array of allowed
values for a discrete coordinate. This determines the reference measure for the
bundled candidate integration. Mixed continuous/discrete products are supported.
More general constrained supports or importance proposals need a custom candidate
product with correct quadrature/proposal weights; the rectangular design is not
a universal integrator.

The Gaussian example separates likelihood parameters from inferred parameters.
`mean:*`, `std:*`, `mean:j`, and `std:j` are conveniences in that application,
not syntax hard-coded in the framework. Arbitrary selected subsets have automatic
network input sizes. Unselected coordinates remain fixed; integrating nuisance
parameters would be a different sampling/inference implementation.

## A shared post-training dataset

A `Product` has a builder, named dependencies, a settings selector, optional source
dependencies, and an explicit version. The builder receives a `Context`, a mapping
of already-built dependency artifacts, and a `Writer`. It returns JSON metadata.
For example, put this in a normal Python module:

```python
import numpy as np
from nnpd import Product, Metric
from gaussian import GaussianExperiment


def build_absolute_errors(context, dependencies, writer):
    truth = dependencies["observations"].array("truth")
    estimate = dependencies["inference"].array("ratio_mode")
    writer.array("absolute_error", np.abs(estimate - truth))
    return {"parameters": list(context.problem.names)}


def mean_error(context, dependencies):
    return float(dependencies["absolute_errors"].array("absolute_error").mean())


def median_error(context, dependencies):
    return float(np.median(dependencies["absolute_errors"].array("absolute_error")))


class MyExperiment(GaussianExperiment):
    def products(self):
        return {
            **super().products(),
            "absolute_errors": Product(
                build_absolute_errors,
                needs=("observations", "inference"),
                settings=lambda context: {},  # dependencies completely determine this dataset
                version="1",
            ),
        }

    def metrics(self):
        return {
            **super().metrics(),
            "custom_mean": Metric(mean_error, needs=("absolute_errors",)),
            "custom_median": Metric(median_error, needs=("absolute_errors",)),
        }
```

Select `custom_mean` and `custom_median` in the settings file's `metrics` list.
Both consume the same persisted absolute-error data. The model and the shared
inference product are not retrained/recomputed just because another consumer was
registered. If no selected hook needs a product, it is not built.

For large arrays, use `writer.allocate(name, shape, dtype)` to get a writable
NumPy memory map and fill it in chunks. `writer.array` writes an ordinary numeric
array; `writer.json` writes small structured metadata. Object arrays are rejected.
Ragged data can be represented by flat numeric arrays plus offsets.

## Cache correctness is part of the hook contract

`Product.settings(context)` must include **every setting read by its builder**
that is not already captured in a dependency artifact. The default is the entire
configuration: conservative and safe, but less reusable. Reduce it only when the
builder truly depends on fewer fields. Include explicit dataset-release IDs or
file checksums for external inputs.

The complete source modules containing builders and declared `sources` are
hashed, together with entry-point identity, an explicit version, and Python,
NumPy, SciPy and PyTorch versions. Declare helpers from other modules:

```python
Product(builder, needs=("model",), sources=(external_helper, CustomSimulator))
```

This is intentionally not a magical recursive import tracker. Dynamically loaded
code, external files, service responses and simulator releases are **not** inferred.
List their source dependencies or change the explicit version/config identifier.
The experiment's `sources("data")`, `sources("model")` and
`signature(stage, context)` follow the same rule for root datasets and checkpoints.
Changing a metric or a plotting module does not belong in a model fingerprint.

## Metrics that read the model directly

A metric may declare `needs=("model",)` and use `context.model`; it does not need
an intermediate product. It may also combine model access with any number of
saved datasets. See `parameter_count` in `gaussian/extensions.py`.

Metrics return finite JSON-compatible values: numbers, booleans, strings, lists
and dictionaries, or NumPy values that can be converted to those. Use `None` for
an undefined quantity, not NaN. Large arrays belong in products, not in metric
JSON. The runner writes both one JSON per metric and a combined `metrics.json`.

## Plot anything

A plot hook receives the same full context and dependency mapping as a metric.
Return a dictionary of descriptive filename stems to Matplotlib figures. One hook
can return any number of figures. The runner saves PNGs and closes the figures.
You can call the hook directly from Python/notebooks and use `figure.savefig(...)`
for another format or display the figure interactively. Source data, raw scores,
simulator sampling, prior evaluation and model calls are all available.

`gaussian/extensions.py::observation_histogram` is a complete working example
accessing shared observations, the model class and the prior measure. The built-in
plots are examples, not a required visual style or a restrictive plotting API.

## Adding a model or replacing training

The bundled classifier contract is a tensor `(batch, observation_dim + parameter_dim)`
with observations first, parameters second, returning one **logit** per row. Do not
add a sigmoid when using `fit_binary`; sigmoid probabilities can be computed for
presentation separately. For another architecture, add a factory to `NETWORKS`
in `nnpd/nre/models.py`, or override `build_model` and the application's validation.
Factories receive `(input_size, model_settings)`.

To change optimization, return a different trainer from the experiment's `train`
method; the orchestration remains unchanged. The existing trainer supports Adam,
AdamW, best/last checkpoints, gradient clipping and optional AMP. A custom trainer
must honor the runtime's distributed contract itself. Merely subclassing
`Experiment` does not automatically make an arbitrary training algorithm DDP-safe.
The built-in trainer assumes the model has no train-time cross-example behavior
such as unsynchronized batch normalization.

Only complete trained state dictionaries are persisted. Mid-epoch optimizer,
scheduler and gradient-scaler continuation is not implemented. This keeps the
training interface small; add an explicit checkpoint product/trainer if that form
of fault tolerance is needed.

## Reload and compare results

```python
from nnpd import load_experiment, restore
from nnpd.results import runs, export_csv, comparison_plot

records = runs("outputs/smoke")
record = records[0]
experiment = load_experiment(record["config"]["application"])
context = restore(record["path"], experiment, device="cpu")
model = context.model
observations = context.require("observations").array("observations")
summary = context.require("inference")

export_csv("outputs/smoke", "outputs/summary.csv", {
    "prior": "member.prior",
    "dimension": "config.problem.dimension",
    "absolute_bias": "metrics.bias.mean_absolute",
    "coverage_68": "metrics.coverage.projected_mean.0",
})
```

`restore` reloads data/weights without training. Requiring a derived product checks
its current recipe; changed analysis code may rebuild that product. Use the saved
source snapshot and compatible package versions for an old-analysis reproduction.
Or inspect its artifact arrays directly using the paths in `artifacts.json`.
Filter `records` before plotting to keep different OFAT axes, replicas or backends
from being accidentally pooled as though they were the same experiment.
