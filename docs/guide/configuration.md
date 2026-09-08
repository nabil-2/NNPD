# Configuration and one-factor sweeps

## One settings file

`settings.py` defines `SETTINGS`, `DEFAULT_PROFILE`, and `make_config(profile)`.
The function deep-copies the main dictionary and applies explicit preset
adjustments. Values set later by a preset override corresponding values in
`SETTINGS`; edit the relevant part of this same file.

The [settings reference](../reference/settings.md) displays the complete original
file directly, so names and default values are not maintained in a separate copy.
Application and settings modules are trusted Python, not sandboxed configuration.

## Baseline plus one changed knob

Use `Choice` to mark alternatives:

```python
from nnpd import Choice, expand_sweep

variants = expand_sweep({
    "data": {"train_size": Choice([50_000, 100_000, 150_000])},
    "training": {"learning_rate": Choice([0.1, 0.01, 0.001])},
})
assert len(variants) == 5
```

| Training size | Learning rate | Changed setting |
|---:|---:|---|
| 50,000 | 0.1 | Baseline |
| 100,000 | 0.1 | `data.train_size` |
| 150,000 | 0.1 | `data.train_size` |
| 50,000 | 0.01 | `training.learning_rate` |
| 50,000 | 0.001 | `training.learning_rate` |

The first option is always the baseline. Every other option changes just its own
knob against that baseline. Repeated resolved configurations are removed.
Dictionary insertion order determines presentation order, not configuration
identity. There is no hidden Cartesian-product mode.

## Literal lists and atomic choices

An ordinary list is one value. For example, `[64, 64]` specifies two hidden
layers. To compare complete architectures, wrap the alternatives:

```python
Choice([[64, 64, 64, 64], [128, 128]])
```

Options may also be complete dictionaries. Use an atomic dictionary option when
several dependent settings must change together; do not nest a `Choice` inside
another option or inside a literal list. That nesting is rejected.

## Change the model or the inferred parameters

The following are **edits in `settings.py`**, not additional command-line flags:

```python
# Values for problem.dimension, problem.infer, and model.kind, respectively:
2
Choice([["mean:*"], ["mean:*", "std:1"]])
Choice(["mlp", "residual"])
```

`mean:*` means all observation means. `std:1` adds the standard deviation of the
second observation dimension; indices start at zero. At observation dimension
two, these parameter sets have two and three inferred coordinates, giving model
input sizes four and five. NNPD derives the input size.

The `mixed` preset implements this comparison together with a learning-rate
alternative. It has four resolved configurations and two prior cohort members,
for eight model jobs. Each alternative is validated against the baseline; a
`std:1` alternative is not valid with a one-dimensional baseline.

Unselected likelihood parameters stay fixed at `fixed_mean`, `fixed_std`, or
named entries in `problem.fixed`. They are **not marginalized nuisance
parameters**. Parameter-family bounds live in `problem.parameters`; selected
coordinate overrides live in `problem.overrides`.

## Priors and three different kinds of repetition

`cohort.priors` is normally a literal list repeated within every configuration.
To make the prior itself an OFAT alternative, use:

```python
Choice([["uniform"], ["normal"], ["exponential"], ["grid"]])
```

as the value of `cohort.priors`.

| Setting | What repeats |
|---|---|
| `cohort.replicas` | Independent training data/models for a prior. |
| `inference.repeats` | Independent observation ensembles at each truth. |
| `inference.observations` | Observations sharing one truth within each ensemble. |

These settings are not interchangeable.

## Fixed runtime and compatible analysis choices

`runtime` must be fixed within one execution launch. Compare CPU/CUDA or
`jobs`/`ddp` through separate launches. Inference targets, selected metrics, and
plots must also be compatible: for example, bias and coverage require the `ratio`
target. Planning checks every scientific alternative before training begins;
backend availability is checked when the runtime is created.

A useful workflow is: edit the settings, print the plan, inspect every workload's
warnings/errors, and only then execute the desired stage.
