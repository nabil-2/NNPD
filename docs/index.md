# NNPD

## Train once. Analyze as often as you like.

NNPD separates experiment execution from scientific choices, and training from
analysis. The reusable core handles planning, training, shared artifacts, and
results. The Gaussian application implements the neural-prior-dependence example.
What is trained is set in **`settings_training.py`**; what is computed from the
trained models is set in **`settings_analysis.py`**, and can be changed and
recomputed at any time without retraining.

Start with [your first run](guide/quickstart.md), then learn
[how configurations and sweeps work](guide/configuration.md). To add a simulator,
prior, training procedure, metric, or plot, read the
[extension guide](EXTENDING.md).

```bash
uv sync
source .venv/bin/activate
python run.py plan --profile smoke
python run.py run --profile smoke
```

The `smoke` profile checks the complete workflow with small datasets. It is not a
scientific convergence study. The default command only prints a plan.

### Five small concepts

| Concept | What it owns |
|---|---|
| **Experiment** | What is trained: the problem, prior, training data, model, and training procedure. |
| **Analysis** | What is computed from the trained models: products, metrics, and plots. |
| **Product** | A reusable dataset with declared dependencies and a persistent cache. |
| **Metric** | A measurement from shared products, the model, or both. |
| **Plot** | Ordinary plotting code with access to the full experiment context. |

There is no fixed number of metrics or plots. Several metrics can share a product
without resampling its data or retraining a model. Sweep alternatives vary **one
knob at a time**, not every possible combination.

```{important}
A working smoke run is not a reproduction of the paper's full study. Read the
[scientific notes](SCIENTIFIC_NOTES.md) and [testing guide](guide/testing.md) for
the implemented definitions, numerical-convergence limits, and what the tests cover.
```

```{toctree}
:maxdepth: 2
:caption: Get started

guide/quickstart
guide/concepts
guide/configuration
guide/notebooks
```

```{toctree}
:maxdepth: 2
:caption: Use and extend

guide/results
guide/analysis
EXTENDING
guide/cuda
STORAGE
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/settings
reference/cli
reference/api
SCIENTIFIC_NOTES
guide/testing
guide/troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Documentation website

publishing
```

The repository overview is also available as a
{download}`Markdown file <../README.md>`.
