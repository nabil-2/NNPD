# NNPD

## Configure an experiment. Reuse its data. Extend its analysis.

NNPD separates experiment execution from scientific choices. The reusable core
handles planning, training stages, shared artifacts, and results. The Gaussian
application implements the neural-prior-dependence example. All ordinary
experiment settings live in **`settings.py`**.

Start with [your first run](guide/quickstart.md), then learn
[how configurations and sweeps work](guide/configuration.md). To add a simulator,
prior, training procedure, metric, or plot, read the
[extension guide](EXTENDING.md).

```bash
python -m pip install -e ".[dev]"
python run.py plan --profile smoke
python run.py run --profile smoke
```

The `smoke` profile checks the complete workflow with small datasets. It is not a
scientific convergence study. The default command only prints a plan.

### Four small concepts

| Concept | What it owns |
|---|---|
| **Experiment** | The problem, prior, training data, model, and training procedure. |
| **Product** | A reusable dataset with declared dependencies and a persistent cache. |
| **Metric** | A measurement from shared products, the model, or both. |
| **Plot** | Ordinary plotting code with access to the full experiment context. |

There is no fixed number of metrics or plots. Several metrics can share a product
without resampling its data or retraining a model. Sweep alternatives vary **one
knob at a time**, not every possible combination.

```{important}
This documentation describes the code shipped in this repository. The existing
scientific notes and test report are included without rewriting their claims.
A working smoke run is not reproduction of the paper's full study; physical CUDA
validation and numerical-convergence limitations remain as documented.
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
guide/gallery
TEST_REPORT
guide/troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Documentation website

publishing
DOCS_TEST_REPORT
```

The original repository overview is also available as a
{download}`Markdown file <../README.md>`.
