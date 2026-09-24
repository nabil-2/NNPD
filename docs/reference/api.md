# Python API

The pages below are generated from the **actual Python sources**, including
signatures, abstract methods, dataclass fields, and existing docstrings. AutoAPI
parses the files without importing the application. Installing CUDA, PyTorch, or
NNPD itself is not required to build these pages.

## Start with the extension surface

| Module | Main responsibilities |
|---|---|
| `nnpd.core.api` | `Experiment`, `Analysis`, `Context`, `Product`, `Metric`, `Plot`, `load_experiment`, `load_analysis`. |
| `nnpd.core.settings` | `Choice`, `Variant`, `expand_sweep`, `load_settings` and selectors. |
| `nnpd.core.runner` | `plan`, `execute`, `restore`, `analyze_run`, `expand_jobs`, `verify_store`. |
| `nnpd.core.storage` | `Store`, `Artifact`, `Writer`, manifests and atomic writes. |
| `nnpd.core.runtime` | `Runtime` and process/device coordination. |
| `nnpd.results` | `runs`, `export_csv`, `comparison_plot`. |

The usual convenience imports are:

```python
from nnpd import (
    Analysis, Choice, Experiment, Metric, Plot, Product,
    execute, expand_sweep, load_analysis, load_experiment, load_settings, plan, restore,
)
```

`Context`, `Artifact`, `Writer`, and `Runtime` are imported from their respective
`nnpd.core` modules rather than the top-level package.

## Source API

```{toctree}
:maxdepth: 3

../autoapi/nnpd/index
../autoapi/gaussian/index
```

Use the [extension guide](../EXTENDING.md) for full hook contracts and the
[analysis guide](../guide/analysis.md) for the Gaussian product/metric mapping.
Private implementation helpers are not the main extension interface.
