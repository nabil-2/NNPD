# Working with saved results

Start after a completed smoke run. Paths here are relative to the repository
root. The [storage guide](../STORAGE.md) describes manifests, cache recipes,
array shapes, and relocation in detail.

## List runs and restore a model

```python
from nnpd import load_experiment, restore
from nnpd.results import runs

records = runs("outputs/smoke")
if not records:
    raise RuntimeError("First run: python run.py run --profile smoke")

record = records[0]
experiment = load_experiment(record["config"]["application"])
context = restore(record["path"], experiment, device="cpu")
model = context.model
observations = context.require("observations").array("observations")
inference = context.require("inference")
```

`runs` returns complete runs by default. Pass `complete_only=False` to include
trained-only or failed records and inspect their `status`. `restore` loads the
saved configuration and verifies existing artifact files. It defaults to CPU and
does not inherit stale distributed rank variables.

`context.model` builds the matching architecture and loads saved tensor weights;
it does not train. `context.require(name)` resolves the **current** product
recipe, reusing a compatible artifact or building the requested derived product.
Changed analysis code or dependencies can therefore recalculate derived data.

## Read an original artifact without recomputing it

For direct inspection of a product referenced by a saved run:

```python
import json
from pathlib import Path
from nnpd.core.storage import Artifact

run_dir = Path(record["path"])
references = json.loads((run_dir / "artifacts.json").read_text())
if "inference" not in references["items"]:
    raise RuntimeError("This run did not save an inference product reference.")
saved = Artifact((run_dir / references["items"]["inference"]).resolve())
saved.verify(deep=True)
mode = saved.array("ratio_mode")
```

This reads the saved product as-is; it does not rebuild it under today's source.
For a reproducible analysis rerun, use the saved source snapshot and compatible
package versions. Moving a whole output root preserves relative references;
moving only one run directory does not preserve its cache.

## Call a metric or plot directly

```python
from matplotlib import pyplot as plt

metric = experiment.metrics()["bias"]
value = metric.compute(context, context.dependencies(metric.needs))
print(value)

plot = experiment.plots()["inference"]
figures = plot.draw(context, context.dependencies(plot.needs))
try:
    output = Path("outputs/manual-plots")
    output.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(output / f"{name}.svg", bbox_inches="tight")
finally:
    for figure in figures.values():
        plt.close(figure)
```

The built-in runner saves PNGs. Calling the same plot hook directly allows other
formats and interactive notebook display. Close figures you create outside the
runner when you no longer need them.

## Export a comparison table

```python
from nnpd.results import export_csv

export_csv("outputs/smoke", "outputs/summary.csv", {
    "prior": "member.prior",
    "dimension": "config.problem.dimension",
    "absolute_bias": "metrics.bias.mean_absolute",
    "coverage_68": "metrics.coverage.projected_mean.0",
})
```

Dotted paths traverse dictionaries and lists. The `.0` coverage example assumes
the first configured level is 0.68; inspect `metrics.coverage.levels` when changing
level order. Missing fields become blank CSV cells.

`comparison_plot(records, x_path, metric_path, group_path="member.prior")` returns
a figure for a selected scalar metric. Filter records before comparing them so
that different OFAT axes, replicas, parameter sets, and training backends are not
accidentally pooled together.
