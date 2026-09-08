# Notebooks and Python scripts

The three notebooks are thin entry points, not a separate implementation. Their
functions live in importable Python modules. Use a kernel from the environment
where you installed NNPD.

| Notebook | Purpose |
|---|---|
| `01_run.ipynb` | Plan and run the smoke study. |
| `02_inspect.ipynb` | Read completed results, reload models/products, call hooks, and export CSV. |
| `03_extensions.ipynb` | Run the additional-metric and custom-plot example using compatible cached data/models. |

They are included in the repository under `notebooks/`. The website offers the
same files as downloads, without executing them during documentation builds:

{download}`01_run.ipynb <../../notebooks/01_run.ipynb>` ·
{download}`02_inspect.ipynb <../../notebooks/02_inspect.ipynb>` ·
{download}`03_extensions.ipynb <../../notebooks/03_extensions.ipynb>`

The development dependencies include an IPython kernel and notebook execution
tools. Use your existing notebook editor or install a notebook interface
separately; NNPD does not require one for ordinary Python execution.

## Equivalent script

Run this from the repository root in a normal Python file:

```python
from settings import make_config
from nnpd import execute, load_experiment, plan


def main():
    config = make_config("smoke")
    experiment = load_experiment(config["application"])
    jobs = plan(config, experiment)
    print(f"Planned models: {len(jobs)}")
    return execute(config, experiment, settings_file="settings.py")


if __name__ == "__main__":
    main()
```

Alternatively, use the existing `run.py` instead of creating a script.

Inspection does not train missing models. Run `01_run` or the smoke CLI command
before `02_inspect`. Requiring a derived product may calculate it from compatible
saved inputs; this is distinct from training. See [Results](results.md) for
examples of restoring contexts and accessing saved arrays.

For collective training, launch the Python entry point with `torchrun` rather
than trying to create multiple GPU worker processes from notebook cells.
