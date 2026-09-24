# Notebooks and Python scripts

The three notebooks are thin entry points, not a separate implementation. Their
functions live in importable Python modules. Use a kernel from the environment
where you installed NNPD, `.venv` after `uv sync`.

| Notebook | Purpose |
|---|---|
| `01_run.ipynb` | Plan, train and analyze the selected profile. |
| `02_inspect.ipynb` | Read completed results, reload models/products, call hooks, and export CSV. |
| `03_extensions.ipynb` | Analyze the trained models again with `settings_analysis.py`, and try new metrics on a saved run. Never trains. |

Like the command line, the notebooks read `settings_training.py` and
`settings_analysis.py`; to change what is trained or analyzed, edit those files.
Each notebook starts with `PROFILE = "smoke"`. Set it to `"default"` or `"paper"`
in all three to work with that profile instead; `03` only needs its models to
have been trained, by `01` or by `python run.py train --profile <name>`.

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
from nnpd import execute, plan


def main():
    jobs = plan("smoke")
    print(f"Planned models: {len(jobs)}")
    return execute("run", profile="smoke")  # or "train" / "analyze"


if __name__ == "__main__":
    main()
```

`plan` and `execute` read the two settings files from the current folder, or from
`settings_dir=...`, exactly like the command line. Alternatively, use the existing
`run.py` instead of creating a script.

Inspection does not train missing models. Run `01_run` or the smoke CLI command
before `02_inspect`. Requiring a derived product may calculate it from compatible
saved inputs; this is distinct from training. See [Results](results.md) for
examples of restoring contexts and accessing saved arrays.

For collective training, launch the Python entry point with `torchrun` rather
than trying to create multiple GPU worker processes from notebook cells.
