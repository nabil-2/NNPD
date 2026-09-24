# Your first run

Every study, small or large, follows the same steps. The examples use the
`smoke` profile, which uses small datasets and only checks that everything works.
For a scientific run, replace `smoke` with `default`, `paper`, or your own profile.

| Step | Command line | Notebook |
|---|---|---|
| 1. Install | `uv sync` | — |
| 2. Choose the settings | edit `settings.py` | — |
| 3. Check the plan | `python run.py plan --profile smoke` | `01_run` |
| 4. Train | `python run.py train --profile smoke` | `01_run` |
| 5. Analyze | `python run.py analyze --profile smoke` | `01_run` |
| 6. Add analysis later | edit `metrics`/`plots`, then `analyze` again | `03_extensions` |
| 7. Inspect the results | `python run.py verify --profile smoke` | `02_inspect` |

`python run.py run` does steps 4 and 5 in one go. Run all commands from the
repository root, the directory containing `settings.py` and `run.py`.

## 1. Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
```

`uv sync` creates `.venv` with Python 3.11 or newer, downloading Python if needed,
and installs NNPD and the development tools at the versions pinned in `uv.lock`.
The commands below assume the environment is active. Alternatively, prefix them
with `uv run`, as in `uv run python run.py plan --profile smoke`.

Without uv, `python -m pip install -e . --group dev` (pip 25.1 or newer) installs
the same packages into your own environment, but resolves their versions itself
instead of using `uv.lock`.

On Linux, the PyTorch that `uv sync` installs is a CUDA build. For other GPUs or
CUDA versions, see the [CUDA guide](cuda.md).

## 2. Choose the settings

All experiment choices live in `settings.py`. `SETTINGS` is the complete
dictionary, and each profile (`default`, `paper`, `smoke`) is a named set of
changes to it. `--profile` selects one; without it, `DEFAULT_PROFILE` is used.
Wrap a value in `Choice([...])` to compare alternatives. See
[Configuration](configuration.md) and the [settings reference](../reference/settings.md).

## 3. Check the plan

```bash
python run.py plan --profile smoke
```

The plan lists every model to train: one per configuration and prior. For each it
shows the changed setting, the network input size, the truth design, and the
estimated analysis cost. It computes nothing. Read its `warnings`. If a
configuration's analysis would exceed the limits in `settings.py`, `plan` exits
with an error that names the settings to reduce, and the next steps refuse to
start. See [the feasibility check](configuration.md#analysis-must-be-feasible-before-anything-is-computed).

## 4. Train

```bash
python run.py train --profile smoke
```

This samples the training data and trains each model, then stops. Models that
already exist with the same training settings are reused, not retrained. For
several GPUs or processes, launch the same command with `torchrun`; see the
[CUDA guide](cuda.md).

## 5. Analyze

```bash
python run.py analyze --profile smoke
```

This builds the analysis data (truth observations, candidates, inference) and
writes every selected metric and plot. It needs the trained models from step 4
and **never trains**: if they are missing, it stops with an error.

## 6. Add analysis later

Analysis can be extended at any time without retraining. Add built-in names to
`metrics` or `plots` in `settings.py` and run `analyze` again: the saved models
and shared analysis data are reused. New metrics and plots are registered in
Python; `notebooks/03_extensions.ipynb` adds three to already trained models, and
the [extension guide](../EXTENDING.md) explains how to write your own.

## 7. Inspect the results

`notebooks/02_inspect.ipynb` reloads a trained model and its saved data, calls
metrics and plots directly, and exports a CSV table of all runs. The same works
in plain Python; see [Working with results](results.md).

```bash
python run.py verify --profile smoke
```

`verify` recomputes the checksum of every saved file to detect corruption. It is
an integrity check, not a measure of scientific quality.

The notebooks run the same steps as the command line. Each starts with
`PROFILE = "smoke"`; set the same profile in all three. See [Notebooks](notebooks.md).

## Change settings and run again

A profile's output folder holds one experiment, and **the latest settings win**.
Edit `settings.py` and run again: results of the earlier settings are replaced, so
`runs/` always matches what you last ran. Only what changed is recomputed. For
example, a new learning rate retrains the models but reuses the training data.

The cache keeps the trained models and analysis data of the **last three distinct
settings**, so switching back to one of them is fast. Older ones are deleted.
A failed launch deletes nothing.

Every successful launch copies the settings file it started from, byte for byte,
to `outputs/<profile>/settings.py`. If your `settings.py` has changed since, copy
what you need back from that file, or run the saved file directly with the same
profile:

```bash
python run.py run --config outputs/smoke/settings.py --profile smoke
```

Settings changed in Python on top of the file, as notebook 03 does, are not part
of the copy; each run's `run.json` records the complete settings that were used.
A launch from Python without `settings_file=` saves no copy.

To keep results side by side instead, give each experiment its own profile in
`settings.py`; each profile has its own output folder.

## Where everything is saved

Everything goes into `outputs/<profile>/`:

```text
outputs/smoke/
  settings.py             copy of the settings file of the latest launch
  history.json            the last three distinct settings and what they used
  source/<hash>.zip       snapshot of the code and settings.py that were used
  cache/                  shared, checksummed artifacts, reused across runs
    training/  model/     training data; model weights and loss history
    observations/  candidates/  inference/  ...   analysis data
  runs/baseline-<hash>/uniform-r0-<hash>/   one folder per model
    run.json              full resolved settings, package versions, backend
    status.json           running, trained, complete, or failed (with traceback)
    artifacts.json        which cache entries this run uses
    metrics.json          all metrics, also one file each in metrics/
    plots/<plot>/*.png    figures
```

The saved `settings.py`, each `run.json` and the `source/` snapshot record exactly
what produced the current results, so later edits to your `settings.py` do not
lose them. Move the whole
`outputs/<profile>/` folder, never a single run folder, because runs refer to the
shared cache. The [storage guide](../STORAGE.md) has the details.
