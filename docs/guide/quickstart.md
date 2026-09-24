# Your first run

Every study, small or large, follows the same steps. The examples use the
`smoke` profile, which uses small datasets and only checks that everything works.
For a scientific run, replace `smoke` with `default`, `paper`, or your own profile.

| Step | Command line | Notebook |
|---|---|---|
| 1. Install | `uv sync` | — |
| 2. Choose the settings | edit `settings_training.py` and `settings_analysis.py` | — |
| 3. Check the plan | `python run.py plan --profile smoke` | `01_run` |
| 4. Train | `python run.py train --profile smoke` | `01_run` |
| 5. Analyze | `python run.py analyze --profile smoke` | `01_run` |
| 6. Change the analysis later | edit `settings_analysis.py`, then `analyze` again | `03_extensions` |
| 7. Inspect the results | `python run.py verify --profile smoke` | `02_inspect` |

`python run.py run` does steps 4 and 5 in one go. Run all commands from the
repository root, the directory containing the two settings files and `run.py`.

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

The settings are split into two files:

| File | Decides | Sections |
|---|---|---|
| `settings_training.py` | What is trained | `problem`, `cohort`, `priors`, `data`, `model`, `training`, `runtime` |
| `settings_analysis.py` | What is computed from the trained models | `inference`, `verification`, `metrics`, `plots`, `plotting` |

In each file, `SETTINGS` is the complete dictionary, and each profile (`default`,
`paper`, `smoke`) is a named set of changes to it. `--profile` selects the same
profile in both files; without it, `DEFAULT_PROFILE` from `settings_training.py`
is used. Wrap a training value in `Choice([...])` to compare alternatives. See
[Configuration](configuration.md) and the [settings reference](../reference/settings.md).

## 3. Check the plan

```bash
python run.py plan --profile smoke
```

The plan lists every model to train: one per configuration and prior. For each it
shows the changed setting, the network input size, the truth design, and the
estimated analysis cost. It computes nothing. Read its `warnings`. If a model's
analysis would exceed the limits in `settings_analysis.py`, `plan` exits with an
error that names the settings to reduce, and `run` and `analyze` refuse to start.
See [the feasibility check](configuration.md#analysis-must-be-feasible-before-anything-is-computed).

## 4. Train

```bash
python run.py train --profile smoke
```

This samples the training data and trains each model, then stops. It reads only
`settings_training.py`. Models that already exist with the same training settings
are reused, not retrained. For several GPUs or processes, launch the same command
with `torchrun`; see the [CUDA guide](cuda.md).

## 5. Analyze

```bash
python run.py analyze --profile smoke
```

This analyzes the models of the latest successful training with
`settings_analysis.py`: it builds the analysis data (truth observations,
candidates, inference) and writes every selected metric and plot. It **never
trains**: if there are no trained models, it stops with an error. From
`settings_training.py` it only reads `output` and `runtime`, so later edits to the
training settings do not change which models are analyzed.

## 6. Change the analysis later

Analysis can be changed at any time without retraining. Edit
`settings_analysis.py`, for example add names to `metrics` or `plots` or change the
inference resolution, and run `analyze` again: the saved models are reused, and so
is every analysis dataset whose settings did not change. The new analysis replaces
the previous one. `notebooks/03_extensions.ipynb` shows this, and tries new metrics
on a saved run before adding them; the [extension guide](../EXTENDING.md) explains
how to write your own.

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

A profile's output folder holds one experiment, and **the latest settings win**:

- `train` with changed training settings replaces the earlier runs, so `runs/`
  always matches what you last trained. Only what changed is retrained: a new
  learning rate retrains the models but reuses the training data. Training starts
  each run's analysis over; run `analyze` afterwards.
- `analyze` with changed analysis settings replaces the metrics and plots of every
  run. It never retrains.

The cache keeps the trained models and analysis data of the **last three distinct
training settings**, so switching back to one of them is fast. Older ones are
deleted. A failed launch deletes no runs, models or cached data.

Every successful launch copies the settings files it used, byte for byte, into the
output folder: `train` copies `settings_training.py`, `analyze` copies
`settings_analysis.py`, and `run` copies both. If your files have changed since,
copy what you need back from those copies, or run them directly:

```bash
python run.py run --profile smoke --settings-dir outputs/smoke
```

To keep results side by side instead, give each experiment its own profile in both
settings files; each profile has its own output folder.

## Where everything is saved

Everything goes into `outputs/<profile>/`:

```text
outputs/smoke/
  settings_training.py    copy of the training settings of the latest training
  settings_analysis.py    copy of the analysis settings of the latest analysis
  history.json            the last three distinct training settings and what they used
  source/<hash>.zip       snapshot of the code and settings files that were used
  cache/                  shared, checksummed artifacts, reused across runs
    training/  model/     training data; model weights and loss history
    observations/  candidates/  inference/  ...   analysis data
  runs/baseline-<hash>/uniform-r0-<hash>/   one folder per model
    run.json              resolved training settings, package versions, backend
    status.json           training state: running, trained, or failed (with traceback)
    analysis.json         analysis settings and state: running, complete, or failed
    artifacts.json        which cache entries this run uses
    metrics.json          all metrics, also one file each in metrics/
    plots/<plot>/*.png    figures
```

The saved settings files, each `run.json` and `analysis.json`, and the `source/`
snapshots record exactly what produced the current results, so later edits to your
settings files do not lose them. Move the whole `outputs/<profile>/` folder, never
a single run folder, because runs refer to the shared cache. The
[storage guide](../STORAGE.md) has the details.
