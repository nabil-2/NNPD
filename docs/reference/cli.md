# Command-line reference

From the repository root:

```bash
python run.py [plan|train|analyze|run|verify] [--profile NAME] [--settings-dir DIR]
```

After installation, `nnpd` is the equivalent command. Both use the same CLI and
public execution functions. Run `python run.py --help` for the parser's help text.

## Actions

| Action | Reads | Behavior |
|---|---|---|
| `plan` | both settings files | Default. Resolve and validate the sweep and print JSON jobs with training and analysis workload estimates; computes nothing. |
| `train` | `settings_training.py` | Sample/reuse training data, train/reuse models, and stop at checkpoints. Resets each run's analysis. |
| `analyze` | `settings_analysis.py`, plus `output` and `runtime` from `settings_training.py` | Analyze the models of the latest successful training. Never trains. Replaces each run's metrics and plots. |
| `run` | both settings files | `train`, then `analyze`. |
| `verify` | `output` from `settings_training.py` | Deep-check committed artifacts in the output folder. |

`plan`, `run` and `analyze` check that the analysis of every model fits the limits
in `settings_analysis.py`. If one does not, `plan` prints the plan and exits with
an error, and `run` and `analyze` refuse to start before computing anything.
`train` does not depend on the analysis settings and does not check them.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--profile NAME` | `DEFAULT_PROFILE` in `settings_training.py` | Select the preset of that name in both settings files. |
| `--settings-dir DIR` | The working directory | Folder containing `settings_training.py` and `settings_analysis.py`. |
| `-h`, `--help` | — | Show help and exit. |

The experiment class is set by `experiment` in the training settings and the
analysis class by `analysis` in the analysis settings. Every successful launch
copies the settings files it used byte for byte into the output folder, so
`--settings-dir outputs/<profile> --profile <profile>` runs them again in the same
output folder.

Do not expect a flag for every hyperparameter. Ordinary knobs deliberately live
in the two settings files. GPU visibility and distributed worker counts are
selected through the environment and `torchrun`; see [CUDA](../guide/cuda.md).

## Python equivalents

```python
from nnpd import execute, plan

jobs = plan("smoke")  # computes nothing

# Training/analysis is explicit; planning above does neither.
paths = execute("run", profile="smoke")
```

Valid `execute` stages are `"train"`, `"analyze"`, and `"run"`. Both functions
accept `settings_dir=` like `--settings-dir`. For checksums in Python, use
`nnpd.core.runner.verify_store(output_root)`.
