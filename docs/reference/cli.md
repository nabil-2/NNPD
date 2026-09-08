# Command-line reference

From the repository root:

```bash
python run.py [plan|run|train|analyze|verify] [--config PATH] [--profile NAME] [--application MODULE:CLASS]
```

After installation, `nnpd` is the equivalent command. Both use the same CLI and
public execution functions. Run `python run.py --help` for the parser's help text.

## Actions

| Action | Behavior |
|---|---|
| `plan` | Default. Resolve and validate the sweep and print JSON jobs/workload estimates; no data sampling or training. |
| `run` | Train missing compatible models and run selected analysis. |
| `train` | Sample/reuse training data, train/reuse models, and stop at checkpoints. |
| `analyze` | Require matching training/model artifacts; run analysis without implicit training. |
| `verify` | Deep-check committed artifacts in the planned output roots. |

`plan` can report guard errors in a workload without exiting unsuccessfully.
`run` and `analyze` reject workloads exceeding configured guards. `train` does not
allocate inference products and does not enforce those inference guards.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--config PATH` | `settings.py` in the working directory | Trusted Python file exposing `DEFAULT_PROFILE` and `make_config(profile)`. |
| `--profile NAME` | That file's `DEFAULT_PROFILE` | Select a preset defined by the settings file. |
| `--application MODULE:CLASS` | `config["application"]` | Override the experiment class for this invocation. |
| `-h`, `--help` | — | Show help and exit. |

Do not expect a flag for every hyperparameter. Ordinary knobs deliberately live
in one settings file. GPU visibility and distributed worker counts are selected
through the environment and `torchrun`; see [CUDA](../guide/cuda.md).

## Python equivalents

```python
from settings import make_config
from nnpd import execute, load_experiment, plan

config = make_config("smoke")
experiment = load_experiment(config["application"])
jobs = plan(config, experiment)

# Training/analysis is explicit; planning above does neither.
paths = execute(config, experiment, stage="run", settings_file="settings.py")
```

Valid `execute` stages are `"run"`, `"train"`, and `"analyze"`. `plan` is a
separate function. For checksums in Python, use
`nnpd.core.runner.verify_store(output_root)`.
