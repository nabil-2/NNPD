# Your first run

Run the commands below from the repository root: the directory containing
`settings.py`, `run.py`, and `pyproject.toml`. The documentation website can be
built separately without installing NNPD; see [Publishing](../publishing.md).

## Install the experiment environment

The application declares Python **3.11 or newer**. Create an isolated environment
and install the package and its development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead of
`source`. For GPU execution, install the appropriate CUDA-enabled PyTorch build
for your machine; the [CUDA guide](cuda.md) explains NNPD's execution modes.

## Inspect before running

```bash
python run.py plan --profile smoke
```

The JSON plan reports each cohort member, the setting changed from baseline,
network input size, and estimated inference work. This preset plans four models:
one per prior, at one baseline configuration and one training replica.

The workload's `errors` and `warnings` are meaningful even when the plan command
returns successfully. In particular, inference guards are reported in the plan;
`run` and `analyze` enforce them before allocating those products.

## Train and analyze

```bash
python run.py run --profile smoke
python run.py verify --profile smoke
```

`run` samples data, trains missing compatible models, builds requested analysis
products, computes metrics, and writes figures. The smoke output root is
`outputs/smoke`. `verify` recomputes checksums of committed cache artifacts; it is
an integrity check, not a scientific quality score.

A second identical run reuses compatible datasets and checkpoints. Metrics and
plots are evaluated again; this is not an all-or-nothing cached terminal output.

## Separate the stages

```bash
python run.py train --profile smoke
python run.py analyze --profile smoke
```

`train` stops after committed weights and history. `analyze` requires matching
training data and model artifacts and **does not train implicitly**. Use the same
settings and compatible training backend across those commands. For inspecting
an existing model on a different device, use `restore`, as shown in
[Working with results](results.md).

## Move to a larger profile deliberately

The shipped presets are `legacy`, `paper`, `smoke`, `mixed`, `extended`, and
`smoke_ddp`. The default preset is `legacy`, but the default action is `plan`.
Start with:

```bash
python run.py plan --profile legacy
```

Do not treat the exhaustive high-dimensional paper plan as a quick example.
Truth counts, candidate evaluation, and retained arrays can become very large.
Read the [scientific notes](../SCIENTIFIC_NOTES.md) before interpreting agreement
or disagreement with the original paper or code.
