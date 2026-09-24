# NNPD — a small, modular experiment framework

NNPD implements the Gaussian study of *Prior Dependence in Neural Ratio
Estimation* on top of a small, reusable experiment framework. The framework
does not import the Gaussian application or the settings files.

Everything is split into **training** (what is trained) and **analysis** (what is
computed from the trained models). Start with the four files that define them:

| | Settings | Implementation |
|---|---|---|
| Training | `settings_training.py` | `gaussian/experiment.py` |
| Analysis | `settings_analysis.py` | `gaussian/analysis.py` |

Full documentation — the usage guides, the settings and CLI reference, and the
generated Python API — is published at **<https://nabil-2.github.io/NNPD/>**.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from
this directory:

```bash
uv sync
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python run.py plan --profile smoke
python run.py run --profile smoke
python run.py verify --profile smoke
```

`uv sync` creates `.venv` with Python 3.11 or newer and installs NNPD and the
development tools at the versions pinned in `uv.lock`. Instead of activating the
environment, you can prefix commands with `uv run`.

The smoke profile trains the four prior models and exercises the complete analysis
with small datasets. It checks the plumbing; it is not a scientific convergence
study. The default action is **plan**, so `python run.py` cannot inadvertently start
a large run. `nnpd` is the equivalent installed command.

For scientific settings:

```bash
python run.py plan --profile default
python run.py train --profile default
python run.py analyze --profile default
```

`train` trains the models of `settings_training.py`. `analyze` computes the
metrics and plots of `settings_analysis.py` from the latest trained models and
**never trains**. `run` does both. Rerunning reuses completed artifacts. A failed,
uncommitted stage is restarted, not mistaken for a valid checkpoint.

**Read the plan before running the full study.** The analysis cost of every
trained model is checked against the limits in `settings_analysis.py` before
anything is computed. If any model exceeds them, `run` and `analyze` refuse to
start and name the settings to reduce; nothing is reduced automatically. Truth points
are Sobol points by default (1,024 at one inferred parameter, doubling per
additional one). `paper` uses `size_adaptive`: the exact `25**p` grid while it is
small enough, Sobol above. See [Truth design](docs/guide/configuration.md#truth-design).

## All ordinary choices live in two files

`settings_training.py` holds everything that defines the trained models: problem,
priors, data, model, training and runtime. `settings_analysis.py` holds everything
computed from them: inference, verification, metrics, plots and plotting. Change
the analysis settings and run `analyze` again at any time; the models are reused.
Edit these files rather than chasing defaults through implementation files.

Both files define the profiles `default`, `paper`, and `smoke`; `--profile`
selects the same profile in both. `default` is the full study exactly as defined
in the two `SETTINGS` dictionaries. `paper` makes the paper-facing choices
explicit; it is not a claim of bitwise or figure-for-figure reproduction.

### Sweeps: baseline plus one changed knob

```python
"data": {"train_size": Choice([50_000, 100_000, 150_000]), ...},
"training": {"learning_rate": Choice([0.1, 0.01, 0.001]), ...},
```

There are **five configurations**, not nine:

| Training rows | Learning rate |
|---:|---:|
| 50,000 | 0.1 |
| 100,000 | 0.1 |
| 150,000 | 0.1 |
| 50,000 | 0.01 |
| 50,000 | 0.001 |

The first value is the baseline. Each alternative changes just that knob, and
duplicate resolved configurations are removed. This works for nested scalar
settings and for atomic list/dictionary options:

```python
"kind": Choice(["mlp", "residual"]),
"hidden": Choice([[64, 64, 64, 64], [128, 128]]),
"infer": Choice([["mean:*"], ["mean:*", "std:1"]]),
```

**An ordinary list is a single literal value**, not a sweep. This is essential:
`[64, 64]` means two hidden layers, and `["mean:*", "std:1"]` means a single
inference-parameter set. Wrap alternatives in `Choice(...)` to make a sweep.
`Choice` inside an option or inside an ordinary list is deliberately rejected.

The four priors form a separately declared **fixed comparison cohort** repeated
for each configuration. With four priors and one training replica, the five
configurations above train twenty models. This is intentional and is not a
Cartesian sweep of independent knobs. To make the prior itself an OFAT knob,
use `"priors": Choice([["uniform"], ["normal"], ["exponential"], ["grid"]])`
inside `cohort`. `replicas` means independently trained models, not repeated
observations in an inference case.

`Choice` is for training settings only. An analysis is always a single setting:
change it and analyze again. Runtime settings must remain fixed within a launch;
compare CPU/CUDA or `jobs`/`ddp` using separate launches. Every alternative is
validated before the first job starts.

### Inferred parameter sets

Indices are **zero-based**. `mean:*` expands to all observation means; `std:1`
adds the standard deviation of the second observation dimension. For example,
with `dimension=2`, these alternatives have two versus three inferred parameters,
and automatically construct networks with four versus five input features.

Unselected likelihood parameters stay at `fixed_mean`, `fixed_std`, or the named
values in `problem.fixed`; they are **fixed, not marginalized nuisance parameters**.
Bounds and prior-family parameters come from `problem.parameters`, with optional
name-specific `problem.overrides`. Standard deviations must be positive. Invalid
combinations, such as `std:1` in a one-dimensional baseline, fail during planning.
Set a compatible baseline dimension rather than relying on Cartesian combinations.

## Structure and extension points

```text
settings_training.py       What is trained: training settings and profiles
settings_analysis.py       What is computed from trained models: analysis settings and profiles
run.py                     Thin Python entry point
nnpd/
  core/
    api.py                 Experiment and Analysis; Product, Metric, Plot, Context
    settings.py            Explicit Choice, OFAT expansion, settings loading
    runner.py              Planning, execution, restore, integrity verification
    storage.py             Atomic artifact store and array/checkpoint access
    runtime.py             CPU/CUDA and torchrun process management
  nre/
    distributions.py       Simulator/Prior interfaces and bounded independent priors
    data.py                Balanced joint-versus-factorized pairs
    models.py              MLP/residual implementations and factory registry
    training.py            Logit BCE training, prediction and classification
    inference.py           Ensemble log ratios and numerical HLD summaries
  results.py               Read runs, export CSV, compare arbitrary saved metrics
gaussian/
  experiment.py            What is trained: problem, prior, data, model, training
  problem.py               Gaussian likelihood and selected parameter layout
  sampling.py              Training-data builder
  evaluation.py            Shared truth ensembles, candidates, inference products
  diagnostics.py           Pairwise ratios, reweighting, normalization datasets
  metrics.py               Independent metric functions
  plots.py                 Ordinary Matplotlib plotting functions
  analysis.py              What is computed: named products, metrics and plots
  extensions.py            Working custom metric/model metric/plot example
notebooks/                 Three thin executable notebooks; no function definitions
tests/                     Unit, science, integration, notebook and distributed tests
```

An `Experiment`, named by `"experiment"` in the training settings, supplies the
problem, prior, data sampler, model builder and trainer. An `Analysis`, named by
`"analysis"` in the analysis settings, registers products, metrics and plots. A
`Product` builds a reusable dataset once. Each metric or plot declares which
products it needs, and receives those artifacts plus the full context: model,
simulator, prior, both settings, runtime, run directory and artifact store. There
is no fixed number of hooks and no hard-coded metric switch in the runner.

Analysis can be added after training. `gaussian/extensions.py` adds a median-bias
metric sharing the existing inference data, a direct model-parameter-count metric,
and a custom observation plot. To compute them for already trained models, change
`settings_analysis.py` and analyze again; nothing is retrained:

```python
"analysis": "gaussian.extensions:ExtendedAnalysis",
"metrics": [..., "median_absolute_bias", "parameter_count"],
"plots": [..., "observation_histogram"],
```

`notebooks/03_extensions.ipynb` analyzes trained models again and shows these
metrics on a saved run. See [Extending the framework](docs/EXTENDING.md) for
contracts and examples.

## Notebooks and Python are the same workflow

Execute `notebooks/01_run.ipynb`, then `02_inspect.ipynb`, then
`03_extensions.ipynb`. The first trains and analyzes a study; the second reloads
saved models/data, calls metrics and plots, and exports CSV; the third analyzes
the trained models again and tries new metrics on them. Each starts with
`PROFILE = "smoke"`; set it to another profile in all three to use that one.
The second and third notebooks never train.

All of this is also ordinary Python, reading the same two settings files:

```python
from nnpd import execute, plan

jobs = plan("smoke")
run_directories = execute("run", profile="smoke")  # or "train" / "analyze"
```

## CUDA and multiple GPUs

On Linux, the PyTorch that `uv sync` installs is a CUDA build; for other hardware
see the [CUDA guide](docs/guide/cuda.md).
`runtime.device="auto"` chooses CUDA when available, otherwise CPU. Explicit
`"cuda"` fails clearly when CUDA is unavailable; it never silently falls back.
`"cuda:1"` selects a single device outside a distributed launch.

There are two simple parallel modes:

**`jobs`** assigns different complete model/prior/configuration jobs to workers,
including their analysis. For these small networks this is usually the useful
first option; each job fits on one device.

```bash
# settings_training.py: runtime.parallel="jobs", device="auto" or "cuda"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run.py run --profile smoke
```

**`ddp`** trains each model collectively using DistributedDataParallel, then runs
that model's analysis on rank zero. Batch size is global, not multiplied by the
GPU count. The trainer handles unequal and empty final shards without duplicating
or dropping real training examples.

```bash
# settings_training.py: runtime.parallel="ddp"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run.py run --profile smoke
```

Use one worker per visible GPU and homogeneous GPU types within a DDP group.
CPU distributed runs use Gloo; CUDA DDP uses NCCL. Multi-node launch requires
correct torchrun rendezvous settings and a shared output filesystem; it is not
covered by the tests. DDP analysis is deliberately not sharded within a model.
`float16` AMP requires CUDA; `bfloat16` is also supported on capable CPUs/GPUs.
A stochastic model or different numerical backend need not reproduce identical
weights across world sizes.

The distributed tests run two-process Gloo DDP and independent jobs on CPU.
Single- and multi-GPU tests are included; they run only where CUDA hardware is
available. See [Testing and validation](docs/guide/testing.md).

## Outputs, correctness and scope

Run folders contain the resolved training settings, cohort member, training
status, the analysis settings and state, source-snapshot references, metrics,
plots and relative artifact references.
Shared datasets and model weights live in a content-addressed store; multiple
metrics do not duplicate them. Numeric arrays use `.npy` with optional memory
mapping; metadata uses JSON; models use tensor state dictionaries loaded with
`weights_only=True`. Each profile's output folder holds one experiment: running
again with changed settings replaces the earlier results, copies of both settings
files are saved there, and the cache keeps what the last three distinct training
settings used. See [Storage and reproducibility](docs/STORAGE.md).

The example distinguishes the normalized ensemble **likelihood ratio** from a
**posterior**. Bias, projected HLD coverage, widths, candidate-resolution checks,
analytic-likelihood references, pairwise log-ratio errors, sliced Wasserstein
reweighting and evidence-normalization checks are separate metrics. Raw inference
scores can be kept for a few diagnostic cases or for every case.

Several choices need explicit interpretation: the exponential sign (the `default`
and `paper` presets differ), the normal-prior scale, the truth design and the
discrete integration measure. Read the [Scientific notes](docs/SCIENTIFIC_NOTES.md)
before comparing results with the paper.
Additional priors, publication-specific panels and scheduler-specific launch
policy belong in the application's experiment or analysis, not the framework.

## Tests

```bash
python -m pytest -q
python -m pytest -q --cov=nnpd --cov=gaussian --cov-report=term-missing
python -m pytest -q -m distributed
python -m pytest -q -m cuda       # executes on suitably provisioned CUDA hardware
```

The tests include mathematical reference cases, exact OFAT expansion, bounded
prior sampling, arbitrary parameter layouts, both networks, shared caching,
corruption detection, interrupted-stage recovery, safe reload/relocation,
independent non-Gaussian framework use, notebook execution and real distributed
processes. They are evidence for the tested implementation—not a guarantee of
scientific convergence for arbitrary high-dimensional configurations. See
[Testing and validation](docs/guide/testing.md) for coverage details and the
full-size reference run.
