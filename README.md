# NNPD — a small, modular experiment framework

This is a fresh implementation of the supplied Gaussian neural-prior-dependence
study, not a wrapper around its notebook globals or numbered output directories.
The reusable framework does not import the Gaussian application or `settings.py`.

**Start with `settings.py`, `gaussian/experiment.py`, and `gaussian/hooks.py`.**
They respectively define the experiment choices, the scientific/training
implementation, and the datasets/metrics/plots to expose.

## Run it

Use Python 3.11 or newer. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
python run.py plan --profile smoke
python run.py run --profile smoke
python run.py verify --profile smoke
```

The smoke profile trains the four prior models and exercises the complete analysis
with small datasets. It checks the plumbing; it is not a scientific convergence
study. The default action is **plan**, so `python run.py` cannot inadvertently start
a large run. `nnpd` is the equivalent installed command.

For scientific settings:

```bash
python run.py plan --profile legacy
python run.py train --profile legacy
python run.py analyze --profile legacy
```

`run` does training and analysis. `train` stops after committed checkpoints.
`analyze` requires matching data and checkpoints and never trains implicitly.
Rerunning reuses completed artifacts. A failed, uncommitted stage is restarted,
not mistaken for a valid checkpoint.

**Read the plan before running the full study.** The literal paper grid contains
`25**d` truth points. Candidate evaluation and storage guards can reject the full
high-dimensional plan before allocation. The `legacy` profile retains the current
CLI's explicit auto grid/Sobol truth-design policy; `paper` does not silently
replace its exhaustive truth grid. Neither profile automatically reduces the
requested candidate count to fit a guard.

## All ordinary choices live in one file

`settings.py` contains the complete settings dictionary and the named presets.
Edit it rather than chasing defaults through implementation files. The presets
are `legacy`, `paper`, `smoke`, `mixed`, `extended`, and `smoke_ddp`.
`legacy` follows the active old Python entry point, subject to the documented
engineering changes. `paper` makes the paper-facing choices explicit; it is not
a claim of bitwise or figure-for-figure reproduction.

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

Runtime settings must remain fixed within a launch; compare CPU/CUDA or `jobs`/`ddp`
using separate launches. All scientific/model/training/analysis settings can be
swept. Every alternative is validated before the first job starts.

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

`python run.py run --profile mixed` is an executable example comparing the two
parameter sets, MLP/residual architectures, and learning rates with OFAT semantics.

## Structure and extension points

```text
settings.py                All presets and tunable experiment settings
run.py                     Thin Python entry point
nnpd/
  core/
    api.py                 Experiment ABC; Product, Metric, Plot, Context
    config.py              Explicit Choice and OFAT expansion
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
  experiment.py            Wires the example to the abstract framework
  problem.py               Gaussian likelihood and selected parameter layout
  sampling.py              Training-data builder
  evaluation.py            Shared truth ensembles, candidates, inference products
  diagnostics.py           Pairwise ratios, reweighting, normalization datasets
  metrics.py               Independent metric functions
  plots.py                 Ordinary Matplotlib plotting functions
  hooks.py                 Named product/metric/plot registrations
  extensions.py            Working custom metric/model metric/plot example
notebooks/                 Three thin executable notebooks; no function definitions
tests/                     Unit, science, integration, notebook and distributed tests
validation/                Executed validation evidence and rerun script
```

An `Experiment` supplies the problem, prior, data sampler, model builder and
trainer. A `Product` builds a reusable dataset once. Each metric or plot declares
which products it needs, and receives those artifacts plus the full context:
model, simulator, prior, settings, runtime, run directory and artifact store.
There is no fixed number of hooks and no hard-coded metric switch in the runner.

For a complete, tested extension, read `gaussian/extensions.py` and run:

```bash
python run.py run --profile extended
```

It adds a median-bias metric sharing the existing inference data, a direct
model-parameter-count metric, and a custom observation plot. It deliberately
uses the smoke artifact store so previously trained smoke models are reused.
See [Extending the framework](docs/EXTENDING.md) for contracts and examples.

## Notebooks and Python are the same workflow

Execute `notebooks/01_run.ipynb`, then `02_inspect.ipynb`, then
`03_extensions.ipynb`. The first runs the smoke study; the second reloads saved
models/data, calls metrics and plots, and exports CSV; the third runs the extension.
The inspection notebook intentionally does not train missing models.

All of this is also ordinary Python:

```python
from settings import make_config
from nnpd import execute, load_experiment, plan

config = make_config("smoke")
experiment = load_experiment(config["application"])
jobs = plan(config, experiment)
run_directories = execute(config, experiment, settings_file="settings.py")
```

## CUDA and multiple GPUs

Install a CUDA-enabled PyTorch build appropriate to the machine before running.
`runtime.device="auto"` chooses CUDA when available, otherwise CPU. Explicit
`"cuda"` fails clearly when CUDA is unavailable; it never silently falls back.
`"cuda:1"` selects a single device outside a distributed launch.

There are two simple parallel modes:

**`jobs`** assigns different complete model/prior/configuration jobs to workers,
including their analysis. For these small networks this is usually the useful
first option; each job fits on one device.

```bash
# settings.py: runtime.parallel="jobs", device="auto" or "cuda"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run.py run --profile smoke
```

**`ddp`** trains each model collectively using DistributedDataParallel, then runs
that model's analysis on rank zero. Batch size is global, not multiplied by the
GPU count. The trainer handles unequal and empty final shards without duplicating
or dropping real training examples.

```bash
# smoke_ddp already sets runtime.parallel="ddp"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run.py run --profile smoke_ddp
```

Use one worker per visible GPU and homogeneous GPU types within a DDP group.
CPU distributed runs use Gloo; CUDA DDP uses NCCL. Multi-node launch requires
correct torchrun rendezvous settings and a shared output filesystem; it was not
validated here. DDP analysis is deliberately not sharded within a model.
`float16` AMP requires CUDA; `bfloat16` is also supported on capable CPUs/GPUs.
A stochastic model or different numerical backend need not reproduce identical
weights across world sizes.

**Validation here used CPU-only PyTorch.** Two-process Gloo DDP and independent
jobs were executed; physical single-/multi-GPU tests are included but were skipped.
See [TEST_REPORT.md](TEST_REPORT.md) for the exact environment and results.

## Outputs, correctness and scope

Run folders contain the full resolved configuration, cohort member, execution
status, source-snapshot reference, metrics, plots and relative artifact references.
Shared datasets and model weights live in a content-addressed store; multiple
metrics do not duplicate them. Numeric arrays use `.npy` with optional memory
mapping; metadata uses JSON; models use tensor state dictionaries loaded with
`weights_only=True`. See [Storage and reproducibility](docs/STORAGE.md).

The example distinguishes the normalized ensemble **likelihood ratio** from a
**posterior**. Bias, projected HLD coverage, widths, candidate-resolution checks,
analytic-likelihood references, pairwise log-ratio errors, sliced Wasserstein
reweighting and evidence-normalization checks are separate metrics. Raw inference
scores can be kept for a few diagnostic cases or for every case.

The paper and old repository do not define one identical configuration. The
exponential sign, truth-design settings, discrete integration measure, and older
notebook posterior computations need explicit interpretation. Read
[Scientific and migration notes](docs/SCIENTIFIC_NOTES.md) before comparing results.
Historical exploratory gamma/sinusoidal priors, every old publication panel and
site-specific Slurm policy are not copied verbatim; they belong in optional
application hooks, not the framework.

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
scientific convergence for arbitrary high-dimensional configurations.
