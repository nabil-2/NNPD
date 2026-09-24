# NNPD — a small, modular experiment framework

NNPD implements the Gaussian study of *Prior Dependence in Neural Ratio
Estimation* on top of a small, reusable experiment framework. The framework
does not import the Gaussian application or `settings.py`.

**Start with `settings.py`, `gaussian/experiment.py`, and `gaussian/hooks.py`.**
They respectively define the experiment choices, the scientific/training
implementation, and the datasets/metrics/plots to expose.

Full documentation — the usage guides, the settings and CLI reference, and the
generated Python API — is published at **<https://nabil-2.github.io/NNPD/>**.

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
python run.py plan --profile default
python run.py train --profile default
python run.py analyze --profile default
```

`run` does training and analysis. `train` stops after committed checkpoints.
`analyze` requires matching data and checkpoints and never trains implicitly.
Rerunning reuses completed artifacts. A failed, uncommitted stage is restarted,
not mistaken for a valid checkpoint.

**Read the plan before running the full study.** Every configuration's analysis
cost is checked against the limits in `settings.py` before anything is computed.
If any configuration exceeds them, `train`, `run` and `analyze` refuse to start
and name the settings to reduce; nothing is reduced automatically. Truth points
are Sobol points by default (1,024 at one inferred parameter, doubling per
additional one). `paper` uses `size_adaptive`: the exact `25**p` grid while it is
small enough, Sobol above. See [Truth design](docs/guide/configuration.md#truth-design).

## All ordinary choices live in one file

`settings.py` contains the complete settings dictionary and the named presets.
Edit it rather than chasing defaults through implementation files. The presets
are `default`, `paper`, and `smoke`.
`default` is the full study exactly as defined in `SETTINGS`. `paper` makes the
paper-facing choices explicit; it is not a claim of bitwise or figure-for-figure
reproduction.

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
```

An `Experiment` supplies the problem, prior, data sampler, model builder and
trainer. A `Product` builds a reusable dataset once. Each metric or plot declares
which products it needs, and receives those artifacts plus the full context:
model, simulator, prior, settings, runtime, run directory and artifact store.
There is no fixed number of hooks and no hard-coded metric switch in the runner.

Analysis can be added after training. `gaussian/extensions.py` adds a median-bias
metric sharing the existing inference data, a direct model-parameter-count metric,
and a custom observation plot. After a smoke run, this analyzes the already
trained models without retraining (any other trained profile works the same way):

```python
config = make_config("smoke")
config["application"] = "gaussian.extensions:ExtendedGaussian"
config["metrics"] += ["median_absolute_bias", "parameter_count"]
config["plots"] += ["observation_histogram"]
execute(config, load_experiment(config["application"]), stage="analyze")
```

`notebooks/03_extensions.ipynb` runs exactly this for its `PROFILE`. See
[Extending the framework](docs/EXTENDING.md) for contracts and examples.

## Notebooks and Python are the same workflow

Execute `notebooks/01_run.ipynb`, then `02_inspect.ipynb`, then
`03_extensions.ipynb`. The first runs a study; the second reloads saved
models/data, calls metrics and plots, and exports CSV; the third adds new metrics
and a plot to the trained models. Each starts with `PROFILE = "smoke"`; set it to
another profile in all three to use that one.
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
# settings.py: runtime.parallel="ddp"
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

Several choices need explicit interpretation: the exponential sign (the `default`
and `paper` presets differ), the normal-prior scale, the truth design and the
discrete integration measure. Read the [Scientific notes](docs/SCIENTIFIC_NOTES.md)
before comparing results with the paper.
Additional priors, publication-specific panels and scheduler-specific launch
policy belong in optional application hooks, not the framework.

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
