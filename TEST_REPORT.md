# Executed test report

Validation date: **8 September 2026**. This report describes tests actually run,
not tests merely supplied for someone else to run.

## Environment and complete automated suite

Linux x86-64; Python **3.13.5**; PyTorch **2.10.0+cpu**; NumPy **2.3.5**;
SciPy **1.17.0**; Matplotlib **3.10.8**; pytest **9.0.2**.
The environment had **no CUDA devices**. Exact versions are recorded in
`validation/environment.json` and `validation/requirements-tested.txt`.

Executed command:

```bash
python -m pytest -q --cov=nnpd --cov=gaussian --cov-report=term-missing \
  --cov-report=json:validation/coverage.json --junitxml=validation/junit.xml
```

**Result: 83 passed, 3 skipped, 0 failed in 117.16 seconds.**
Reported line coverage of `nnpd` and `gaussian`: **94% rounded** (1,640 of 1,747
statements). Coverage includes the main pytest process; successful notebook kernels
and torchrun child processes are not merged into this coverage percentage. Thus
some distributed lines appear uncovered despite real distributed execution.

Evidence: `validation/pytest.log`, `validation/junit.xml`,
`validation/coverage.json`, `validation/distributed_ddp.log` and
`validation/distributed_jobs.log`.

### What passed

| Area | Executed checks |
|---|---|
| Configuration | Exact five-run OFAT example; no Cartesian product; duplicate baseline removal; atomic list/dict options; literal architecture lists; invalid alternative rejection before execution. |
| Priors | Numerical normalization and sample moments for uniform, truncated normal and signed/zero exponential rates; exact grid mass and off-support behavior; mixed continuous/discrete supports. |
| Scientific algebra | Gaussian sampler/reference likelihood; joint/factorized class behavior; ensemble logit sums; extreme-score stability; correct single prior factor; density versus quadrature mass; HLD cutoff ties. |
| Analytic checks | Fine-grid 1D Gaussian mode/HLD width versus analytic values; 2D joint-HLD projections versus marginal intervals; model likelihood oracle versus numerical normalized reference. |
| Networks/training | MLP and residual models; arbitrary selected mean/std input dimensions; Adam/AdamW; CPU full precision and bfloat16; gradient clipping; best/last checkpoints. |
| Artifacts | Atomic commit/failed-stage cleanup; shared products; dependency/source/version invalidation; source entry-point changes; file corruption and deep checksums; safe arrays and checkpoint reload; relocation of a complete output root. |
| Pipeline/extensions | Actual four-prior sampling/training/inference; cached model reuse after adding metrics; custom model-only and dataset-based metrics; every built-in plot; custom plots; separate non-Gaussian experiment. |
| CLI | Plan without allocation; train-only, analyze-only, run, verify; application override preserved in provenance; installed entry-point separately checked below. |
| Notebooks | All three notebooks executed in a clean copy; no function/class definitions in notebook code cells; model counts stayed `[4,4,4]` across run, inspection and extension notebooks. |
| Real distributed execution | Two CPU Gloo DDP ranks; weights/loss compared with single-process global batches; a final global batch containing one row tested an empty second-rank shard; rank-zero-only analysis. |
| Parallel jobs | Two independent CPU workers processed distinct jobs, shared one truth-observation product, and reused model files on a second identical distributed launch. |

The three skipped tests are single-CUDA full precision, single-CUDA float16 AMP,
and two-CUDA NCCL DDP. They are included in `tests/test_distributed.py` and have
explicit hardware guards. Their presence is **not** a claim they passed here.
Multi-node execution and other OS/Python/dependency-version combinations were not
tested. No standalone linter run is claimed.

## Larger, original-size one-dimensional check

In addition to small automated tests, the following was actually run:

```bash
python validation/run_reference.py legacy_1d --output /mnt/data/reference_legacy_1d
```

This selects dimension one from the `legacy` preset, fixes the execution device
to CPU and changes the output destination. It retains the scientific/training and
analysis defaults: four priors, **70,000 training + 15,000 validation + 15,000 test
rows per prior**, four 64-unit hidden layers, Adam .001, five epochs, 15 observations
at each of 25 truth points, 16,384 continuous candidates, native grid candidates,
256 verification pairs, and the one-million-sample fixed reweighting showcase.
All selected metrics and plots completed. Runtime was **210.9 seconds** on this
validation machine; this is not a hardware-independent performance benchmark.

| Prior | Held-out BCE | Held-out AUC | Mean absolute parameter error | Projected 68% coverage | Projected 95% coverage |
|---|---:|---:|---:|---:|---:|
| Uniform | 0.540914 | 0.774522 | 0.478807 | 0.60 | 0.88 |
| Normal | 0.557789 | 0.762329 | 0.512325 | 0.52 | 0.92 |
| Exponential | 0.546799 | 0.767815 | 0.506741 | 0.44 | 0.92 |
| Grid | 0.543451 | 0.773421 | 0.632000 | 0.44 | 0.80 |

These are the observed results, not nominal targets or a reproduction of the
paper's plotted values. The single 25-case design is small, coverages are sometimes
below nominal, and learned reweighting differs visibly from the exact target.
Passing software/reference tests does not establish that five-epoch networks are
well calibrated. Different random streams, numerical choices and grid measure
also prevent a bitwise legacy comparison. No full 1D–6D scientific reproduction
was performed.

Evidence: `validation/legacy_1d.log`, `legacy_1d_results.json` (resolved settings,
provenance and all metrics), and `legacy_1d_summary.csv`. Byte-level integrity
verification passed for **41** committed artifacts in that run's store.

## Mixed-parameter and architecture study

The complete `mixed` example was executed on CPU: **eight trained models**, from
four OFAT configurations and a fixed two-prior cohort. Configurations compare
all means versus all means plus the second observation's standard deviation,
MLP versus residual network, and the alternative learning rate, always holding
other knobs at baseline. Training, all selected metrics and plots completed in
**31.6 seconds**. The mixed profile is intentionally small and is a functionality
check, not a converged inference result.

Evidence: `validation/mixed.log` and `validation/mixed_results.json`. Byte-level
integrity verification passed for **72** committed artifacts. Representative
mean/std projections and one-dimensional inference, training, pairwise and
reweighting plots were visually inspected; copies are in
`validation/example_plots/`.

## Packaging and release checks

All Python source and test files passed `compileall`. An offline wheel build
using `pip wheel --no-deps --no-build-isolation` succeeded. That wheel was installed
into a separate virtual environment, and the installed `nnpd` entry point, called
from outside the source tree, produced the expected eight-job mixed plan.
Dependencies were reused from the existing validation environment; a fresh
network dependency installation was not tested.

Evidence: `validation/packaging.log` and `validation/installed_plan.json`.
A release-candidate ZIP was extracted into a clean directory and its CLI smoke
run completed all four models. Repeated analysis left every checkpoint modification
time unchanged, and deep verification passed for 41 artifacts. All 38 executable,
test and configuration files matched the release source by SHA-256. Details are
in `validation/release_check.json` and `validation/release_smoke.log`. The final
archive adds the resulting report files without changing that tested code.
Large runtime caches are intentionally not shipped;
the package includes scripts/configuration to recreate them and compact evidence
of the executions above.

## Size and reproducibility scope

Physical line counts (including comments and blank lines), recorded in
`validation/source_inventory.json`: the old active entry point plus `src` contain
6,327 Python lines; the new framework/reusable NRE code and Gaussian application
contain 2,426 lines before root settings/entry-point files. Tests and documentation
are excluded from both counts. The counts are not a feature-equivalence claim:
legacy duplicate plotting, exploratory notebook code and cluster-specific launch
policy were deliberately not ported verbatim.

Source-file hashes, resolved settings, named random streams, dependency recipes,
backend identifiers and safe saved arrays improve traceability. They cannot
promise identical training across arbitrary hardware/library versions or validate
unseen user extensions. See `docs/SCIENTIFIC_NOTES.md` for interpretation and
`docs/STORAGE.md` for cache/restart limitations.
