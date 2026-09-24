# Testing and validation

## Run the test suite

Install the development dependencies (`python -m pip install -e ".[dev]"`), then
run from the repository root:

```bash
python -m pytest -q
python -m pytest -q --cov=nnpd --cov=gaussian --cov-report=term-missing
python -m pytest -q -m distributed
python -m pytest -q -m cuda       # executes on suitably provisioned CUDA hardware
```

The plain command runs everything, including notebook and CPU distributed tests.
Markers select subsets:

| Marker | Meaning |
|---|---|
| `distributed` | Launches real `torchrun` processes. CPU runs use the Gloo backend. |
| `cuda` | Requires CUDA hardware; skipped automatically when no suitable device exists. |
| `notebook` | Executes the three notebooks in a clean copy of the repository. |

Skipped CUDA tests are not passed tests. Check the `-ra` summary at the end of a
run to see what was skipped and why.

## What the tests cover

| Area | File | Checks |
|---|---|---|
| Configuration | `test_config.py` | Exact five-run OFAT example; no Cartesian product; duplicate baseline removal; atomic list/dictionary options; literal architecture lists; invalid alternatives rejected before training; preset contents. |
| Scientific algebra | `test_science.py` | Normalization and sampler moments for uniform, truncated normal and signed exponential priors; grid mass and native support; mixed continuous/discrete supports; Gaussian likelihood sampling; balanced joint/product pairs; fine-grid 1D HLD versus the analytic interval; 2D joint projections versus marginal intervals; density ranking and cutoff ties; extreme-score stability; ensemble log-ratio chunking; both networks; weighted Wasserstein-1 versus SciPy. |
| Training modes | `test_modes.py` | AdamW, residual network, gradient clipping, best checkpoint, full precision and bfloat16 on CPU; Sobol truths; disabled targets; runtime validation. |
| Artifacts | `test_storage.py` | Source fingerprints; failed stages never become cache hits; array round trips, memory maps and corruption detection; unsafe names and object arrays rejected; shared products built once; dependency cycles rejected before training. |
| Pipeline | `test_pipeline.py` | Four-prior sampling, training and inference; model reuse after adding metrics; learning-rate sweeps reusing data; relocation of an output root; train-then-analyze without implicit training; recovery from a failed stage; analytic oracle and a single prior factor; mean/standard-deviation parameter sets; plots and result readers; workload guards; a separate non-Gaussian experiment. |
| CLI | `test_cli.py` | `plan` without side effects; `train`, `analyze`, `verify`; application override. |
| Notebooks | `test_notebooks.py` | All three notebooks execute; no function or class definitions in notebook cells; inspection and extension notebooks do not retrain models. |
| Distributed | `test_distributed.py` | Two CPU DDP ranks match single-process global batches, including an empty last shard; two independent job workers share observations. CUDA single-device (full precision, float16 AMP) and two-GPU NCCL DDP tests need hardware. |
| Documentation | `test_docs_check.py` | The documentation checker detects broken pages, fragments, assets, includes, Python syntax, and root-relative URLs. |

Multi-node launches are not covered by the test suite. The tests are evidence for
the implementation, not a guarantee of scientific convergence for arbitrary
high-dimensional configurations.

## Reference runs

The test suite uses tiny configurations. A larger reference run exercises
realistic sizes on CPU. Keep its output outside the repository:

```bash
python validation/run_reference.py --output ../nnpd-reference
```

It runs the `default` preset at dimension one: four priors, 70,000 training
plus 15,000 validation and 15,000 test rows per prior, four 64-unit hidden layers,
Adam at 0.001 for five epochs, 15 observations at each of 1,024 Sobol truth
points, 16,384 continuous candidates, native grid candidates, 256 verification
pairs, and the one-million-sample reweighting showcase. With the default single
CPU thread (`runtime.cpu_threads = 1`) it takes roughly 20 minutes; more threads
or a GPU make it considerably faster.

It writes `results.json` into the output directory, holding the resolved
configurations, provenance and all metrics. Plots are under each run's `plots/`
folder. For a byte-level integrity check of the store, call
`nnpd.core.runner.verify_store("<output directory>")`.

Treat this as a functionality and sanity check, not a converged scientific result.
Five-epoch networks need not be well calibrated, and learned reweighting can
differ visibly from the exact target. Different random streams, numerical backends
and library versions change individual numbers.

## Documentation checks

The documentation workflow runs these on every push to `main` and every pull
request:

```bash
python docs/check.py
python -m sphinx -b html -W --keep-going -E -a docs docs/_build/html
python docs/check.py --html docs/_build/html
```

The first command checks page links, navigation targets, included source files,
downloads, images and Python example syntax. The last checks internal links,
fragment targets and assets in the generated HTML. See
[Publishing](../publishing.md) for the environment setup.

## Reproducibility scope

Source-file hashes, resolved settings, named random streams, dependency recipes,
backend identifiers and safe saved arrays make results traceable. They cannot
promise identical training across arbitrary hardware or library versions, or
validate unseen user extensions. See the [scientific notes](../SCIENTIFIC_NOTES.md)
for interpretation and [Storage](../STORAGE.md) for cache and restart limitations.
