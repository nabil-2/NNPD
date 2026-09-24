# Settings reference

`settings_training.py` and `settings_analysis.py` are the source of truth. This
page groups their responsibilities and then includes both files verbatim. Website
builds read the files; they do not import them or execute `make_settings`.

## Training settings: what is trained

| Section | Responsibility and important conventions |
|---|---|
| `experiment`, `output`, `seed` | Trusted `module:class` experiment, output folder, base random seed. |
| `problem` | Observation dimension, ordered inferred parameters, fixed likelihood coordinates, bounds and prior-shape parameters. |
| `cohort` | Priors and independent training replicas repeated within a configuration. |
| `priors` | Named prior definitions; `by_parameter` may choose a family for an inferred coordinate. |
| `data` | Training, validation, and test row counts. Each must be even for exactly balanced classes. |
| `model` | `mlp` or `residual`; hidden widths or residual width/blocks; activation and dropout. |
| `training` | Epochs, global batch size, optimizer, learning rate, precision, clipping, checkpoint policy, evaluation batch size. |
| `runtime` | CPU/CUDA selection, independent jobs versus DDP, CPU threads, determinism, process-group timeout. Also used by `analyze`. |

`Choice` marks sweep alternatives here. `training.batch_size` is global even under
DDP. `training.evaluation_batch_size` also sets the model-evaluation batch during
analysis.

## Analysis settings: what is computed from the trained models

| Section | Responsibility and important conventions |
|---|---|
| `analysis` | Trusted `module:class` analysis: the registered products, metrics, and plots. |
| `inference` | Truth design, observation ensembles, candidate measure/resolution, HLD levels, targets, score retention, and analysis limits. |
| `verification` | Shared pairwise-ratio, reweighting, normalization, and showcase data. |
| `metrics`, `plots` | Literal lists of registered hook names to run. |
| `plotting` | Figure resolution, histogram sizes, prior samples, and displayed inference cases. |

`Choice` is rejected here: an analysis is one setting, and the latest one wins.
`inference.truth_design` is `sobol` (default), `grid` or `size_adaptive`; see
[Truth design](../guide/configuration.md#truth-design). The `max_*` inference
settings are the analysis limits checked before anything is computed. HLD
`levels` are **enclosed masses**, not tail probabilities.
`inference.retain="diagnostic"` keeps score rows for a subset while preserving
per-case summaries; `"full"` retains all score rows. The limits estimate selected
inference arrays, not complete peak RAM, disk use, or wall time.

Each successful launch also copies the settings files it used into
`outputs/<profile>/`; see
[Change settings and run again](../guide/quickstart.md#change-settings-and-run-again).

## Presets

Both files define the same presets; `--profile` selects one in both.

| Preset | Training | Analysis |
|---|---|---|
| `default` | The full study exactly as defined in `SETTINGS`, dimensions one through six. | The full analysis exactly as defined in `SETTINGS`. |
| `paper` | The paper's exponential sign. | Exhaustive truths while feasible, no truth margin, continuous ratio-candidate measure, and ratio/exact targets. |
| `smoke` | Small four-prior end-to-end run. | Small analysis of the smoke models. |

The [scientific notes](../SCIENTIFIC_NOTES.md) map these settings to the paper
and explain the `default`/`paper` differences in detail.

## Complete settings and preset definitions

```{literalinclude} ../../settings_training.py
:language: python
:linenos:
:caption: settings_training.py
```

```{literalinclude} ../../settings_analysis.py
:language: python
:linenos:
:caption: settings_analysis.py
```
