# Settings reference

`settings.py` is the source of truth. This page groups its responsibilities and
then includes the file verbatim. Website builds read the file; they do not import
it or execute `make_config`.

## Sections

| Section | Responsibility and important conventions |
|---|---|
| `application`, `output`, `seed` | Trusted `module:class` application, output root, base random seed. |
| `problem` | Observation dimension, ordered inferred parameters, fixed likelihood coordinates, bounds and prior-shape parameters. |
| `cohort` | Priors and independent training replicas repeated within a configuration. |
| `priors` | Named prior definitions; `by_parameter` may choose a family for an inferred coordinate. |
| `data` | Training, validation, and test row counts. Each must be even for exactly balanced classes. |
| `model` | `mlp` or `residual`; hidden widths or residual width/blocks; activation and dropout. |
| `training` | Epochs, global batch size, optimizer, learning rate, precision, clipping, checkpoint policy, evaluation batch size. |
| `inference` | Truth design, observation ensembles, candidate measure/resolution, HLD levels, targets, score retention, and workload guards. |
| `verification` | Shared pairwise-ratio, reweighting, normalization, and showcase data. |
| `metrics`, `plots` | Literal lists of registered hook names to run. |
| `plotting` | Figure resolution, histogram sizes, prior samples, and displayed inference cases. |
| `runtime` | CPU/CUDA selection, independent jobs versus DDP, CPU threads, determinism, process-group timeout. |

`inference.truth_design` is `sobol` (default), `grid` or `size_adaptive`; see
[Truth design](../guide/configuration.md#truth-design). The `max_*` inference
settings are the analysis limits checked before anything is computed.
`training.batch_size` is global even under DDP. HLD `levels` are **enclosed
masses**, not tail probabilities. `inference.retain="diagnostic"` keeps score
rows for a subset while preserving per-case summaries; `"full"` retains all
score rows. Storage guards estimate selected inference arrays, not complete
peak RAM, disk use, or wall time.

Each successful launch also copies the settings file it started from to
`outputs/<profile>/settings.py`; see
[Change settings and run again](../guide/quickstart.md#change-settings-and-run-again).

## Presets

| Preset | Purpose |
|---|---|
| `default` | The full study exactly as defined in `SETTINGS`, dimensions one through six. |
| `paper` | Explicit paper-facing changes: exponential sign, exhaustive truths, continuous ratio-candidate measure, and ratio/exact targets. |
| `smoke` | Small four-prior end-to-end run. |

The [scientific notes](../SCIENTIFIC_NOTES.md) map these settings to the paper
and explain the `default`/`paper` differences in detail.

## Complete settings and preset definitions

```{literalinclude} ../../settings.py
:language: python
:linenos:
:caption: settings.py
```
