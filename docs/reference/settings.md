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

`training.batch_size` is global even under DDP. HLD `levels` are **enclosed
masses**, not tail probabilities. `inference.retain="diagnostic"` keeps score
rows for a subset while preserving per-case summaries; `"full"` retains all
score rows. Storage guards estimate selected inference arrays, not complete
peak RAM, disk use, or wall time.

## Presets

| Preset | Purpose |
|---|---|
| `legacy` | The active old Python entry-point settings with documented engineering changes. |
| `paper` | Explicit paper-facing changes: exponential sign, exhaustive truths, continuous ratio-candidate measure, and ratio/exact targets. |
| `smoke` | Small four-prior end-to-end run. |
| `mixed` | Small OFAT comparison of mean/standard-deviation parameter sets, architecture, and learning rate. |
| `extended` | Additional metrics/plot; shares the smoke output root to demonstrate cache reuse. |
| `smoke_ddp` | Small uniform-prior example with collective training selected. |

The existing [scientific notes](../SCIENTIFIC_NOTES.md) describe the source
settings and interpretation differences in detail.

## Complete settings and preset definitions

```{literalinclude} ../../settings.py
:language: python
:linenos:
:caption: settings.py (read from the unchanged source file)
```
