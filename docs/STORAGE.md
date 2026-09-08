# Storage, caching and reproducibility

## Layout

Each selected output root owns its own store. For example:

```text
outputs/smoke/
  source/<source-hash>.zip
  cache/
    training/<key>/
      artifact.json
      train_x.npy, train_theta.npy, train_y.npy
      validation_*.npy, test_*.npy
    model/<key>/
      artifact.json
      weights.pt
      history.json
    observations/<key>/...
    candidates/<key>/...
    inference/<key>/...
    verification_observations/<key>/...
    verification_predictions/<key>/...
    reweighting_distances/<key>/...
    ...
  runs/
    baseline-<configuration-hash>/
      uniform-r0-<job-hash>-<backend-hash>/
        run.json
        status.json
        artifacts.json
        metrics.json
        metrics/<metric-name>.json
        plots/<plot-name>/<figure-name>.png
    training_learning_rate-<configuration-hash>/...
  locks/...
```

Folder hashes disambiguate settings; readable names identify the changed knob and
cohort member. `run.json` is the authority for the full configuration—do not decode
settings from a filename. A run identity includes the resolved configuration and
cohort member; backend identity is separate. The named metric list is part of the
run configuration, so adding metrics creates a new result record while reusing the
same underlying training/model artifacts where compatible.

`source/<hash>.zip` snapshots the framework, the experiment class's application
folder, local settings/entry-point/packaging files, and an explicitly supplied
custom settings file. Third-party packages, remotely imported helpers outside
those folders, external datasets and execution environments are not vendored by
this snapshot. The resolved settings and package/backend identifiers are also
saved independently in the run and artifact records.

## Artifact recipes

Every immutable artifact has a recipe and a manifest (`artifact.json`). A recipe
contains selected settings, dependency artifact keys, implementation source hashes,
version/environment identifiers, and—in the model case—the training backend.
Array manifests record dimensions, dtype, file size and SHA-256 checksums. Model
artifacts store tensor state dictionaries, not arbitrary pickled model instances.

Typical sharing in the Gaussian study:

| Change | Reused | Recomputed |
|---|---|---|
| Add a metric using existing inference | Training, weights, observations, inference | The selected metric outputs/plots |
| Change learning rate | Training data, common truth/verification observations, candidates | Model and model-dependent products |
| Change network architecture | Training data, common observations, candidates | Model and predictions/inference |
| Change selected parameter set | Only compatible products, if any | Data/model/layout-dependent products |
| Change HLD levels or retained scores | Data, model, truth ensembles, candidates | Inference product and consumers |
| Change plot labels/implementation | Scientific products and weights | Plots |
| Change CPU to CUDA or DDP world size | Compatible sampled data | Distinct model artifact |

Selectors are intentionally conservative in places. For example, changing an
unrelated field within `verification` may rebuild several verification products.
The inferred HLD summaries and retained scores are currently one product: changing
levels rebuilds inference rather than providing a separate automatic offline
threshold-recomputation engine. This is a simplicity/storage tradeoff. A custom
raw-score product can separate those operations for a more specialized workflow.

Different output roots do not share a global cache automatically. Use the same
root for related configurations and analysis-only extensions. The dedicated
`extended` preset does so for the smoke study.

## Numeric products and shape conventions

For `N` cases, `m` observations per case, `d` observation dimensions, `p` inferred
parameters, `K` candidates and `L` HLD levels:

| Product/array | Shape | Meaning |
|---|---|---|
| `observations/truth` | `(N,p)` | True parameter vector for each ensemble |
| `observations/observations` | `(N,m,d)` | Independent observations conditional on each truth |
| `candidates/theta` | `(K,p)` | Candidate parameter points |
| `candidates/log_measure` | `(K,)` | Log quadrature/counting weights |
| `inference/ratio_mode` | `(N,p)` | Maximizer on the candidate approximation |
| `inference/ratio_lower`, `ratio_upper` | `(N,L,p)` | Projected joint-HLD bounds |
| `inference/ratio_ess` | `(N,)` | Candidate mass effective sample size |
| `inference/ratio_enclosed_mass` | `(N,L)` | Attained mass including cutoff ties |
| `inference/ratio_joint_contains_truth` | `(N,L)` | Density-threshold truth membership |
| `inference/retained_cases` | `(R,)` | Indices of cases with retained scores |
| `inference/ratio_log_score` | `(R,K)` | Unnormalized ensemble log-ratio scores |

The `posterior_` and `exact_` prefixes use the same schema for enabled targets.
The candidate count can be smaller than the requested upper bound after native
grid enumeration or deduplication; actual counts and measure are recorded.
`log_score` values are not already normalized probability masses. Combine them
with `log_measure` and a log-sum-exp normalization, or use `normalized_mass`.

`retain="diagnostic"` keeps scores for only `diagnostic_cases` ensembles while
retaining all per-case summary arrays and original observation ensembles.
`retain="full"` retains every score row in an on-disk memory map. Both avoid
keeping the full N-by-K-by-m network input tensor in memory. Input evaluation is
batched; HLD summaries are calculated one case at a time. Very large candidate
sets still require memory and sorting per case.

The workload plan estimates inference evaluations and saved array storage; it is
**not a complete peak-RAM, disk-space or wall-time prediction**. Training data,
verification products, plots, intermediate allocation and multiple cache versions
add costs not represented by that inference estimate.

## Atomicity and interruption

A product is built in a temporary sibling directory under a per-artifact file
lock. The manifest is written only after data files exist and checksums are
computed; an atomic directory rename commits the finished product. Uncommitted
failed products are removed. Repeated calls resolve the same recipe to the same
artifact instead of duplicating a stochastic dataset.

A hard-killed process can leave an uncommitted temporary directory; it is not a
valid cache entry and can be removed when no writer is active. File locks must
work on the selected filesystem. Different torchrun `jobs` workers have disjoint
complete jobs and may safely share common products. Do not deliberately launch
two independent analyses writing the **same result directory** concurrently;
artifact writes are locked, but run-status/metric/plot orchestration is not a
transaction spanning the entire run.

Resume is stage-level. Fully committed data and checkpoints survive an interrupted
analysis. An interrupted training stage restarts training because optimizer state
and mid-epoch RNG/position are not saved. A failed status record contains a traceback.
The runner does not replace nonfinite scores with a fake uniform distribution.

## Integrity, movement and security

Ordinary cache access verifies manifest files exist and have the expected sizes.
For a full byte-level checksum verification:

```bash
python run.py verify --profile smoke
```

This recomputes checksums for all committed cached artifacts under the profile's
output root. It does not hash every plot or validate a scientific result. Same-size
file corruption is detectable by this explicit deep check, not by the fast check.

`artifacts.json` uses relative paths, so moving the **whole output root** preserves
reload references. Moving an individual run folder without its store does not.
`restore` defaults to CPU and does not inherit stale torchrun rank variables.
The original config remains a provenance record; it is not silently rewritten when
an output root is relocated.

Arrays are loaded with `allow_pickle=False`; checkpoints use
`torch.load(..., weights_only=True)`. Settings and application modules are trusted
Python code and execute normally; they are not sandboxed. These protections do
not make an arbitrary malicious download safe. No legacy pickle is loaded by the
new framework, and source snapshots do not include the uploaded paper or old code.

Cache cleaning is manual: keep the root intact for portable provenance or delete
an unused root once no processes are using it. There is no hidden cache eviction,
old-format migration or remote object-store backend.
