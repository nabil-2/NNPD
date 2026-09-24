# Troubleshooting

## The default command did not train anything

That is intentional: `python run.py` uses the `plan` action. Use
`python run.py run --profile smoke` for a small end-to-end run.

## "Analysis is not feasible for these configurations"

A configuration's estimated analysis exceeds `max_model_evaluations`,
`max_saved_bytes` or `max_candidate_points`. Nothing was computed. The message
names each configuration and the settings to reduce: typically the truths
(`truth_design`, `sobol_truths`, `truth_points_per_axis`), `repeats`,
`observations` or `candidate_count`. Raise a limit only if the hardware can
afford it. Nothing is reduced automatically.

## Analysis says no matching checkpoint exists

`analyze` never trains implicitly. First run `train` or `run` with matching
scientific/training settings and a compatible backend. Adding analysis hooks can
reuse a model, but changing its training recipe intentionally selects a distinct
artifact. For loading existing weights on a different device, use `restore`.

## A list did not produce several configurations

Literal lists are one setting value. Use `Choice([baseline, alternative, ...])`
for alternatives. `Choice([[64, 64], [128, 128]])` is a network-width comparison;
`[64, 64]` by itself is just one two-layer architecture.

## A selected parameter does not exist

Parameter indices start at zero. `std:1` needs at least two observation
dimensions. OFAT alternatives are checked against the baseline, not against a
helpful combination of other alternatives. Use a compatible baseline or one
atomic dictionary choice for coupled settings.

## An extra metric unexpectedly rebuilt a dataset

Check its product settings selector, dependency keys, source dependencies,
explicit version, output root, and recorded package versions. Cache keys are
conservative. Products whose builders read new settings must declare them;
removing dependencies just to force cache reuse would undermine correctness.

## Results of earlier settings are gone

That is intended: in an output folder the latest settings win, and the cache
keeps only what the last three distinct settings used. The settings of the latest
launch are saved as `outputs/<profile>/settings.py`. To keep several experiments side
by side, give each its own profile. See
[Change settings and run again](quickstart.md#change-settings-and-run-again).

## A result was moved and no longer loads

Move the whole output root, including `cache`, `source`, and `runs`, rather than
one individual run directory. Relative references require their corresponding
artifact store. `verify` checks cache integrity but does not recreate missing
artifacts.

## CUDA or distributed execution fails immediately

Confirm a CUDA-enabled PyTorch installation, visible devices, and worker count.
Use `device="cuda"` under `torchrun`; indexed `cuda:N` settings are for a single
process. The [CUDA guide](cuda.md) separates independent jobs from collective DDP.
Physical GPU availability is not checked merely by generating a plan.

## Coverage or reweighting looks poor

Do not infer scientific calibration from passing software tests. Inspect
candidate resolution, attained HLD masses, truth repeats, training replicas,
reference likelihood results, and overlap/importance-weight ESS. Read the
[scientific notes](../SCIENTIFIC_NOTES.md) for the implemented definitions and
finite-candidate limitations.

## The documentation has not appeared on GitHub

The repository must contain `.github/workflows/docs.yml` at its root. Enable
**Settings → Pages → Build and deployment → Source: GitHub Actions**. Publishing
is manual: pushes only build-check the documentation, so run the documentation
workflow on the desired branch with `deploy` ticked. Inspect the build/deploy jobs
in the Actions tab. Detailed steps and permission caveats are in
[Publishing the website](../publishing.md).
