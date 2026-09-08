# Troubleshooting

## The default command did not train anything

That is intentional: `python run.py` uses the `plan` action. Use
`python run.py run --profile smoke` for a small end-to-end run.

## The plan printed successfully but execution rejects it

Inspect each workload's `errors` and `warnings`. The planning API reports
inference guards; `run` and `analyze` enforce them. Explicitly reduce the relevant
candidate/truth/retention settings or allocate adequate resources. The code does
not silently reduce an exhaustive paper grid to fit a guard.

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

The repository must contain `.github/workflows/docs.yml` at its root; uploading
only the ZIP does not install the workflow. Enable **Settings → Pages → Build
and deployment → Source: GitHub Actions**. Push to the default branch or manually
run the documentation workflow on that branch. Inspect the build/deploy jobs in
the Actions tab. Detailed steps and permission caveats are in
[Publishing the website](../publishing.md).
