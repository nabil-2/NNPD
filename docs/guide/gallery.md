# Existing validation plots

These images were already included in the original codebase archive under
`validation/example_plots`. The documentation displays them without changing or
regenerating them. Their provenance and limitations are recorded in the
[original application test report](../TEST_REPORT.md).

The one-dimensional images come from the original-size `legacy_1d` validation
run. The mixed mean/standard-deviation image comes from the smaller `mixed`
functionality run. They are **not newly reproduced paper figures** and do not
establish convergence of the full study.

## Training and parameter inference

```{figure} ../../validation/example_plots/legacy_1d_uniform_training.png
:alt: Original one-dimensional uniform-prior training and validation loss plot.

Saved training-loss diagnostic for the uniform-prior model.
```

```{figure} ../../validation/example_plots/legacy_1d_uniform_inference.png
:alt: Original one-dimensional uniform-prior parameter estimates with HLD projections.

Saved parameter-estimation diagnostic. HLD projections are not standard errors
on an average bias.
```

```{figure} ../../validation/example_plots/legacy_1d_uniform_marginal.png
:alt: Original uniform-prior marginal comparison of ratio, posterior, and exact targets.

Saved marginal probability-mass comparison for a retained diagnostic case.
```

## Pairwise ratios and reweighting

```{figure} ../../validation/example_plots/legacy_1d_uniform_pairwise_worst.png
:alt: Saved worst-pair learned versus exact log-ratio diagnostic.

An existing pairwise log-ratio diagnostic; consult the saved metrics for values.
```

```{figure} ../../validation/example_plots/legacy_1d_uniform_reweighting.png
:alt: Existing learned, exact, and unweighted reweighting comparison.

Saved reweighting comparison. Finite-sample overlap and learned-ratio error remain
relevant even when software checks pass.
```

## Mixed inference parameters

```{figure} ../../validation/example_plots/mixed_mean_std_projection.png
:alt: Existing joint projection involving inferred Gaussian mean and standard deviation.

Example from the mixed-parameter functionality study, not a converged scientific
comparison.
```
