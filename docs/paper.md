# The paper

NNPD implements the study of

> **Prior Dependence in Neural Ratio Estimation**<br>
> Nabil Salama, Katherine Fraser, Po-Wen Chang, Benjamin Nachman<br>
> 2026 Conference on Physics and AI (PAI26)

[Read it on OpenReview](https://openreview.net/pdf?id=kdy81q0tOX) ·
{download}`Download the PDF <NNPD_paper_PAI26.pdf>`

## Abstract

Neural ratio estimation is an effective technique for using neural networks to
learn likelihood ratios from simulated data, which can be used for applications
such as parameter inference and reweighting. While it can be theoretically shown
that in principle the resulting likelihood-to-evidence ratio should be independent
of the prior used to select the training data, that independence relies on
achieving an idealized classifier, which is not possible in practice. It is
therefore unclear to what extent prior independence survives under realistic
training conditions, particularly for more complicated higher-dimensional
problems. In this paper, we test the prior independence of parameter estimation
for a simple Gaussian toy example. We plan to extend this study to realistic
particle physics datasets in the future.

## Cite

```text
@inproceedings{salama2026prior,
  title     = {Prior Dependence in Neural Ratio Estimation},
  author    = {Salama, Nabil and Fraser, Katherine and Chang, Po-Wen and Nachman, Benjamin},
  booktitle = {2026 Conference on Physics and AI (PAI26)},
  year      = {2026},
  url       = {https://openreview.net/pdf?id=kdy81q0tOX}
}
```

## From the paper to the code

| Paper | Code | Output |
|---|---|---|
| Eq. (1), (3): joint versus factorized training classes | `nnpd.nre.data.balanced_pairs`, `gaussian/sampling.py` | `training` artifact |
| Eq. (2): classifier odds equal the likelihood-to-evidence ratio | `nnpd.nre.training.fit_binary`; the network returns the log ratio as a logit | `model` artifact |
| Section 2: Gaussian data, four priors, network and training | `settings_training.py`, `gaussian/experiment.py` | — |
| Eq. (4): ensemble of 15 observations, normalized over parameter space | `nnpd.nre.inference.ensemble_log_ratio`, `normalized_mass` | `inference` product |
| Eq. (5): likelihood-ratio maximizer and bias | `ratio_mode` in the `inference` product | `bias` metric |
| Eq. (6): HLD region | `nnpd.nre.inference.summarize_density` | `inference` product |
| Eq. (7): projected HLD bounds and widths | `ratio_lower`, `ratio_upper` | `width` metric |
| Eq. (8): mean coverage, width and bias | `gaussian/metrics.py` | `coverage`, `width`, `bias` metrics |
| Figure 1: one- and two-dimensional projections | `inference` plot | `marginal-*` and `projection-*` figures |
| Figure 2: estimates and HLD intervals per prior | `inference` plot | `estimates-*` figures |
| Figure 3: metrics across priors and dimensions | `nnpd.results.comparison_plot` | built from saved metrics, see below |

The [scientific notes](SCIENTIFIC_NOTES.md) map every quantity of the paper to its
setting and explain where the implementation has to choose an interpretation:
the exponential sign, the normal-prior scale, the truth design, the discrete
integration measure, and coverage of the true parameter.

## Run the study

The `paper` profile makes the paper's choices explicit: exponential rate `-0.1`,
no truth margin, the exact 25-point-per-axis truth grid where it is feasible
(up to four inferred parameters, Sobol truths above), continuous candidates for
the normalized ratio, and ratio and exact targets.

```bash
python run.py plan --profile paper
python run.py run --profile paper
```

Read the plan first: the full study trains 24 models, four priors in each of six
dimensions. After the run, compare the priors across dimensions as in Figure 3:

```python
from nnpd.results import comparison_plot, runs

records = runs("outputs/paper")
coverage = comparison_plot(records, "settings.problem.dimension", "metrics.coverage.projected_mean.0")
bias = comparison_plot(records, "settings.problem.dimension", "metrics.bias.mean_absolute")
```

`projected_mean.0` is the coverage of the first level in `settings_analysis.py`,
68% by default. The runs use their own random streams and numerical choices, so
they reproduce the study, not the paper's exact figures.

The paper restricts itself to parameter inference. The analysis also computes the
reweighting and normalization diagnostics that its conclusion anticipates for a
longer version; see [Products, metrics, and plots](guide/analysis.md).
