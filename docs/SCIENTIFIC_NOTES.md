# Scientific notes

## Basis and scope

The Gaussian application implements the study described in *Prior Dependence in
Neural Ratio Estimation*. This page records what that paper specifies, how its
quantities map to settings, where the implementation has to choose an
interpretation, and which numerical limits apply when reading results.

## What the paper specifies

Page 1, equations (1)–(2), gives the balanced joint-versus-factorized classifier
construction and classifier-odds identity. Page 2, section 2, specifies diagonal
Gaussian observations in dimensions one through six, mean parameters in
`[0,10]^d`, standard deviation 2, the four prior families, 100,000 total examples,
a 70/15/15 split, four hidden layers with 64 ReLU units, five epochs, and 15
observations per inference ensemble. Equation (4) normalizes the product of
likelihood ratios over parameter space. Pages 2–3 define the likelihood-ratio
maximizer, bias, HLD region and its coordinate-projected boundaries.

Figure 1 on page 3 illustrates one- and two-dimensional projections for a 3D
exponential-prior example. Figure 2 on page 4 compares one-dimensional estimates
and HLD intervals across priors. Figure 3 shows dimension-wise coverage and
absolute-bias/width summaries. The paper reports little prior dependence in its
study. That conclusion comes from the paper; small smoke or reference runs of this
package do not establish it anew.

## Settings for the study's quantities

| Quantity | Setting | `default` value |
|---|---|---|
| Observation dimension | `problem.dimension` | `Choice([1,2,3,4,5,6])` |
| Inferred coordinates | `problem.infer` | `["mean:*"]`: all Gaussian means |
| Fixed observation standard deviation | `problem.fixed_std` | `2.0` |
| Mean domain | `problem.parameters.mean.low/high` | `[0,10]` |
| Prior families | `cohort.priors`, `priors` | uniform, normal, exponential, grid |
| Normal prior | `normal_mean`, `normal_std` | `5`, **`4` as a standard deviation** |
| Exponential shape | `exponential_rate` | `+0.1` (`paper`: `-0.1`) |
| Discrete grid | `grid_step` | `0.2`: 51 support points per mean axis |
| Training/validation/test rows | `data.train_size`, `validation_size`, `test_size` | 70,000 / 15,000 / 15,000 |
| Architecture | `model.hidden` | `[64,64,64,64]`, ReLU |
| Optimizer | `training.optimizer`, `learning_rate` | Adam, `0.001` |
| Training | `training.epochs`, `batch_size`, `cohort.replicas` | 5 epochs, batch 128, one model per prior |
| Ensemble size | `inference.observations` | 15 |
| Truth count | `truth_points_per_axis` | 25 per axis |
| Truth margin | `truth_margin_fraction` | `0.1` of the domain width, hence `[1,9]` for means |
| Large truth designs | `truth_design`, `max_grid_truths`, `sobol_truths` | `auto`: a grid up to 1,000,000 truths, otherwise 65,536 Sobol truths |
| Candidate approximation | `candidate_count`, `inference.seed` | 16,384 Sobol candidates, QMC seed 2026 |
| Nominal region masses | `levels` | `[0.68, 0.95]`, enclosed masses rather than tail probabilities |
| Pairwise verification | `verification.pairs`, `samples_per_endpoint` | 256 pairs, 512 samples per endpoint |
| Reweighting verification | `source_samples`, `target_samples`, `directions` | 2,048 source and target samples; 32 directions |
| Marginal ratio check | `marginal_samples`, `normalization_thetas` | 1,000 observations; 5 theta settings |
| Fixed showcase | `verification.showcase` | All means 3 to 7; 1,000,000 samples; 1,000 bins |
| Prior histogram | `plotting.prior_points`, `bins` | 200,000 samples, 100 bins |

The standard-deviation inference bounds `[.2,4]` and associated prior parameters
are **example choices of this implementation**. The paper's study infers means
only. Change them in the same settings file.

## Interpretation and implementation choices

**Exponential sign.** The paper states a rate of `-0.1`. The `default` preset uses
`+0.1`; the `paper` preset uses `-0.1`. The implemented bounded density is
proportional to `exp(-rate * theta)` on the finite parameter box. A positive rate
therefore decreases toward the upper bound, and a negative rate increases. Either
sign is well-defined after normalization. A negative rate is never passed to an
unbounded exponential sampler.

**Normal notation.** The paper writes `N(5,4)` without saying in that expression
whether 4 is a variance or a standard deviation. This implementation uses a
standard deviation of 4 (`normal_std=4`).

**Evaluation batches.** `training.evaluation_batch_size` (65,536 rows by default)
sets how many network inputs are evaluated at once during prediction and
inference. It is a memory setting, not a change in the evaluated quantity.

**Discrete measure.** A grid prior is a probability mass function, not a continuous
PDF that is zero almost everywhere. `default` uses the grid's native counting
measure (or a QMC approximation when its support is too large). The `paper` preset
uses continuous candidates for the normalized ratio, reflecting the continuous
integral in equation (4), and disables posterior inference for that choice.
This is an explicit interpretation: the paper does not fully specify how its
continuous integral is numerically reconciled with a discrete training prior.
Off-lattice predictions of a grid-trained network require interpolation that is
not constrained by training at every off-lattice theta. Comparisons using counting
versus Lebesgue measure are therefore not interchangeable. Set
`native_grid_prior=False` with ratio/exact targets to use common continuous
candidate measure across priors; a discrete posterior requires native support.

**Truth points and repeats.** `align_truths_to_grid` snaps continuous truth
coordinates to the nearest eligible lattice points of each parameter's
`grid_step`. The flag is independent of which priors are in the cohort. The truth
set therefore stays the same when comparing subsets of priors. Snapping can
produce duplicate truths; they are retained and reported rather than silently
removing inference cases. `repeats` creates independently sampled observation
ensembles at each truth; `observations` is the number within each ensemble;
`replicas` controls independently trained models. These are three separate
concepts.

**Sampling and training.** Each stochastic step draws from its own named,
deterministic random stream rather than one mutable global generator. Truncated
normal samples use inverse-CDF sampling rather than rejection. Each data split is
exactly class-balanced and therefore requires an even row count. The classifier
returns logits, is trained with BCE-with-logits, and inference uses those logits
directly as log ratios. This avoids potentially saturated sigmoid odds. Epoch loss
is sample-weighted. A fixed seed does not promise identical weights across
devices, world sizes, or dependency versions.

## Mathematical clarification: ratio, likelihood shape and posterior

The following is the implementation's explicit algebraic interpretation, not a
claim that the paper used these exact words. With balanced classes, an ideal
logit is

```text
ell_i(x, theta) = log p(x | theta) - log p_i(x).
```

The evidence `p_i(x)` depends on the training prior. Thus the unnormalized
likelihood-to-evidence ratio is not literally identical across priors at a fixed
`(x,theta)`. Its dependence on that evidence is constant with respect to theta.
It cancels in a theta-to-theta likelihood ratio and when normalizing an ensemble
product over theta. This is the distinction relevant to the parameter-inference
comparison in equation (4).

For observations `x[1],...,x[m]` sharing one theta, the targets are:

```text
ratio:      log score(theta) = sum_k ell_i(x[k], theta)
posterior:  log score(theta) = sum_k ell_i(x[k], theta) + log prior(theta)
exact:      log score(theta) = sum_k log p(x[k] | theta)
```

Each is normalized using the declared candidate measure. The posterior includes
one prior factor, not `prior(theta)**m`. Here the posterior prior is the configured
training prior; a separately chosen inference prior can be implemented as a
custom product. The paper's ratio metrics do not acquire a prior factor.
All internal logarithms are natural logarithms.

## HLD, coverage, bias and finite numerical resolution

HLD candidates are ranked by **density** with respect to the declared measure;
quadrature mass is used for cumulative probability, not for density ordering.
Ties at the cutoff are all included, so actual enclosed mass can exceed the nominal
level. `resolution.mean_enclosed_mass` reports that value. The mode maximizes
density, not quadrature-weighted mass.

The saved HLD coordinate bounds are minima/maxima of the selected **joint** region.
They are not separately computed one-dimensional marginal credible intervals,
and their bounding box may contain points outside the joint region. Multimodal
sets can have gaps within the reported projection bounds, as in the extrema-based
definition in the paper's equation (7). Therefore the output distinguishes:

- `coverage.projected_mean`: true coordinate inside the projected bounds, averaged
  across cases and inferred coordinates;
- `coverage.joint_density`: the truth's evaluated joint density exceeds the
  numerical HLD threshold, averaged across cases.

The paper's prose description of coverage refers to estimated values, whereas
its operational comparison uses the true parameter. This package explicitly uses
**true-parameter membership**. Checking whether the maximizer lies in a region
defined around that maximizer would not be a useful coverage test. One random
ensemble at each of 25 truth points gives only 25 empirical 1D cases; it does not
establish precise frequentist coverage at every fixed parameter.

Both signed bias (equations (5)/(8)) and mean absolute bias (figure 3's label) are
saved, avoiding confusion between absolute mean signed bias and mean absolute
error. The figures do not present an HLD width as a standard error on a
cross-case mean bias.

`exact_reference` evaluates the analytic Gaussian likelihood **on the same finite
candidate approximation**. Its difference from the learned estimate diagnoses
model error, but it does not certify candidate integration accuracy. Increasing
dimension or inferring a narrow standard deviation can leave far too few candidates
near a peak. Use candidate-count convergence studies, adequate truth repeats and
training replicas; inspect effective sample size and attained HLD masses. A low
candidate ESS can indicate poor resolution but is not itself a convergence bound.
The test suite separately checks fine-grid 1D Gaussian HLD widths against an
analytic result and tests that 2D joint projections differ from marginal intervals.

## Reweighting and normalization diagnostics

Pairwise diagnostics compare `ell(x,theta1)-ell(x,theta0)` with the exact Gaussian
log-likelihood ratio on shared samples drawn from both endpoints. Reweighting
uses normalized positive weights from that same log-ratio on source samples;
weighted one-dimensional Wasserstein-1 distances are averaged over saved random
unit directions. Exact-ratio and unweighted baselines are stored alongside the
learned result. Even exact weights can have poor finite-sample performance when
source and target overlap little; a small importance-weight ESS makes that visible.

The normalization check estimates `E_{p_i(x)}[r_i(x,theta)] = 1` using observations
from **each model's own prior-predictive evidence**, not one common wrong evidence.
It reports log mean ratio and its deviation from zero, with a finite-sample caveat.
All these observation sets and intermediate predictions are reusable products;
changing which metrics are requested does not resample the underlying experiment.
