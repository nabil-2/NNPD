# Scientific interpretation and migration from the supplied sources

## Basis and scope

The supplied sources are `NNPD camera ready.pdf` (five pages, *Prior Dependence in
Neural Ratio Estimation*) and `NNPD-toy_example_poc.zip`. The source inventory and
SHA-256 checksums are recorded in `validation/source_inventory.json`. The paper,
active Python entry point, current notebook, historical exploratory notebooks,
analysis utilities, tests and launch scripts were inspected. The active entry
point is the baseline for the `legacy` preset; historical notebooks do not all
represent the same experiment.

The framework is newly structured. Its Gaussian application carries over the
active study's scientific ingredients while exposing ambiguities and implementation
changes rather than claiming identical random draws, checkpoints or published
figures. The old data/checkpoint/figure formats are not imported automatically.

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
study. That conclusion is a result of the supplied paper, not established anew
by this package's small validation runs.

## Mapping to the new settings

| Quantity | Active old source | New setting/default |
|---|---|---|
| Observation dimension | CLI `n_parameters`; notebook config | `problem.dimension = Choice([1,2,3,4,5,6])` |
| Inferred coordinates | All Gaussian means | `problem.infer = ["mean:*"]` |
| Fixed observation standard deviation | `data.std_dev = 2` | `problem.fixed_std = 2.0` |
| Mean domain | `(0,10)` | `problem.parameters.mean.low/high` |
| Uniform/normal/exponential/grid models | `build_priors`, prior samplers | `cohort.priors`, `priors`, parameter options |
| Normal prior | `(5,4)` | `normal_mean=5`, **`normal_std=4`** |
| Exponential shape | CLI/current notebook `+0.1` | `exponential_rate=+0.1` in `legacy` |
| Discrete grid | `step=0.2` | `grid_step=0.2`, 51 support points per mean axis |
| Training/validation/test | 70,000 / 15,000 / 15,000 | `data.train_size`, `validation_size`, `test_size` |
| Architecture | Initial dense layer plus `n_hidden_layers=3` | `model.hidden=[64,64,64,64]` |
| Optimizer | Adam, learning rate .001 | `training.optimizer`, `learning_rate` |
| Training | 5 epochs, batch 128, one model iteration | `training.epochs/batch_size`, `cohort.replicas=1` |
| Ensemble size | `n_repititions_per_parameter=15` | `inference.observations=15` |
| Truth count | 25 per axis | `truth_points_per_axis=25` |
| Truth margin | CLI `.1` of domain width | `truth_margin_fraction=.1`, hence `[1,9]` for means |
| Large truth designs | Auto grid limit 1,000,000; 65,536 Sobol truths | `truth_design`, `max_grid_truths`, `sobol_truths` |
| Candidate approximation | CLI `2**14`, QMC seed 2026 | `candidate_count=16384`, `inference.seed=2026` |
| Nominal region masses | 68%, 95% | `levels=[.68,.95]`, not tail probabilities |
| Pairwise verification | 256 pairs, 512 samples per endpoint | `verification.pairs/samples_per_endpoint` |
| Reweighting verification | 2,048 source and target; 32 directions | `source_samples/target_samples/directions` |
| Marginal ratio check | 1,000 observations; 5 theta settings | `marginal_samples/normalization_thetas` |
| Fixed showcase | All means 3 to 7; 1,000,000 samples; 1,000 bins | `verification.showcase` |
| Prior histogram | 200,000 samples, 100 bins | `plotting.prior_points/bins` |

The standard-deviation inference bounds `[.2,4]` and associated prior parameters
are **new example choices**, not values prescribed by the paper's means-only
study. Change them in the same settings file.

## Explicit source differences and engineering choices

**Exponential sign.** The paper states lambda `-0.1`; the active CLI and current
notebook use `+0.1`. `legacy` uses the latter and `paper` the former. The implemented
bounded density is proportional to `exp(-lambda * theta)` on the finite parameter
box, so either sign is well-defined after normalization. Negative rate is not
passed to an unbounded exponential sampler.

**Normal notation.** The paper writes `N(5,4)` without saying in that expression
whether 4 is a variance or a standard deviation. The supplied code explicitly
uses a standard deviation of 4; the new application follows that code convention.

**Current notebook versus CLI.** The current notebook uses zero inference margin,
65,536 candidate points, prediction batches of 32,768, and a larger evaluation
cap (`2e15`) than the CLI defaults. It also uses a 4-to-6 fixed reweighting showcase
with 100,000,000 samples. Those notebook settings are not silently substituted for
the active CLI preset. The new evaluator defaults to 65,536 prediction rows per
batch rather than the CLI's `2**21`; this is an engineering memory choice, not an
intended change in the evaluated mathematical quantity.

**Discrete measure.** A grid prior is a probability mass function, not a continuous
PDF that is zero almost everywhere. `legacy` uses the grid's native counting
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

**Truth points and repeats.** The current CLI aligns truths to the grid when the
grid prior participates. The new `align_truths_to_grid` flag is independent of
which priors are in the cohort, making that dependency explicit and stable when
comparing subsets. The new design snaps continuous truth coordinates to eligible
lattice points; it retains and reports duplicate truths rather than silently
removing inference cases. The old grid-axis builder selects evenly spaced eligible
support indices and guards distinctness. `repeats` creates independently sampled
observation ensembles at each truth; `observations` is the number within each
ensemble; `replicas` controls independently trained models. These are three
separate concepts.

**Sampling and training.** The new code uses dedicated deterministic random streams
rather than one mutable global generator; exact RNG sequences differ. Truncated
normal samples use inverse-CDF sampling rather than rejection. Each data split is
exactly class-balanced and requires an even row count, rather than a single
balanced dataset followed by an approximately balanced random split. The classifier
returns logits, with BCE-with-logits and direct log-ratio inference, instead of
forming potentially saturated sigmoid odds. Epoch loss is sample-weighted. These
changes deliberately prevent numerical/pathological behavior; they mean a fixed
seed is not a bitwise legacy reproduction.

**Historical notebooks.** Earlier 1D and 2D notebooks explore gamma/sinusoidal priors
and inference of a Gaussian mean plus standard deviation. The old nD notebook
also contains repeated implementations, unbounded samplers alongside bounded
normalizations, and posterior products that apply the prior once per observation.
The active `src/posterior.py` already applies the prior only once; the new code
preserves that active behavior. The historical alternatives are not treated as
requirements to reproduce bugs. Site-specific Slurm partition preferences,
VEGAS experiments, multiprocessing benchmarks, duplicate ROC/panel layouts and
hard-coded configuration-number analysis are not copied into the generic core.
The optional legacy `quadratic` sample-scaling shorthand is also not a hidden
rule: express actual desired sample sizes or atomic problem/data configurations
explicitly in the settings file.

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
All internal logarithms are natural logarithms; the old `r10` terminology means
ratio for theta-one over theta-zero, not a base-10 logarithm.

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
  across cases and inferred coordinates, matching the active code's convention;
- `coverage.joint_density`: the truth's evaluated joint density exceeds the
  numerical HLD threshold, averaged across cases.

The paper's prose description of coverage refers to estimated values, whereas
its operational comparison and active code use the true parameter. This package
explicitly uses **true-parameter membership**. Checking whether the maximizer lies
in a region defined around that maximizer would not be a useful coverage test.
One random ensemble at each of 25 truth points gives only 25 empirical 1D cases;
it does not establish precise frequentist coverage at every fixed parameter.

Both signed bias (equations (5)/(8)) and mean absolute bias (figure 3's label) are
saved, avoiding confusion between absolute mean signed bias and mean absolute
error. The new figures do not call an HLD width a standard error on a cross-case
mean bias.

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
