# Monotone generation and adaptive calibration

## Scope and fixed evaluation design

This experiment tests two small changes, separately and together. It retains
the original WGAN-GP critic, gradient penalty, 1,200 generator steps, five critic
steps per generator step, the orthogonal reserve score, three outer folds, and
the existing bandwidth path. It does not add covariates, an additional loss, or
a new identifying assumption.

The four ablations are original generator with fixed mixing, monotone generator
with fixed mixing, original generator with adaptive mixing, and monotone
generator with adaptive mixing. Empirical-local DML and full-sample smoothed
plug-in remain comparators. A lognormal maximum-order-statistic MLE is added as
a strong, correctly specified reference in the lognormal design and an openly
misspecified parametric reference in the Weibull design.

Evaluation is fixed at ten repetitions for each combination of n=500 or 2,000
and LN(0,0.5^2) or Weibull(shape=1.5, scale=1). Both designs satisfy the baseline
positive-value regularity conditions. Development uses seed 20260929; final
evaluation uses seed 20260930. No test-seed outcome changes the architecture,
weight candidates, bandwidth, or reported cells. This is a pilot, not evidence
of uniform dominance or precise coverage. Every run is retained.

## Monotone generator

Represent log valuation as a continuous piecewise-linear function of
z=Phi^{-1}(u), with eleven fixed knots at
(-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3). Ten positive slopes and one intercept are
learned: eleven generator parameters in total. The boundary slopes extend
linearly beyond the outer knots, so this architecture has no finite upper
valuation cap. It differs from the original tanh generator, whose log-value
support is centered on the training log-maxima mean with half-width four times
their standard deviation. Changing architecture and removing this cap are a
joint intervention; their separate causal contributions are not identified here.

Initialization regresses empirical log-maximum quantiles on Phi^{-1}(p^{1/N})
at 39 probabilities from 0.05 to 0.95. It uses fitting data only. This is a
linear log-quantile initialization that can be particularly favorable under
lognormality; the Weibull design and MLE comparator make that limitation visible.
Initialization itself is an eligible training checkpoint. Diagnostics retain
initial and final training W1 and the chosen step, including step zero.

For a monotone Q, max_i Q(U_i) has the same law as Q(U^{1/N}). This identity
generates maxima directly. Kernel moments are evaluated on 32,768 deterministic
midpoint quantiles of the maximum distribution, rather than 50,000 random
generated maxima used by the original network. Tests check the order-statistic
identity and numerical convergence on a known distribution. This is numerical
integration, not additional observed information.

The monotone generator learning rate is 0.0005; the critic learning rate remains
0.0001. The original network retains its original 0.00002 generator rate.
Both use the original adversarial loss, GP coefficient 0.1, and training-only
W1 checkpoint selection. These settings are frozen before evaluation.

## Nested validation and inference

Within each outer training fold, hold out 25% for internal validation. Fit each
architecture to the other 75%. At each bandwidth, choose a single shared
CDF/density mixing weight from (0,0.25,0.5,1). Evaluate squared prediction error
against the observed Gaussian-smoothed CDF and density signals at 21 prices
between 0.6 and 1.4 times the fitting-sample direct-inversion reserve. Divide
each component by its fitting-sample signal variance, floored at 0.01. Thus
neither grid location nor variance scaling uses validation or outer holdout
outcomes. Exact risk ties favor the smaller WGAN weight.

Refit each generator on the full outer training fold after weight selection,
then evaluate the existing orthogonal score on its untouched outer holdout.
The same learned weight is used for CDF and density at every trial price for a
given fold and bandwidth. Fixed-weight ablations retain 50/(n_training+50).
All versions share observed samples, outer folds, and the appropriate trained
generators. The direct-inversion root reference and numerical root grid are
identical across all nonparametric comparators, inherited from the prior pilot.

Fixed-bandwidth results target r_h at h=0.15. Exact-reserve results use the
existing covariance-aware Richardson combination, with
h=0.15(n/500)^(-0.2) and sqrt(2)h. Cross-bandwidth covariance is retained by
combining same-observation influence functions. These intervals are diagnostic.
The adaptive estimator does not inherit the fixed O(1/n) mixture argument;
local nuisance-rate and root conditions must still be verified. Nested
validation alone is not a proof of valid inference.

MLE maximizes log N + (N-1) log Phi(z) + log phi(z) - log sigma - log B,
where z=(log B-mu)/sigma. Its interval uses a sandwich parameter covariance and
the reserve's delta-method gradient. Under Weibull misspecification this
targets a pseudo-true lognormal reserve, not generally the actual reserve;
reported actual-target coverage is a diagnostic, not a coverage guarantee.

## Reproduction and audit

```bash
python3 code/test_monotone_adaptive.py
python3 code/monotone_adaptive_pilot.py
```

The runner writes raw estimates, candidate validation risks and selected
weights, per-fit training diagnostics, paired MSE differences with Monte Carlo
standard errors, configuration, a table, and a figure into
output/tables/monotone_adaptive. Training caches are keyed by architecture,
configuration, data, seed, and source hashes. No original artifact is replaced.

Independent design cells may run in parallel using --distributions and --sizes;
combine them with --combine-dirs. The merger rejects conflicting configurations,
different source fingerprints, and duplicate cells. Plot axes and table labels
distinguish exact-reserve results from fixed-bandwidth results. No method or
seed is omitted because of poor performance.
