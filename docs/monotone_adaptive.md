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
positive-value regularity conditions. Development uses seed 20260929. Final
evaluation uses base seed 90260930 for lognormal and 20260930 for Weibull
(the latter also has the fixed 9,000,000 DGP offset). No test-seed outcome changes the architecture,
weight candidates, bandwidth, or reported cells. This is a pilot, not evidence
of uniform dominance or precise coverage. Every run is retained.

### Seed-overlap correction

The initially scheduled lognormal base seed 20260930 overlapped eight of ten
data seeds per sample size with the earlier calibration experiment, whose
base seed was 20260928: both scripts add the repetition index directly. This
was identified during the audit, before final reporting. The entire affected
lognormal cohort, including its two non-overlapping repetitions, was replaced
by a fresh prespecified cohort with base seed 90260930. The estimator source
and all tuning settings stayed frozen at commit d5c6a5a. The already completed
Weibull cohort did not overlap and was retained. Original lognormal results
are archived separately as `monotone_overlap_raw.csv`; they are not pooled
with the new evaluation. A regression test checks seed disjointness. This
replacement is due to provenance, not performance; no fresh seed is selected
or removed based on an outcome.
As a provenance cross-check, all sixteen overlapping original-fixed corrected
estimates exactly match their earlier calibration-study counterparts. The
final cohort's seed set is disjoint from both the earlier study and this
superseded cohort.

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
The fixed knot count and linear normal-quantile tails remain approximation
restrictions; unbounded support is not a guarantee of accurate tail estimation.

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

## Final pilot results

Exact-reserve RMSE in the independent evaluation is:

| Design | Auctions | Original fixed | Monotone adaptive | Smoothed plug-in |
| --- | ---: | ---: | ---: | ---: |
| Lognormal | 500 | 0.0593 | 0.0525 | 0.0530 |
| Lognormal | 2,000 | 0.0335 | 0.0318 | 0.0349 |
| Weibull | 500 | 0.0552 | 0.0512 | 0.0503 |
| Weibull | 2,000 | 0.0327 | 0.0393 | 0.0328 |

Each cell has ten paired repetitions. Against the plug-in, the combined
variant's paired MSE differences (Monte Carlo standard errors) are -0.000052
(0.000214), -0.000207 (0.000236), +0.000094 (0.000193), and +0.000469
(0.000292), respectively. The lognormal point improvements do not establish
a stable ranking; the larger Weibull cell is materially worse. Original
generator plus adaptive mixing is worse than original fixed calibration in
all four cells. Monotone generation with fixed mixing closely follows the
empirical benchmark. The fixed-calibration procedure remains the main
safeguarded estimator; adaptive mixing remains experimental.

The correctly specified lognormal MLE has RMSE 0.0151 and 0.0073 in the
lognormal cells, but about 0.097 and zero actual-target coverage under Weibull
misspecification. This is not a WGAN-specific advantage: the empirical
benchmarks also avoid that parametric misspecification. Ten repetitions do
not support precise coverage claims or a broad superiority claim.

The primary archive contains 520 estimates, 600 weight selections, and 480
generator fits. The superseded overlap archive is separate and contains 260
estimates. Automated checks verify seeds, paired completeness, finite results,
summary arithmetic, validation argmin choices, and training diagnostics.

## Reproduction and audit

```bash
python3 code/test_monotone_adaptive.py
python3 code/run_monotone_release.py
```

The runner writes raw estimates, candidate validation risks and selected
weights, per-fit training diagnostics, paired MSE differences with Monte Carlo
standard errors, configuration, a table, and a figure into
output/tables/monotone_release. Training caches are keyed by architecture,
configuration, data, seed, and source hashes. No original artifact is replaced.

Independent design cells may run in parallel with `monotone_adaptive_pilot.py`
using --distributions, --sizes, and the correct distribution-specific --seed;
combine them with `run_monotone_release.py --combine-dirs`. The release merger
requires the predeclared distribution/seed mapping, identical estimator
settings and source fingerprints, and no duplicate cells. Each archived row
records its actual evaluation base seed. Plot axes and table labels
distinguish exact-reserve results from fixed-bandwidth results. No method or
seed is omitted because of poor performance.
