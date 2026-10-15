# Two Threshold Reserve Price Experiment

This archive contains the October 7, 2026 exploratory experiment associated
with *Semiparametric Estimation of Optimal Reserve Prices with Generative Deep
Learning*. It is separate from the manuscript's main fixed-calibration method.
No results from this experiment replace files in `paper/tables/` or
`paper/figures/`.

## Estimation Target

Let `H` be the distribution of the observed maximum submitted bid and `N` the
known number of bidders. For a candidate reserve `r` and fixed half-window
`delta`, the nuisance values are `p_minus = H(r - delta)` and
`p_plus = H(r + delta)`. The score corrects the difference between
`q(r + delta)` and `q(r - delta)`, where `q(r) = r[1 - H(r)^(1/N)]`.

The target maximizes a uniform-window average of this single-buyer-equivalent
revenue objective. It is not exactly the continuous Myerson reserve, and `q`
is not the actual revenue of the `N`-bidder auction. Under the maintained
regular symmetric IPV model, the unaveraged objective and actual auction
revenue share the same continuous optimal reserve. Reported revenue regret
uses actual expected second-price auction revenue.

This procedure avoids kernel density estimation but retains the price-window
regularization. Its two probabilities are evaluations of one CDF, not two
independent machine-learning models or a doubly robust pair.

## Archived Runs

All runs use lognormal individual valuations with `mu=0`, `sigma=0.5`, three
folds, sample sizes `500`, `2000`, and `10000`, and the four prespecified
half-windows `0.05`, `0.10`, `0.15`, and `0.20`.

| Directory under results | Design |
|---|---|
| `two_threshold_pilot/` | Five bidders; 10 paired repetitions per sample size, using existing GAN fits and empirical benchmarks |
| `two_threshold_empirical_200/` | Five bidders; 200 empirical-CDF repetitions per sample size |
| `two_threshold_n2_200/` | Two bidders; 200 empirical-CDF repetitions per sample size |

The archived files retain raw estimates, summaries, configurations, population
targets, paired comparisons where applicable, plots, and the original experiment
notes. Development smoke runs, synthetic-sample caches, model weights, and local
PDF exports are excluded. Historical configuration records are preserved rather
than relabeled as newly run experiments.

## Findings

At `delta=0.15`, exact-reserve RMSE in the paired five-bidder pilot is
`0.1657`, `0.1006`, and `0.0647` for calibrated two-threshold GAN estimation.
The existing calibrated Richardson estimator gives `0.0897`, `0.0472`, and
`0.0173` on the same data and folds. Ten repetitions are exploratory, not a
confirmation of a stable ranking.

In the 200-repetition five-bidder empirical runs, target-score coverage at this
window is `18.5%`, `27.0%`, and `80.0%`. The lower-threshold training-fold
probability is zero in about `95.5%`, `78.8%`, and `31.2%` of cases. With two
bidders, target-score coverage is `94.5%`, `96.0%`, and `94.5%`; these are
empirical-CDF results, not evidence of a GAN-specific benefit. All four windows
are retained, including unsuccessful cells.

The current results therefore do not support replacing the manuscript's main
procedure in the five-bidder maximum-only setting.

## Inference and Numerical Limits

Confidence sets invert pointwise score tests without estimating a density or
Jacobian. Coverage is checked directly at the population target during
evaluation; reported set components are grid approximations. They are not
guaranteed outer envelopes, and the simulations do not establish a new theorem.

The orthogonal score floors probabilities at `1e-8` to handle negative powers.
Clipping and zero-probability diagnostics are retained; the floor is not assumed
to be harmless. Raw CDF plug-in estimates are not floored and have no claimed
confidence interval. The nested indicators are evaluated on the same held-out
observation, retaining their covariance.

Roots are down-crossings nearest a data-only direct-inversion reference.
Bisection locates jumps rather than certifying exact zeros; residuals and
multiple-root diagnostics are recorded. The comparison with the original
Gaussian pipeline also differs in root-grid resolution and selection rules,
so it does not isolate only the effect of removing a kernel.

## Audit and Reproduction

From the repository root, after installing `requirements.txt`:

```bash
make experimental-check
```

This runs deterministic tests and verifies the committed archive without
training a GAN or changing archived results. The main manuscript checks remain
available through `make smoke`.

The experiment implementation remains in `code/two_threshold_pilot.py`. New
output goes to local `output/tables/` directories rather than overwriting this
archive. Empirical runs require no GAN cache:

```bash
python3 code/two_threshold_pilot.py --no-gan --reps 200 --output output/tables/two_threshold_empirical_rerun
python3 code/two_threshold_pilot.py --no-gan --bidders 2 --reps 200 --output output/tables/two_threshold_n2_rerun
```

The historical GAN pilot reused synthetic samples authenticated by training-data,
configuration, source-code, and seed hashes. Those large local caches are not
distributed here. Without matching caches, explicitly permit retraining:

```bash
python3 code/two_threshold_pilot.py --reps 10 --allow-training --output output/tables/two_threshold_gan_rerun
```

Retraining can take substantial CPU time. The archived environment and seeds
document the original runs but do not guarantee bitwise equality across software
versions or hardware. The provenance directory retains the matching historical
experiment source; subsequent archive/audit improvements do not rewrite the
historical configuration hashes.
