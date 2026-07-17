# Bias-Corrected Exact-Reserve Inference

## Purpose

Shrinking the score bandwidth reduces the difference between the regularized
reserve `r_h` and the exact Myerson reserve `r_0`, but the first bandwidth-path
experiment still undercovers `r_0`. This experiment tests a covariance-aware
Richardson correction rather than shrinking the bandwidth more aggressively.

## Estimator

For a common sample, fold partition, and bandwidth ratio `lambda = sqrt(2)`,
the empirical-local DML score produces estimates at `h` and `lambda * h`. If
`r_h = r_0 + b_2 h^2 + O(h^4)`, the corrected estimator is

`r_R = (lambda^2 * r_hat_h - r_hat_{lambda h}) / (lambda^2 - 1)`.

With `lambda = sqrt(2)`, this becomes `2 * r_hat_h - r_hat_{sqrt(2) h}`. The
standard error uses the same linear combination of the two observation-level
influence functions. This retains their covariance and is materially different
from treating the two estimates as independent. The large-bandwidth root
anchors root selection at the smaller bandwidth.

## Monte Carlo Results

The formal experiment uses 200 common-seed repetitions at 500, 2,000, and
10,000 auctions and compares bandwidth exponents `alpha = 0.2` and `0.35`.

- With `alpha = 0.2`, exact-reserve coverage rises from 59.0 to 92.5 percent at
  `n=2,000` and from 60.0 to 88.5 percent at `n=10,000`.
- The corresponding exact-target RMSE falls from 0.0512 to 0.0420 and from
  0.0327 to 0.0296.
- The extrapolated population-target bias at `n=10,000` is 0.0080, compared
  with 0.0252 before correction.
- With `alpha = 0.35`, correction increases variance and does not repair
  undercoverage. At `n=10,000`, 30.5 percent of samples have multiple roots at
  one of the two bandwidths.

## Implication

The correction is useful but not a complete inferential solution. The
`alpha=0.2` design is the preferred exact-target prototype because it improves
bias, RMSE, and coverage without root instability. Its remaining undercoverage
shows that higher-order bias, nonlinear root selection, or non-Gaussian score
behavior still matters. Fixed-bandwidth inference for `r_h` remains the clean
baseline; exact-target inference should be described as bias-corrected and
finite-sample diagnostic rather than fully validated.
