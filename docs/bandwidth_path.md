# Shrinking-Bandwidth Path

## Purpose

The empirical-local score provides reliable inference for the fixed-bandwidth
target `r_h`, but `r_h` is not the exact Myerson reserve `r_0`. This experiment
asks whether a simple power rule

`h_n = 0.15 * (n / 500)^(-alpha)`

can remove regularization bias while retaining stable score inference.

## Designs

The exponents are `alpha = 0, 0.2, 0.35, 0.5`. Each design uses 200 common-seed
replications at 500, 2,000, and 10,000 auctions. The score nuisance is estimated
directly from maximum bids in the training folds.

Under a smooth fixed-density heuristic, the regularization bias is order
`h_n^2` and the stochastic scale is order `(n h_n)^(-1/2)`. Undersmoothing for
the exact target therefore suggests `n h_n^5 -> 0`, or an exponent larger than
one fifth, while `n h_n -> infinity` prevents the local variance from exploding.

## Results

- Fixed `h=0.15` retains approximately 95 percent coverage of `r_h` but exact
  target coverage falls to 1 percent at `n=2,000` and zero at `n=10,000`.
- `alpha=0.2` has the lowest exact-target RMSE at `n=10,000` (0.0327), but exact
  coverage is only 60 percent.
- `alpha=0.35` raises exact-target coverage to 82 percent at both larger sample
  sizes. At `n=10,000`, 10.5 percent of samples have multiple score roots.
- `alpha=0.5` shrinks too aggressively: at `n=10,000`, the multiple-root rate is
  69 percent and exact-target coverage falls to 73 percent.

## Implication

No tested power rule simultaneously provides low RMSE, nominal exact-target
coverage, and stable root selection in the current sample range. Fixed-
bandwidth inference for `r_h` is the defensible baseline. Inference for `r_0`
requires explicit bias handling, higher-order correction, or confidence sets
that account for the regularization path and root uncertainty.
