# External-Validity Monte Carlo

## Purpose

This experiment tests whether the paper's fixed-bandwidth empirical-local DML
claim survives changes in competition and valuation shape. It deliberately
does not introduce another loss function, bandwidth rule, or interval variant.

## Design

- Bidders: 3, 5, and 10.
- Auctions: 2,000 and 10,000.
- Valuations: baseline lognormal, heavy-tail lognormal, and a two-component
  lognormal mixture.
- Inference: three-fold empirical-local DML with Gaussian bandwidth 0.15.
- Replications: 200 per cell, for 3,600 estimates in total.
- Benchmark: direct inversion of the empirical maximum-bid distribution.

The primary coverage statement concerns the fixed-bandwidth population root
`r_h`. Error relative to the global monopoly-price maximizer `r_0` is reported
separately so that regularization bias is visible.

## Main finding

With three or five bidders, coverage of `r_h` lies between 92.0 and 96.5
percent across all distributions and sample sizes. With ten bidders and 2,000
auctions, coverage falls to roughly 78 percent and 9.5--13 percent of samples
have multiple roots. These cells contain essentially no observed maxima below
the reserve. At 10,000 auctions, multiple roots disappear and coverage rises.

The useful interpretation is a local-information boundary: cross-fitting and
orthogonality control nuisance-estimation effects, but they cannot recover a
reserve located in a region that the observed order statistic almost never
visits. Increasing competition makes this lower-tail problem more severe.

## Reproduction

Run `python3 code/external_validity_monte_carlo.py --reps 200`. The script
writes raw replications, a summary CSV, its configuration, and the LaTeX table
to `paper/tables/`.
