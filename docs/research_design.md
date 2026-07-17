# Research Design

## Core Question

Can a structural WGAN-GP estimate the latent valuation distribution from
incomplete auction data while a cross-fitted orthogonal score provides valid
inference on a low-dimensional Myerson reserve-price parameter?

## Baseline Observation Model

- Auction-level observation: highest bid, bidder count, and auction covariates.
- Conditional symmetric independent private values.
- Known bidder count in the baseline design.
- Independent latent noise for every generated bidder.

## Estimator Sequence

1. Direct order-statistic inversion benchmark.
2. WGAN-GP plug-in estimator.
3. Cross-fitted WGAN-GP plug-in estimator.
4. Cross-fitted orthogonal WGAN-DML estimator.

## Orthogonal-Score Status

- A fixed-bandwidth target and its observation-level influence correction are
  implemented in `code/orthogonal_wgan_dml.py`.
- The first pilot rejects the working assumption that global WGAN fit is enough
  for the local nuisance rates required by orthogonal inference.
- A 200-repetition ablation shows that training-fold empirical local moments
  restore near-oracle RMSE and coverage. The preferred baseline is therefore a
  hybrid rather than a WGAN-only score nuisance.
- A shrinking-bandwidth path shows that first-order undersmoothing reduces the
  gap to the exact reserve but does not deliver uniform nominal coverage in the
  current sample range; aggressive shrinkage creates multiple roots.
- A covariance-aware Richardson correction on the `n^(-0.2)` path reduces
  exact-target bias and RMSE and raises coverage to 92.5 and 88.5 percent at
  2,000 and 10,000 auctions. It improves inference but does not fully restore
  nominal coverage.

## Open Theoretical Tasks

- State primitive local nuisance-rate and simulation-rate conditions.
- Derive an explicit second-order remainder bound near the reserve.
- Separate inference for the regularized target from approximation to the exact
  Myerson reserve.
- Bound the residual higher-order bias after Richardson correction and develop
  a root-robust confidence set for the exact reserve.
- Compare empirical-local, tail-weighted WGAN, and conditional local learners.

## Data Requirements

- Clear distinction between winning bid and transaction price.
- Observed bidder count or a defensible model of competition.
- Auction format and reserve-price censoring information.
- Repeated auctions with comparable items or rich observable characteristics.
