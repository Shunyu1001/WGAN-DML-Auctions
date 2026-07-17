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

## Open Theoretical Tasks

- State primitive local nuisance-rate and simulation-rate conditions.
- Derive an explicit second-order remainder bound near the reserve.
- Separate inference for the regularized target from approximation to the exact
  Myerson reserve.
- Study a shrinking-bandwidth sequence and local tail-weighted nuisance learner.

## Data Requirements

- Clear distinction between winning bid and transaction price.
- Observed bidder count or a defensible model of competition.
- Auction format and reserve-price censoring information.
- Repeated auctions with comparable items or rich observable characteristics.
