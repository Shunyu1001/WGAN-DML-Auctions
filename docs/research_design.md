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

## Open Theoretical Tasks

- Define the regularized reserve-price target.
- Derive the pathwise derivative through the order-statistic mapping.
- Construct the influence-function or Riesz correction.
- State nuisance-rate and simulation-rate conditions.
- Separate inference for the regularized target from approximation to the exact
  Myerson reserve.

## Data Requirements

- Clear distinction between winning bid and transaction price.
- Observed bidder count or a defensible model of competition.
- Auction format and reserve-price censoring information.
- Repeated auctions with comparable items or rich observable characteristics.
