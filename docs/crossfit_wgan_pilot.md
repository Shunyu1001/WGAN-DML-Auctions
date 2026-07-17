# Fold-Averaged WGAN Pilot

## Purpose

This experiment implements the sample-splitting layer that precedes the
orthogonal reserve-price score. It is deliberately labeled a fold-averaged
WGAN plug-in rather than a complete DML estimator.

## Construction

For each Monte Carlo sample, auctions are randomly partitioned into three
folds. For fold `k`, a structural bidder-level WGAN-GP is trained on all
auctions outside the fold. The trained generator produces a fold-specific
latent valuation distribution and plug-in reserve. The reported estimator is
the average of the three fold-specific reserves.

The held-out maxima are used to calculate an out-of-fold Wasserstein
diagnostic. They are not yet used in an observation-level reserve score.
Consequently, this experiment measures the effects of sample splitting and
generator averaging, but it does not yet have the Neyman-orthogonality or
inference properties of DML.

## Reproducibility

- DGP: lognormal valuations with `N=5`, identical to the direct-inversion and
  full-sample WGAN pilots.
- Sample sizes: 500, 2,000, and 10,000 auctions.
- Monte Carlo seeds: the same five data seeds used in the full-sample WGAN
  pilot.
- Folds: 3.
- Per-fold training: 1,200 generator updates, five critic updates per generator
  update, batch size 512, and gradient-penalty weight 0.1.
- Fold-specific generated values: 50,000.
- Fold-specific diagnostic maxima: 20,000.

## Pilot result

Fold averaging reduces RMSE from 0.0687 to 0.0368 at `n=500`, from 0.0579 to
0.0270 at `n=2,000`, and from 0.0612 to 0.0274 at `n=10,000`, relative to the
full-sample WGAN runs. The improvement is mostly a reduction in seed variance.
The estimator still has negative bias of roughly 0.019 to 0.033, so sample
splitting alone does not remove the nuisance-estimation error.

Held-out maximum-bid Wasserstein distance falls from 0.0885 at `n=500` to
0.0375 at `n=10,000`. Latent-value Wasserstein distance does not improve in the
same way, reinforcing the distinction between observable fit and recovery of
the reserve-relevant latent distribution.

## Next step

Derive and implement an observation-level regularized reserve score. For each
held-out auction, evaluate that score with the nuisance estimates trained
outside its fold, then add the influence-function or Riesz correction needed
for Neyman orthogonality. Only that stage should be labeled WGAN-DML.
