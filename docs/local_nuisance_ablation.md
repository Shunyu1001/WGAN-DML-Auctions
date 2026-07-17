# Local-Nuisance Ablation

## Question

Does the WGAN-DML pilot fail because the orthogonal reserve score is wrong, or
because WGAN-generated maximum bids estimate the score's local CDF and density
moments too poorly?

## Three nuisance designs

1. **WGAN generated:** the original five-seed pilot estimates both local
   moments from maximum bids simulated by a training-fold WGAN.
2. **Training-fold empirical:** each held-out fold uses local moments calculated
   directly from the observed maximum bids outside that fold.
3. **Oracle:** the score uses population local moments from the known Monte
   Carlo DGP.

All methods use the same Gaussian bandwidth `h=0.15` and target
`r_h=0.8318`. The empirical and oracle designs use 200 repetitions at each
sample size.

## Results

| Auctions | WGAN coverage | Empirical coverage | Oracle coverage |
|---:|---:|---:|---:|
| 500 | 0.400 | 0.900 | 0.940 |
| 2,000 | 0.600 | 0.945 | 0.960 |
| 10,000 | 0.000 | 0.950 | 0.955 |

The empirical-local RMSE relative to the regularized target is 0.0350, 0.0171,
and 0.0079. The corresponding oracle RMSE is 0.0311, 0.0164, and 0.0078. The
near-oracle performance validates the score implementation and isolates the
first pilot's failure in the WGAN local nuisance approximation.

At the exact reserve, the sample contains on average only 1.05, 5.04, and 25.69
maximum bids at or below the reserve. This explains why a global distributional
criterion can look accurate while missing the economically decisive local tail.

## Implication

Under the baseline without covariates, the two score nuisances are directly
observable moments of maximum bids; forcing them through the WGAN is
unnecessary. A defensible hybrid uses WGAN-GP for the full latent distribution
and structural counterfactuals, while estimating reserve-local score moments
directly on training folds. With covariates or richer auction heterogeneity,
these observable local moments can be estimated by a dedicated local learner.
