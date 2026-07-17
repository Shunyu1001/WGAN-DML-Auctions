# Fixed-Bandwidth Theory Audit

## What Is Now Proved

For a fixed smoothing bandwidth and a fixed number of folds, the training-fold
empirical estimates of the local maximum-bid CDF and density moments converge
uniformly at `O_p(sqrt(log(n) / n))` on a compact reserve neighborhood. The
argument uses bounded, Lipschitz Gaussian kernel classes and training folds of
order `n`. A local CDF-overlap condition keeps the powers of the CDF in the
orthogonal score away from their singular boundary.

This rate is `o_p(n^(-1/4))`. Together with root separation, positive score
variance, and numerical consistency, it verifies the fixed-bandwidth DML
theorem for the empirical-local nuisance learner. The appendix also makes the
cross-fit expansion explicit: the held-out empirical-process remainder is
`O_p(rho_n / sqrt(n))`, while Neyman orthogonality reduces the population
nuisance remainder to `O_p(rho_n^2)`.

## What Remains High-Level

- The WGAN nuisance must achieve the same local CDF-and-density rate. A small
  global Wasserstein loss does not imply this, especially near the reserve.
- Exact-reserve inference with a shrinking bandwidth requires a triangular-
  array treatment of smoothing bias, variance, and root continuation.
- The selected empirical root must be locally separated and the score variance
  must remain positive. These conditions can fail when bidder competition
  leaves almost no observed maxima near the reserve.
- Formal numerical equivalence requires root tolerance `o(n^(-1/2))` and a
  centered finite-difference step tending to zero slowly enough. The fixed
  grid and derivative step in the Monte Carlo are finite-sample
  approximations.

## Submission Boundary

The strongest theorem is fixed-bandwidth inference for a regularized reserve
using a nuisance learner that satisfies the stated local rate. The primitive
verification covers the empirical-local learner, not the WGAN learner. The
WGAN is therefore best framed as a global structural and counterfactual model,
with local empirical moments supplying the inferential score. Richardson and
residual-bias-envelope intervals for the exact reserve remain transparent
finite-sample prototypes rather than uniformly valid confidence procedures.
