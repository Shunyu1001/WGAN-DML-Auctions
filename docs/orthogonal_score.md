# Observation-Level Orthogonal Reserve Score

## Regularized target

Let `B` be the observed maximum bid from an auction with `N` bidders. For a
fixed Gaussian bandwidth `h`, define

- `S_h(B; theta) = Phi((theta - B) / h)`,
- `K_h(B; theta) = phi((theta - B) / h) / h`,
- `A_h(theta) = E[S_h(B; theta)]`, and
- `G_h(theta) = E[K_h(B; theta)]`.

The regularized reserve moment is

`m_h(theta) = 1 - A_h(theta)^(1/N)
              - (theta/N) A_h(theta)^(1/N-1) G_h(theta)`.

Its zero is `theta_(0,h)`. This is distinct from the exact Myerson reserve
when `h` is fixed.

## Orthogonalization

For nuisance vector `eta=(A,G)`, the observation-level score is

`psi_h = m_h(theta,A,G)
         + m_A(theta,A,G) [S_h(B;theta)-A]
         + m_G(theta,A,G) [K_h(B;theta)-G]`.

At the truth, the residuals have mean zero and the derivative of the score
expectation with respect to `(A,G)` is zero. The implementation estimates
`(A,G)` from maximum bids simulated by a WGAN trained outside the held-out
fold. Each held-out auction then enters the two residual corrections.

## Pilot convention

- Three folds and the same WGAN architecture as the plug-in precursor.
- Fixed bandwidth `h=0.15`.
- 50,000 generated values and 50,000 generated maxima per fold.
- The score root nearest the fold-averaged plug-in is selected on a fixed
  reserve grid.
- Standard errors use the cross-fitted score variance and a numerical score
  Jacobian.

## What the pilot shows

The score is Neyman-orthogonal with respect to the two regularized local
moments, but orthogonality only removes their first-order estimation effect.
In the current design, the reserve lies in the extreme lower tail of the
maximum-bid distribution. The WGAN local nuisance error is too large for the
second-order remainder to be negligible. The correction is positive and large,
and nominal confidence intervals undercover the fixed-bandwidth target.

This is a substantive diagnostic, not a reason to relabel fold averaging as
DML. The next estimator should improve local tail nuisance learning (or exploit
additional order statistics) before claiming valid inference.

The follow-up oracle and empirical-local ablation confirms this diagnosis. When
the same score uses local moments estimated directly from training-fold maximum
bids, its RMSE and coverage nearly match an oracle supplied with population
moments. See `docs/local_nuisance_ablation.md`.
