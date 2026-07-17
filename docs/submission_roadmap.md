# Submission Roadmap

## Current Paper Claim

The defensible contribution is a methodological and Monte Carlo paper:

1. maximum-bid order statistics identify the latent valuation distribution
   under symmetric independent private values;
2. a structural WGAN supplies a global latent-distribution learner;
3. a cross-fitted orthogonal score provides formal inference for a fixed,
   regularized reserve when the local nuisance rate is strong enough;
4. simulations show why global WGAN fit does not guarantee that local rate and
   motivate a hybrid global-WGAN/local-score estimator;
5. Richardson and three-bandwidth intervals are exact-reserve prototypes, with
   their finite-sample limitations reported explicitly.

## Completed Submission Elements

- Reproducible direct, WGAN, cross-fit, orthogonal-score, bandwidth, and bias
  correction experiments.
- Formal fixed-bandwidth asymptotic linearity theorem and proof.
- Exact-reserve Richardson proposition with an explicit residual-bias rate.
- Clear separation between proved fixed-bandwidth inference and diagnostic
  exact-reserve inference.
- No placeholder empirical application or appendix text.
- A bounded external-validity Monte Carlo across three bidder counts and three
  valuation distributions, with the formal regularized target kept separate
  from the exact reserve.
- A primitive fixed-bandwidth nuisance-rate proposition for the empirical-local
  learner and an explicit cross-fitted remainder proof.

## Next Three Sprints

1. **Submission packaging:** choose a target field/econometrics journal, adapt
   length and framing, prepare a clean replication archive, and obtain an
   adviser-level read focused on the contribution and maintained auction
   assumptions.
2. **Replication audit:** run the documented commands from a clean environment,
   record package versions and random seeds, and separate lightweight smoke
   tests from the full Monte Carlo archive.
3. **Empirical bridge:** if a suitable auction dataset becomes available,
   document entry, observed order statistics, and bidder-count variation before
   implementing the estimator. Do not weaken the current identification claim
   to force an application.

The external-validity and bounded theory-audit sprints are complete. The project
should now move to submission packaging; additional tuning of the same
confidence interval has lower priority.
