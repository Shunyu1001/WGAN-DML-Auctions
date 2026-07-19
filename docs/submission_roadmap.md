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
- A submission-positioning memo, concise abstract, keywords, JEL codes,
  data-and-code statement, and a double-blind manuscript entry point.
- A two-tier replication package with deterministic smoke checks, CI, an exact
  environment record, a seed registry, and checksums for archived result files.
- A concise adviser memo, decision matrix, reading order, and three bounded
  questions separating contribution, journal choice, and auction assumptions.
- A current-form IJCEE submission pack with an anonymous review file, cover
  letter, metadata sheet, ethics declarations, and a requirements-based final
  checklist.

## Next Three Sprints

1. **Author and adviser confirmation:** send the prepared review packet, confirm
   exclusivity and author details, and either approve the current-form IJCEE
   route or choose the empirical bridge. Do not reopen estimator tuning before
   this decision.
2. **Empirical bridge:** if a suitable auction dataset becomes available,
   document entry, observed order statistics, and bidder-count variation before
   implementing the estimator.
3. **Submission pass:** incorporate adviser comments, apply the chosen journal
   template and length limit, then freeze the anonymous manuscript and
   replication archive. Do not weaken the current identification claim to force
   an application.

The external-validity, theory-audit, first submission-packaging, and repository-
level replication-audit, adviser-packaging, and current-form IJCEE packaging
sprints are complete. The
expensive Monte Carlo archive has not been rerun in this audit; it is documented
separately from the fast checks. Obtaining the adviser decision is the next
submission-critical step, and additional tuning of the same confidence interval
has lower priority.
