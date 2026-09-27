# Working Paper Roadmap

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

## Completed Working-Paper Elements

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
- A concise abstract, keywords, JEL codes,
  data-and-code statement, and a double-blind manuscript entry point.
- A two-tier replication package with deterministic smoke checks, CI, an exact
  environment record, a seed registry, and checksums for archived result files.
- A concise adviser memo, decision matrix, reading order, and three bounded
  questions separating contribution, claim calibration, and auction assumptions.
- Identified and anonymous manuscript builds, generic working-paper metadata,
  ethics declarations, PDF metadata, and a journal-neutral release checklist.

## Next Three Sprints

1. **Editorial freeze:** complete the journal-neutral release checklist, record
   the author affiliation/contact line, and freeze identified and anonymous PDFs
   from the same commit.
2. **External circulation:** send the prepared adviser packet and working paper,
   collect comments against the bounded contribution and assumption questions,
   and log decisions without reopening general estimator tuning.
3. **Optional empirical bridge:** if a suitable auction dataset becomes
   available, document entry, observed order statistics, and bidder-count
   variation before implementation. Treat this as a later version, not a
   prerequisite for releasing the current methodological paper.

The external-validity, theory-audit, repository-level replication-audit, and
adviser-packaging sprints are complete. The
expensive Monte Carlo archive has not been rerun in this audit; it is documented
separately from the fast checks. Freezing a traceable working-paper release and
obtaining bounded external feedback are the next critical steps; additional
tuning of the same confidence interval has lower priority.
