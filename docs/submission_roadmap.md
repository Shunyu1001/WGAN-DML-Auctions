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

1. **Confirm the simple calibration:** expand the paired ten-repetition pilot
   before making precise coverage claims. Retain the empirical-only and
   same-kernel plug-in comparators and the fixed pseudo-count, and report
   Monte Carlo uncertainty as well as interval length.
2. **Bounded robustness:** check the same calibration across the existing
   heavy-tail and mixture designs, without adding network or score variants.
   Assess whether any incremental benefit over empirical-only estimation
   survives; do not infer that benefit from improvement over raw WGAN-DML.
3. **Editorial freeze:** complete the journal-neutral release checklist,
   confirm the author affiliation/contact line, and freeze identified and
   anonymous PDFs from the same commit. An empirical application remains an
   optional later extension requiring a suitable auction dataset.

The earlier external-validity and theory-audit results are preserved. The
September follow-up adds a deliberately small, paired calibration experiment
with independent evaluation seeds; it does not replace the original Monte
Carlo archive or certify exact-reserve inference.
