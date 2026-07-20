# Working Paper Release Plan

## Objective

The current objective is a self-contained, credible, and reproducible working
paper that can circulate publicly and can later be adapted to an outlet without
changing its scientific claims. No journal is selected at this stage.

## Release Standard

A release is ready when all of the following hold:

1. the title, abstract, introduction, theorem statements, simulations, and
   conclusion describe the same formal and diagnostic claim boundary;
2. identified and anonymous PDFs compile from the same source and contain
   appropriate metadata;
3. every committed figure and table is covered by the artifact manifest and the
   fast replication audit passes;
4. the repository gives a stable reading order, computational instructions,
   citation metadata, and a transparent data/code statement;
5. no placeholder application, unverified empirical claim, journal name, cover
   letter, or reviewer suggestion is required to understand or circulate the
   paper.

## Current Contribution

The working paper combines four elements:

- identification of latent bidder values from auction maxima under symmetric
  independent private values and known bidder counts;
- a bidder-level structural WGAN-GP for global latent-distribution learning;
- cross-fitted orthogonal inference for a fixed-bandwidth regularized reserve,
  with primitive nuisance-rate conditions for the empirical-local learner; and
- Monte Carlo evidence locating the boundary between global generative fit,
  local identifying information, and valid reserve-price inference.

The fixed-bandwidth theorem is the formal inferential contribution. The
shrinking-bandwidth Richardson, residual-bias, and root-union procedures are
finite-sample exact-reserve diagnostics. This separation must remain explicit
in every circulated version.

## Release Assets

- identified manuscript: `paper/main.pdf`;
- anonymous manuscript: `paper/main_anonymous.pdf`;
- adviser memo and bounded review questions;
- fast and full replication paths in `replication/README.md`;
- version and citation metadata in `CITATION.cff` and
  `docs/working_paper_metadata.md`;
- release gate in `docs/working_paper_checklist.md`.

## Upgrade Path After Release

External comments should be classified as contribution, theory, computation,
exposition, or maintained-assumption issues. Only a named issue should reopen a
corresponding part of the project. An empirical auction application can become
a later version if suitable maxima, bidder-count, reserve, and auction-format
data are available; it is not a placeholder in the present paper and is not a
condition for releasing the methodological working paper.
