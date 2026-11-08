# Working Paper Metadata

## Core Record

- **Title:** Semiparametric Estimation of Optimal Reserve Prices with Generative
  Deep Learning
- **Author:** Shunyu Hao
- **Document type:** PSRM working paper
- **Working revision:** 0.1.5 (fixed-bandwidth theory, numerical safeguards,
  full confidence sets, and fresh-seed validation)
- **Date:** November 8, 2026
- **Archived baseline:** 0.1.4, October 15, 2026. `CITATION.cff` identifies that
  archived release; the separately frozen MA thesis remains unchanged.
- **Repository:** https://github.com/Shunyu1001/WGAN-DML-Auctions
- **JEL classifications:** C14; C15; C45; D44

## Abstract

This paper studies inference for optimal reserve prices when only auction
maxima are observed. Under symmetric independent private values, maxima
identify the latent value distribution, which is estimated globally by a
structural Wasserstein generative adversarial network with gradient penalty
(WGAN-GP). I derive a cross-fitted Neyman-orthogonal score for a fixed-bandwidth
regularized reserve. Primitive conditions cover training-fold empirical kernel
moments and a fixed-pseudo-count calibration, with an explicit single-crossing
target and failure rule. These estimators are first-order equivalent to a
same-kernel full-sample plug-in. Inference can use either a Wald interval or the
full set obtained by inverting the studentized score; a conservative interval
envelope is reported separately and does not replace that set. Fresh validation
with 100 repetitions at each of two sample sizes finds a modest small-sample
coverage gain from score inversion, but does not establish a reliable advantage
over Wald inference or an incremental gain from the GAN. Two small-sample
replications produce disconnected, boundary-touching sets because sparse
lower-tail information makes score variance highly sensitive to individual
observations. All components and diagnostic flags are retained. Earlier
experiments illustrate how good global WGAN fit can coexist with inaccurate
local moments; they remain distinct from the fresh validation. Inference for
the exact unsmoothed reserve requires additional bandwidth and bias conditions
and remains an extension.

## Manuscript and Exploratory Results

The main manuscript remains `paper/main.tex`, with fixed-pseudo-count
calibration of GAN and training-fold empirical kernel moments and three-fold
orthogonal estimation. Its workflow figure is embedded in
`paper/sections/inference.tex`.

The October 7 two-threshold experiment is archived separately in
`experiments/two_threshold/`. It is not incorporated into the manuscript's
tables, main method, or formal inference claim. The current revision incorporates
the completed fresh-seed validation and the minimum confidence-set reporting
rules without replacing the main estimator. The score-inversion rule was fixed
before fresh-seed validation. The conservative envelope and diagnostic flags
were added after inspecting the disconnected cases, not as a separately
preregistered performance improvement.

The current PDF is
`output/pdf/WGAN-DML-Auctions-PSRM-Score-Sets-2026-11-08.pdf`.
The report and reproduction paths are indexed in
`docs/psrm_confidence_sets_2026-11-08.md`.

## Keywords

optimal auctions; reserve prices; auction maxima; order statistics; double
machine learning; cross-fitting; Neyman orthogonality; Wasserstein generative
adversarial networks; semiparametric inference; Monte Carlo simulation

## One-Sentence Contribution

Global generative fit can recover a coherent latent auction distribution, but
valid inference on the reserve requires a cross-fitted local nuisance learner
and enough identifying probability mass near the reserve.

## Author Fields to Confirm

| Field | Current value |
|---|---|
| Display name | Shunyu Hao |
| Affiliation | Confirm before public release |
| Department | Confirm before public release |
| Contact email | Confirm before public release |
| ORCID | Optional; not supplied |
| Coauthors | None recorded; confirm |
