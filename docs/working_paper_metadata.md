# Working Paper Metadata

## Core Record

- **Title:** Semiparametric Inference for Myerson Reserve Prices from Auction
  Maxima
- **Author:** Shunyu Hao
- **Document type:** Working paper
- **Version:** 0.1.0 candidate
- **Date:** July 19, 2026
- **Repository:** https://github.com/Shunyu1001/WGAN-DML-Auctions
- **JEL classifications:** C14; C15; C45; D44

## Abstract

This paper studies inference for Myerson reserve prices when only auction
maxima are observed. Under symmetric independent private values, maxima
identify the latent value distribution, which is estimated globally by a
structural Wasserstein generative adversarial network with gradient penalty
(WGAN-GP). I derive a cross-fitted Neyman-orthogonal score for a fixed-bandwidth
regularized reserve. Primitive conditions give training-fold empirical kernel
moments the required o_p(n^-1/4) nuisance rate. Monte Carlo experiments show
that good global WGAN fit can coexist with inaccurate local tail moments and
invalid inference; an empirical-local nuisance restores near-nominal coverage.
Across valuation distributions, coverage remains stable with three or five
bidders but deteriorates with ten bidders when maxima rarely contain local
identifying information. Bias-corrected shrinking-bandwidth procedures improve
exact-reserve inference but remain diagnostic. The results support a hybrid
workflow: use the WGAN for global structural counterfactuals and local empirical
moments for reserve inference.

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
