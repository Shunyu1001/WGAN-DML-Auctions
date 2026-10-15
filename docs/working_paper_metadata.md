# Working Paper Metadata

## Core Record

- **Title:** Semiparametric Estimation of Optimal Reserve Prices with Generative
  Deep Learning
- **Author:** Shunyu Hao
- **Document type:** Working paper
- **Version:** 0.1.4 (title synchronization and separate exploratory archive)
- **Date:** October 15, 2026
- **Repository:** https://github.com/Shunyu1001/WGAN-DML-Auctions
- **JEL classifications:** C14; C15; C45; D44

## Abstract

This paper studies inference for optimal reserve prices when only auction
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
exact-reserve inference but remain diagnostic. A fixed-pseudo-count local
calibration improves the WGAN score in a paired pilot while closely tracking
empirical-only estimation and leaving the network and loss unchanged.
A separate monotone-generator and adaptive-mixing pilot gives mixed results,
without a uniform advantage over empirical benchmarks.

## Manuscript and Exploratory Results

The main manuscript remains `paper/main.tex`, with fixed-pseudo-count
calibration of GAN and training-fold empirical kernel moments and three-fold
orthogonal estimation. Its workflow figure is embedded in
`paper/sections/inference.tex`.

The October 7 two-threshold experiment is archived separately in
`experiments/two_threshold/`. It is not incorporated into the manuscript's
tables, main method, or formal inference claim. This version synchronizes the
title and project descriptions; it does not present new main-method results.

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
