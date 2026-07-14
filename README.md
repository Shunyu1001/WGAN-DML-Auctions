# WGAN-DML Auctions

This repository contains an MA thesis project on semiparametric estimation of
Myerson reserve prices from incomplete auction data. The empirical distribution
of latent bidder values is estimated with WGAN-GP, while cross-fitting and an
orthogonal score are used to estimate and conduct inference on a low-dimensional
reserve-price parameter.

## Project Structure

- `paper/`: LaTeX paper draft for Overleaf.
- `code/`: estimation, simulation, and data-processing code.
- `data/raw/`: original auction data; ignored by Git except for `.gitkeep`.
- `data/processed/`: cleaned analysis data; ignored by Git except for `.gitkeep`.
- `docs/`: research-design notes and implementation decisions.
- `output/`: generated figures and tables; ignored by Git except for directory placeholders.

## Paper

The main LaTeX document is `paper/main.tex`. In Overleaf, select this file as
the project's Main document if it is not detected automatically.

## Current Estimation Plan

1. Establish identification from observed order statistics under a symmetric
   independent-private-values benchmark.
2. Estimate the latent valuation distribution with a structural WGAN-GP.
3. Implement a cross-fitted plug-in estimator as the computational baseline.
4. Derive an orthogonal score for a regularized reserve-price parameter.
5. Compare direct inversion, plug-in WGAN-GP, cross-fitted WGAN-GP, and the
   orthogonal WGAN-DML estimator in Monte Carlo experiments.

Raw or restricted data should never be committed to the repository.
