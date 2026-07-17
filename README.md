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

## Reproduce the First Monte Carlo Benchmark

Install the Python dependencies and run:

```bash
python3 -m pip install -r requirements.txt
python3 code/baseline_direct_inversion.py --reps 500
```

The script simulates maximum bids, applies the order-statistic inversion
\(\widehat F=\widehat H^{1/N}\), estimates the reserve, and writes the table and
figure used by the paper to `paper/tables/` and `paper/figures/`.

The structural WGAN-GP pilot uses the same data-generating process and can be
reproduced with:

```bash
python3 code/wgan_gp_baseline.py --reps 5 --steps 1200 --device cpu
```

The generator is bidder-level and is applied to independent noise draws before
the auction maximum is formed. This preserves the independent-private-values
restriction instead of allowing one network draw to create a correlated vector
of bidder values.

The three-fold WGAN plug-in precursor can be reproduced after the full-sample
pilot with:

```bash
python3 code/crossfit_wgan_pilot.py --reps 5 --folds 3 --steps 1200 --device cpu
```

For each fold, the script trains the WGAN only on the other folds, computes a
fold-specific plug-in reserve, and averages those reserves. Held-out auctions
are used for an out-of-fold fit diagnostic. This is the sample-splitting layer
that precedes DML, not yet the orthogonal WGAN-DML estimator.

The observation-level orthogonal-score pilot can then be reproduced with:

```bash
python3 code/orthogonal_wgan_dml.py --reps 5 --folds 3 --steps 1200 --bandwidth 0.15 --device cpu
```

The score uses Gaussian-smoothed CDF and density moments of generated maximum
bids as fold-specific nuisances. Held-out auctions enter through the associated
influence-function correction. The fixed-bandwidth target and the exact
Myerson reserve are reported separately; the current five-seed pilot is a
diagnostic implementation and does not establish valid asymptotic coverage.

To isolate whether the score or the WGAN nuisance causes the poor coverage,
run the local-nuisance ablation:

```bash
python3 code/local_nuisance_ablation.py --reps 200 --folds 3 --bandwidth 0.15
```

This compares the stored WGAN-DML pilot with (i) score nuisances estimated
directly from maximum bids in each training fold and (ii) oracle nuisances from
the known Monte Carlo DGP. The empirical-local version nearly matches the
oracle, locating the failure in the WGAN's local tail approximation rather than
in the orthogonal-score algebra.

## Current Estimation Plan

1. Establish identification from observed order statistics under a symmetric
   independent-private-values benchmark.
2. Estimate the latent valuation distribution with a structural WGAN-GP.
3. Implement a cross-fitted plug-in estimator as the computational baseline.
4. Evaluate an observation-level orthogonal score for a regularized
   reserve-price parameter.
5. Use oracle and empirical-local ablations to diagnose whether the WGAN's
   local nuisance rates are strong enough for reliable inference.

Raw or restricted data should never be committed to the repository.
