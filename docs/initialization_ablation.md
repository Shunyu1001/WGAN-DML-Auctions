# Initialization versus adversarial training

## Frozen protocol (before computing new ablation outcomes)

This is a retrospective mechanism ablation of the complete monotone/adaptive
pilot at commit 51b0e91, not a new independent confirmation. It uses every one
of its 40 datasets: lognormal and Weibull, n=500 and 2,000, ten repetitions
per cell. No new seed, sample, bandwidth, knot, loss, or tuning choice is made.
The earlier cohort-overlap correction remains in force.

Reconstruct the same data, outer folds, 75/25 inner split, and deterministic
32,768-point maximum-quantile integration. Replace adversarial training by
the existing data-only linear initialization, without a single optimizer
step. This initialization is a lognormal-family restriction, not a generic
nonparametric learner. Compare three additional arms:

1. Initialization fixed: use the existing 50-pseudo-observation weight.
2. Initialization adaptive: reselect weights on the same inner validation
   sample using the same observable-signal risk and candidate grid.
3. Initialization matched: use the weights selected by the trained monotone
   generator, but initialize the outer nuisance without adversarial training.
   This controlled diagnostic holds weights constant; it is not a cheap
   deployable algorithm because selecting those weights requires training.

The fixed comparison isolates the effect of training at fixed calibration.
The matched comparison isolates outer-nuisance training conditional on the
trained learner's weight-selection policy. The adaptive comparison includes
both training and the resulting change in validation-selected weights.
All procedures retain the same cross-fitted score, root rule, bandwidths,
and covariance-aware Richardson correction. Exact-target intervals remain
diagnostic, and the new arms have no additional coverage theorem.

Archived trained estimates and empirical comparators are carried over
unchanged. Cached fits are authenticated by data, seed, settings, and source
hashes. A missing cache stops the default run rather than silently retraining;
an explicit `--train-missing` option supports replication on a fresh machine.
For the first repetition of every cell, recompute the trained fixed/adaptive
estimates and standard errors and require agreement with the prior archive.

Report RMSE, bias, interval diagnostics, root failures, paired training-minus-
initialization MSE differences and their Monte Carlo standard errors. Also
compare initialized and trained maximum-distribution W1 on outer training
and holdout observations using identical integration precision. Holdout W1
is a reporting diagnostic only and never enters fitting or selection.
Step-zero checkpoints must reproduce the initialized synthetic distribution.
Retain all results, including detrimental training effects. Do not interpret
ten repetitions or this retrospective comparison as a broad superiority test.

## Reproduction

Run `python3 code/initialization_ablation.py` with the prior training cache
available, or add `--train-missing` to reproduce missing trained fits.
The output directory is `output/tables/initialization_ablation`.
