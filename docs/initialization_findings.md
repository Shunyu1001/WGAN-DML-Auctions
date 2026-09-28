# Findings: initialization versus adversarial training

The protocol was frozen at commit 8be02ae before computing the new arms.
This is a retrospective paired mechanism study on the prior 40 datasets,
not another independent validation. No loss or estimator setting changed.

Exact-reserve RMSE with each learner selecting its own weights:

| Design | Auctions | Initialization only | Adversarially trained | Smoothed plug-in |
| --- | ---: | ---: | ---: | ---: |
| Lognormal | 500 | 0.0527 | 0.0525 | 0.0530 |
| Lognormal | 2,000 | 0.0314 | 0.0318 | 0.0349 |
| Weibull | 500 | 0.0486 | 0.0512 | 0.0503 |
| Weibull | 2,000 | 0.0366 | 0.0393 | 0.0328 |

Initialization almost reproduces the lognormal adaptive results. It imposes
a lognormal-family shape, so this is not evidence of a generic nonparametric
advantage. With fixed calibration all absolute training-induced RMSE changes
are below 0.0004. All twelve displayed training-minus-initialization MSE
differences have approximate mean-plus/minus-1.96-MCSE intervals containing
zero; ten paired replications do not settle superiority.

The matched-weight control is informative under Weibull at n=2,000. Keeping
the trained learner's weights, training reduces RMSE from 0.0585 to 0.0393
(paired MSE difference -0.001880, MCSE 0.001084). But letting initialization
select its own weights gives 0.0366. The mean generated-component weight
across folds and distinct bandwidths rises from 0.139 to 0.400 after training.
The matched arm is a diagnostic, not a cheap standalone estimator, because
its weights still require the trained learner.

Mean outer-holdout maximum-distribution W1 improves under Weibull, from 0.0955
to 0.0780 at n=500 and from 0.0657 to 0.0483 at n=2,000. Under lognormal,
the corresponding changes are 0.0750 to 0.0757 and 0.04021 to 0.04018.
These are descriptive means over 30 overlapping-fold fits per cell; folds
must not be treated as independent Monte Carlo replications. W1 is a reporting
diagnostic, not a new selection criterion. Better global generalization is
not equivalent to better reserve estimation under the existing mixing rule.

The archive includes 560 estimates, 300 weight selections, 240 authenticated
cached fits, and 32 recomputation checks of prior trained estimates/SEs.
The largest recomputation discrepancy is 8.33e-17; 15 of 120 outer fits select
step zero. All seeds and all outcomes are retained, with no root failures in
this ablation. Original results are not overwritten.

## Implication for the next experiment

Keep fixed calibration as the main safeguarded procedure. A narrowly scoped
next experiment could test conservative caps on learned generated-component
weights, with the cap and candidate grid fixed before fresh evaluation.
This study has not run or validated that modification. Do not claim a new
WGAN-specific advantage from these results; independently check whether
any apparent improvement survives a larger Monte Carlo study.
