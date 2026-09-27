# Simple local calibration pilot

## Fixed design

This follow-up keeps the original WGAN-GP architecture, loss, 1,200 generator
steps, three folds, and 50,000 generated maxima per fold. It adds only a convex
mixture of training-fold empirical and WGAN local moments. The same weight is
used for the CDF and density, preserving a coherent smoothed distribution:

`eta_cal = (1 - w) * eta_empirical + w * eta_wgan`,
`w = 50 / (training_auctions + 50)`.

The pseudo-count 50 is fixed before the evaluation runs. There is no selection
using the true reserve, evaluation-fold observations, or reported coverage.
The WGAN weight is about 13.0%, 3.6%, and 0.74% for sample sizes 500, 2,000,
and 10,000 with three folds. This is a small safeguard, not a new architecture
or a new loss. It approaches the empirical-local estimator as sample size
grows and cannot establish a separate efficiency gain from WGAN.

## Comparisons and targets

All four methods use identical data, fold assignments, and (where applicable)
the same trained WGANs: original WGAN-DML, calibrated WGAN-DML, empirical-local
DML, and full-sample Gaussian-smoothed plug-in with delta-method inference.

The fixed-bandwidth target uses h=0.15. The exact-reserve diagnostic uses the
existing stable path h=0.15(n/500)^(-0.2) and Richardson combination
`2*r(h)-r(sqrt(2)*h)`. Standard errors combine observation-level influence
functions before taking their variance, retaining cross-bandwidth covariance.
No residual-bias envelope is added in this experiment.

Every method uses the direct-inversion estimate as its root-selection reference.
Grid brackets are refined with Brent's method to tolerance 1e-10. A central
Jacobian difference uses step 1e-4. The new WGAN-DML rows are paired comparators,
not byte-for-byte reproductions of the older five-seed table. Missing and
multiple roots are retained and reported. A no-root fallback is explicitly
marked; it is not dropped.

## Reproduction

Development used one sample at n=2,000 with seed 20260927. Evaluation uses ten
repetitions per sample size and independent base seed 20260928. All ten are
retained. This remains a pilot; its coverage estimates have substantial Monte
Carlo uncertainty. Figures display Wilson intervals instead of treating 90%
or 100% observed coverage as precise population coverage.

One alternative algebraic rescaling of the orthogonal moment was also checked
on the development sample only. Its regularized estimate was 0.8532, versus
target 0.8318; the selected calibration gave 0.8236. That alternative is not
part of the evaluation or the manuscript method. No evaluation seeds were
used to choose the pseudo-count or add further estimator variants.

```bash
python3 code/calibrated_wgan_pilot.py --reps 10 --seed 20260928
python3 code/test_calibrated_wgan.py
```

The default output directory is `output/tables/calibration_pilot`, leaving
earlier artifacts intact. Training caches are keyed by source hash,
configuration, data hash, and seed. Separate sample-size runs can be combined
with `--combine-dirs`; incompatible configurations and duplicate cells are
rejected. `calibrated_wgan_paired.csv` records paired squared-error differences
and Monte Carlo standard errors against each comparator.

## Evaluation results

Fixed-bandwidth RMSE for original WGAN-DML is 0.0713, 0.0654, and 0.0444;
calibration gives 0.0456, 0.0182, and 0.0080. Empirical-local DML gives
0.0452, 0.0182, and 0.0080, while the same-kernel plug-in gives 0.0357,
0.0162, and 0.0077. Calibration therefore repairs much of the WGAN error
without establishing an incremental gain over the empirical comparators.

The exact-reserve Richardson RMSE is 0.0897, 0.0472, and 0.0172 for the
calibrated method. Raw WGAN-DML has multiple roots in half the n=10,000
runs and one negative extrapolated reserve. All results, including that
failure, are retained. The summary reports its nonpositive-estimate rate.

## Fixed-bandwidth justification

For h fixed, smoothed CDF values are bounded by 1 and density values by
`1/(sqrt(2*pi)*h)`. Consequently the calibrated nuisance differs uniformly
from the empirical nuisance by O(1/n), even if the WGAN is misspecified.
Under the existing overlap and root assumptions it inherits the empirical
fixed-bandwidth rate. This argument does not establish validity for a shrinking
bandwidth, an exact-reserve interval, or conditional auction models.
