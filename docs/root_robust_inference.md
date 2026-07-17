# Three-Bandwidth and Root-Robust Inference

## Purpose

The two-bandwidth Richardson estimator removes the leading `h^2` smoothing
bias, but its nominal interval still undercovers the exact reserve. This
experiment distinguishes remaining higher-order bias from empirical root
selection.

## Construction

For `lambda = sqrt(2)`, estimates are computed at `h`, `lambda*h`, and
`lambda^2*h`. Two adjacent Richardson estimates are

```text
R(h)        = 2 theta(h)        - theta(lambda h)
R(lambda h) = 2 theta(lambda h) - theta(lambda^2 h).
```

Under an even-order local expansion, the residual bias of `R(h)` is estimated
from `D = R(lambda h) - R(h)`. The reported upper diagnostic is

```text
(abs(D) + 1.96 * se(D)) / (lambda^4 - 1).
```

All standard errors use the joint observation-level influence values, so
covariance across bandwidths is retained.

Every score-grid root at `h` is linked to its nearest root at `lambda*h` and
then `lambda^2*h`. An interval is built for each chain, and the root-robust set
is their union. Disconnected gaps are retained when reporting set length.

## Main Result

- On the stable `n^(-0.2)` path, the bias envelope gives 96.5 percent coverage
  of the exact reserve at 500, 2,000, and 10,000 auctions. There are no multiple
  roots, so root robustification has no additional cost.
- On the `n^(-0.35)` path at 10,000 auctions, 30.5 percent of replications have
  multiple roots. The selected-root and root-union sets both cover in 91.5
  percent of replications, but their mean lengths are 0.2051 and 0.4064.

The omitted roots therefore do not explain the remaining undercoverage in this
design. The stable three-bandwidth envelope is the preferred exact-target
prototype. It is a finite-sample diagnostic, not a proved honest confidence
set; validity still needs a uniform bias expansion and a joint theory for the
root process.

## Reproduction

```bash
python3 code/root_robust_inference.py --reps 200 --exponents 0.2 0.35
```

The script writes raw draws, a summary table, a LaTeX table, its configuration,
and the paper figure under `paper/tables/` and `paper/figures/`.
