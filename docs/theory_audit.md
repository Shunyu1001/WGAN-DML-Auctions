# Fixed Bandwidth Theory and Implementation

The active PSRM working revision targets the unique single-crossing reserve
on a positive, prespecified compact price interval. The MA thesis freeze is
unchanged. The main inferential target is the fixed-bandwidth parameter, not
the exact unsmoothed reserve.

## Primitive Fixed Bandwidth Result

The model requires complete, untruncated highest truthful bids, a fixed known
number of iid private values, independent auctions, enforced second-price
rules, and a seller retention value of zero. The economic reserve and the
Gaussian-regularized reserve are defined separately.

For fixed bandwidth and a fixed number of nondegenerate folds, the training
empirical kernel moments and their first price derivatives converge uniformly
at `O_p(sqrt(log(n)/n))`. A fixed finite pseudo-count gives the generated
moments weight `O(1/n)`; generated Gaussian signals and their derivatives have
fixed envelopes. Thus the calibrated moments inherit the same rates without
assuming a consistent standalone GAN or increasing synthetic-sample size.

Uniform score and derivative convergence, overlap, endpoint signs, and a
strictly negative local slope imply root and Jacobian consistency. The proof
now derives these facts rather than assuming the selected root is consistent.
A uniform empirical-process argument justifies evaluation at the random root.
The orthogonal remainder is `O(delta_A^2 + abs(delta_A*delta_G))`. With a
numerical score residual `o(n^(-1/2))`, asymptotic linearity and Wald coverage
follow for the regularized target.

The score is locally orthogonal but not two-sided doubly robust: correct A
eliminates G misspecification in its expectation, while correct G alone does
not generally eliminate A misspecification. Empirical DML, fixed-pseudo-count
calibrated DML, and the same-kernel full-sample plug-in have the same first-order
influence function under the common assumptions. No GAN efficiency gain is
claimed.

## Numerical Rule

`code/calibrated_wgan_pilot.py` accepts only a single detected interior downward
root with correct endpoint signs, inactive CDF clipping, a sufficiently
negative analytic total Jacobian, a small score residual, and positive finite
influence variance. A finite grid is a diagnostic rather than a global root
certificate; the asymptotic argument supplies eventual uniqueness under its
stated conditions. Root and residual tolerances are explicit and can decrease
with sample size. Floating-point precision remains a finite-sample limit.

Rejected runs retain a failure reason and NaN estimate/SE. Unconditional
coverage counts failure as noncoverage. Conditional coverage and valid-fit
error statistics show their valid counts. Paired comparisons report the number
of jointly valid replications. Richardson propagates any failed bandwidth fit.

The direct-inversion benchmark uses the empirical CDF left limit at each
observed candidate price, so a value equal to the reserve is eligible to trade.
This resolves the discontinuous-objective convention explicitly.

## Results and Remaining Boundary

Existing manuscript tables are archived results from earlier root and
right-continuous benchmark conventions. They have not been silently recomputed.
Revised runs use a new output directory and a distinct algorithm version;
existing raw output is protected against overwrite and legacy/v2 results cannot
be pooled as one estimator.

Exact-reserve Richardson inference remains a conditional extension. Higher-order
smoothness, shrinking-bandwidth uniform rates, stable roots, joint triangular
array limits, and covariance consistency are additional requirements. Bias
envelopes and root unions remain diagnostics, not honest uniform confidence
sets. No new entry model, covariate model, network, or loss was introduced.
