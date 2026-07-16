# Baseline Model and First Estimator

## Observed data

The first benchmark deliberately fixes the economic environment before adding
WGAN-GP or DML. Auction \(j\) has a known and constant number \(N\) of bidders.
Private values are independent across bidders and auctions and satisfy

\[
V_{ij}\overset{\mathrm{iid}}{\sim}F_0.
\]

The econometrician observes \(O_j=(B_j,N)\), where

\[
B_j=\max_{1\leq i\leq N}V_{ij}.
\]

Under truthful bidding in a second-price auction, \(B_j\) is the highest
submitted bid. It is not the transaction price, which is the second-highest
bid when the reserve does not bind. An empirical application must therefore
verify that its observed price variable has the required interpretation.

## Target and identification

For a regular distribution with density \(f_0\), the Myerson reserve is the
scalar \(r_0\) satisfying

\[
1-F_0(r_0)-r_0f_0(r_0)=0.
\]

If \(H_0\) is the CDF of the observed maximum, independence gives

\[
H_0(b)=F_0(b)^N,\qquad F_0(b)=H_0(b)^{1/N}.
\]

This identity is the nonparametric identification result and the benchmark
against which the WGAN nuisance learner will be judged.

## Direct-inversion estimator

Let \(\widehat H\) be the empirical CDF of auction maxima. Define

\[
\widehat F(v)=\widehat H(v)^{1/N},
\qquad
\widehat r=\arg\max_r r\{1-\widehat F(r)\}.
\]

The code evaluates the objective at the observed maxima. This benchmark does
not estimate a density and is intentionally separate from the later WGAN-DML
estimator. Its finite-sample instability reveals how much information the
maximum order statistic contains near the reserve.

## Scope of the first experiment

- Lognormal private values with \(\mu=0\) and \(\sigma=0.5\).
- Five bidders per auction.
- Sample sizes of 200, 500, 2,000, and 10,000 auctions.
- 500 Monte Carlo repetitions and a fixed seed.
- Bias, standard deviation, RMSE, absolute error, and revenue regret.

Later experiments can relax fixed competition, add observed covariates, and
compare direct inversion with WGAN-GP, cross-fitted plug-in, and orthogonal
DML estimators.
