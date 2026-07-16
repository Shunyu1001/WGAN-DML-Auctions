# Structural WGAN-GP Pilot

## Structural restriction

The generator produces one log valuation from one bidder-level noise draw. For
auction $j$, it is called independently $N$ times and the synthetic maximum
is formed only afterwards:

\[
L_{ij}=G_\gamma(Z_{ij}),\qquad
\widetilde B_j=\exp\!\left(\max_i L_{ij}\right).
\]

This differs from a generator that outputs the entire bidder vector from one
common noise draw. The latter generally introduces dependence across bidders
and is inconsistent with the iid private-values benchmark.

## Training convention

- The critic receives standardized log maximum bids.
- The generator and critic each have two hidden layers with 64 units.
- Training uses WGAN-GP with five critic updates per generator update.
- The bidder-level latent dimension is eight.
- Each run uses 1,200 generator steps and a batch size of 512.
- A checkpoint is chosen using the Wasserstein distance between observed and
  generated maximum bids.
- The true reserve is never used for training or checkpoint selection.

The log-generator output is bounded to the observed log-maximum location plus
or minus four standard deviations. This stabilizes the pilot while covering
the region relevant to the reserve. The restriction should be relaxed and
stress-tested in the full Monte Carlo design.

## Pilot design

The pilot uses the same lognormal valuation model and five-bidder auction as
the direct-inversion benchmark. It contains five training seeds at each of
500, 2,000, and 10,000 auctions. These repetitions are sufficient for a
pipeline and failure-mode check, not for final Monte Carlo inference.

The main diagnostic is deliberate: a small Wasserstein discrepancy for the
observable maximum can coexist with a visible lower-tail error in the latent
CDF and a biased reserve. This supplies the concrete motivation for
cross-fitting and an orthogonal reserve score, but it does not by itself show
that DML will remove the bias.
