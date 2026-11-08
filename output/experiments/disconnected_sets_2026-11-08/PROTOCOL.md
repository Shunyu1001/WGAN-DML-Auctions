# Disconnected confidence set diagnosis

This experiment studies repetitions 29 and 50 of the completed fresh-seed n=500 validation. It reuses the original data seeds and saved generated samples. No generator is trained, no observation is removed, and no manuscript or frozen thesis file is edited.

The main numerical audit compares the original score inversion on grids of 281, 561, 1121, and 2241 points with Brent endpoint refinement. It also checks candidate-dependent score variance against a direct centered variance and extended-precision aggregation. CDF floors are not changed. Candidate probes below the prespecified search interval are diagnostic only and never alter the confidence set.

The influence audit records fold CDFs, score mean and variance, the largest observation contributions to variance, and the effective number of variance contributors. Oracle conditional moments are computed by numerical integration solely to distinguish model information from nuisance estimation effects; they do not enter any feasible interval.

Two simple responses are considered. First, retain the full confidence set and report its convex hull only if a single interval is required. This is a conservative presentation envelope, since every original accepted parameter remains in it. The original components and boundary flags remain available. Second, an exploratory fixed-variance score inversion keeps the original nonlinear score mean but fixes the denominator at the original estimated root. It uses the same 1.96 critical value, bounds, 281-point inversion grid and original point estimate. It is evaluated on the saved validation samples and is not independent confirmation or an automatically adopted replacement.

The analysis does not discard components, raise the lower search bound after seeing the data, trim influential observations, or tune cutoffs to obtain connected sets. All diagnostic results are stored separately from the completed simulation and its source snapshot.
