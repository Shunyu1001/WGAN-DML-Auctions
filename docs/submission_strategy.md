# Submission Strategy

Checked against official journal guidance on July 19, 2026.

## Recommended Sequence

### 1. Current-form target: International Journal of Computational Economics and Econometrics

This is the most direct fit for the current methodological and Monte Carlo
paper. The journal explicitly covers computational econometrics, Monte Carlo
simulation, robustness analysis, and software implementation. It accepts both
theoretical and applied work and uses double-blind review. The anonymous build
in `paper/main_anonymous.tex` is therefore the working submission file.

Official sources: [aims and scope](https://www.inderscience.com/jhome.php?jcode=ijcee),
[author guidance](https://www.inderscience.com/mobile/inauthors/index.php?pid=70).

Before submission, obtain an adviser-level judgment on journal quality and
prepare the four external expert suggestions required by the submission system.
Do not submit automatically from this repository.

### 2. Upward target after an empirical bridge: The Econometrics Journal

The methodological subject is in scope, but the present manuscript does not yet
meet the normal submission conditions. The journal requests a summary of at
most 150 words, normally no more than 20 pages including the printed appendix,
an empirical application, and a one-page main-text summary of simulations. The
new abstract meets the 150-word constraint, but the paper remains 28 pages and
contains a Monte Carlo study rather than an empirical application.

Official sources: [scope](https://res.org.uk/journals/the-econometrics-journal/),
[submission guidelines](https://res.org.uk/journals/the-econometrics-journal/submissions/).

Required upgrade path:

1. add a credible auction dataset with observed maxima and bidder counts;
2. reduce the main paper to 20 pages by moving secondary simulation tables,
   figures, implementation detail, and exact-target diagnostics online;
3. retain the fixed-bandwidth theorem and its proof in the printed paper;
4. lead with the local-information boundary and the hybrid estimator, not the
   WGAN training mechanics.

### 3. Method-focused alternative after empirical illustration: Journal of Econometric Methods

The journal welcomes new econometric and computational methods and requires
replication materials, but it also states that methodology should be thoroughly
illustrated with empirical data and written for practitioners. It is therefore
not a clean target for the current simulation-only version. Reconsider it after
the same empirical bridge required for the upward target.

Official source: [journal scope and author information](https://www.degruyterbrill.com/journal/key/jem/html?lang=en).

## Current Submission Assets

- identified manuscript: `paper/main.tex`;
- double-blind manuscript: `paper/main_anonymous.tex`;
- abstract below 150 words;
- keywords and JEL classifications;
- data-and-code availability statement;
- public replication repository for the identified version;
- explicit separation between the fixed-bandwidth theorem and diagnostic
  exact-reserve intervals.

## Next Packaging Sprint

Create a clean replication archive with one lightweight smoke-test command and
one full-results command. Record the Python and LaTeX environments, expected
runtime, random seeds, generated file manifest, and checksums for final tables
and figures. This is higher priority than another estimator or confidence-set
variant.
