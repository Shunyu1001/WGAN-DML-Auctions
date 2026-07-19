# IJCEE Submission Checklist

Verified against the official IJCEE and Inderscience author pages on July 19,
2026. The journal uses double-blind review and its online submission system.

Official references: [IJCEE journal page](https://www.inderscience.com/jhome.php?jcode=IJCEE),
[author guidelines](https://www.inderscience.com/mobile/inauthors/index.php?pid=70),
[article preparation](https://www.inderscience.com/mobile/inauthors/index.php?pid=71),
and [submission checklist](https://www.inderscience.com/mobile/inauthors/index.php?pid=72).

## Completed in the Repository

- [x] Concise title without unexplained acronyms.
- [x] Factual abstract below 150 words; WGAN-GP is defined at first mention.
- [x] Ten searchable keyword phrases and JEL classifications.
- [x] Anonymous PDF entry point with no author name, affiliation,
  acknowledgements, public repository URL, or author PDF metadata.
- [x] Conflicts-of-interest declaration included in the manuscript.
- [x] AI-assisted-tools disclosure included without identifying the author.
- [x] Data statement records that the study uses simulations and no restricted
  or human-participant data.
- [x] Harvard-style name-year citations and alphabetical reference list.
- [x] Identified cover letter source and one-page PDF.
- [x] Replication guide, seeds, environment record, CI smoke test, and checksums.

## Author Confirmation Required

These are facts that the repository cannot decide. Complete them immediately
before entering the online system.

- [ ] Confirm the manuscript is original, has not been published in English,
  and is not under review at another journal.
- [ ] If it was previously submitted elsewhere, retain formal rejection or
  withdrawal evidence.
- [ ] Confirm the definitive author list, order, and corresponding author.
- [ ] Enter position, department, institution, full postal address, and email.
- [ ] Prepare a 100-word biography (maximum 150 words).
- [ ] Confirm all authors have read and approved the submitted version.
- [ ] Decide whether any earlier public repository copy should be treated as a
  preprint and, if so, confirm it meets Inderscience's repository conditions.

## Four Experts Required by the Submission System

- [ ] Record four experts' names, full postal addresses, email addresses, and
  relevant expertise using `docs/ijcee_submission_metadata.md`.
- [ ] Confirm none is on the editorial board of any Inderscience journal.
- [ ] Confirm none is from an author's institution.
- [ ] Confirm at least two are from a country different from every author.
- [ ] Screen recent co-authors, advisers, close collaborators, and personal
  conflicts even where the minimum publisher rule does not explicitly say so.

Do not select experts merely because they are prominent or convenient. Start
from scholars cited in the recent literature on auction order statistics,
semiparametric inference, and generative structural estimation, then verify
current affiliation and conflicts manually.

## Final Editorial Pass

- [ ] Perform a UK-English spelling pass across the full manuscript.
- [ ] Inspect every colour figure in greyscale and confirm labels remain legible.
- [ ] Confirm the editor will accept figures embedded in the review PDF; if not,
  place figures and captions at the end as requested in the general guidance.
- [ ] Check every DOI, year, volume, issue, and page range in the bibliography.
- [ ] Rebuild `paper/main_anonymous.pdf`, inspect its properties, and run the
  identity-string check.
- [ ] Build the submission archive from the exact final commit and rerun
  `make smoke`.

## Files to Upload or Use

- Main review file: `paper/main_anonymous.pdf`.
- Cover letter: `paper/ijcee_cover_letter.pdf`.
- Form metadata: `docs/ijcee_submission_metadata.md`.
- Replication materials: provide the anonymous archive at review; do not expose
  the public repository URL in the double-blind manuscript.

No submission should be sent until every item in “Author Confirmation Required”
and “Four Experts” has been completed by the author.
