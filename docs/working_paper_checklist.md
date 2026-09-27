# Working Paper Release Checklist

This checklist defines a journal-neutral release. Outlet selection and
outlet-specific formatting are later decisions.

## Scientific Content

- [x] The title and abstract state the object, data limitation, estimator, and
  principal result without relying on a journal-specific framing.
- [x] The fixed-bandwidth regularized reserve is identified as the formal
  inferential target.
- [x] Exact-reserve Richardson, residual-bias, and root-union results are
  labelled diagnostic rather than uniformly valid.
- [x] The role of WGAN-GP is separated from the empirical-local score nuisance.
- [x] Maintained auction assumptions and the empirical implementation boundary
  are explicit.
- [x] No empirical result is claimed from simulated data.

## Manuscript Integrity

- [x] The identified and anonymous entry points reuse the same manuscript.
- [x] The abstract is 140 words.
- [x] Keywords and JEL classifications are present.
- [x] Data/code, conflicts-of-interest, and AI-assisted-tool declarations are
  present.
- [x] PDF title, subject, keywords, and identified/anonymous author metadata are
  set from the shared source.
- [x] Cross-references and citations compile without errors or warnings.
- [x] No placeholder, TODO, or journal name appears in the circulated paper.

## Replication and Traceability

- [x] The fast smoke test checks imports, seeds, cross-fitting helpers, score
  helpers, manuscript files, and committed artifact checksums.
- [x] Full Monte Carlo commands and the expensive-computation boundary are
  documented.
- [x] Every committed table and figure is covered by the artifact manifest.
- [x] Citation metadata are available in `CITATION.cff`.
- [ ] Create a version tag only after the author/contact line is confirmed.

## Author Confirmation Before Public Release

- [ ] Confirm the author name exactly as it should appear.
- [ ] Add the preferred affiliation and contact email to the identified PDF.
- [ ] Confirm whether the public version should display a date, version number,
  and repository URL on the title page.
- [ ] Confirm that every coauthor, if any, approves the frozen version.

## Release Command

From the repository root, run:

```bash
make release-check
```

Then record the commit hash, inspect both PDFs visually, and create a versioned
release only after the four author-confirmation items are complete.
