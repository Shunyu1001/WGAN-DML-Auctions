# Fixed bandwidth confidence sets and validation

The active manuscript is `paper/main.tex`; its current PDF is
`output/pdf/WGAN-DML-Auctions-PSRM-Score-Sets-2026-11-08.pdf` (52 pages).
The source archive in `output/psrm_reporting_update_2026-11-08/` contains the
same LaTeX sources. This revision updates the PSRM working paper only. It does
not change the separately frozen MA thesis or the archived 0.1.4 release.

## Statistical scope

The inferential target is the fixed-bandwidth reserve, not the exact unsmoothed
reserve. All detected score-inversion acceptance components are retained.
Disconnected sets and contact with a search boundary are reported as diagnostic
flags, not automatically classified as computational failures. The interval
envelope spans the outer endpoints, includes rejected gaps, and is an additional
conservative report rather than the original confidence set.

Fresh validation uses seed 2026110800, with 100 independent repetitions each at
500 and 2,000 auctions. The three methods use the same paired samples. At 500
auctions, full-set coverage is 93%, 93%, and 92% for calibrated GAN,
empirical-local estimation, and same-kernel plug-in respectively. At 2,000
auctions it is 95% for all three. No computational failures occurred in these
cells. Two underlying 500-auction replications yield four disconnected,
lower-boundary-touching method records. The envelope has the same coverage as
the full set in this validation, but a greater average length for the first two
500-auction cells. These results do not establish a stable GAN advantage or
exact finite-sample 95% coverage.

## Evidence paths

- `output/experiments/fixed_h_score_validation_2026-11-08/`: the fixed protocol,
  seed plan, immutable source snapshot, shard results and combined summaries.
- `output/reports/confidence_sets_2026-11-08/`: original components, full-set
  and envelope summaries, flagged cases, and reporting checks.
- `output/experiments/disconnected_sets_2026-11-08/`: grid and variance
  diagnostics and the exploratory fixed-variance sensitivity exercise.
- `output/psrm_reporting_update_2026-11-08/`: source archive and table-generation
  integrity checks. Its `revision_manifest.json` records the earlier local
  packaging operation; its local-only statement describes that operation,
  not the subsequent online synchronization.

Synthetic GAN caches and process logs are not included. Re-training can differ
across numerical environments. Original archived experiments remain in their
existing directories and are not pooled with the new validation.

## Lightweight reproduction

From the repository root, with `requirements.txt` installed:

```bash
python code/build_artifact_manifest.py --check
python -m unittest discover -s code -p 'test_confidence_set_reporting.py'
python code/build_psrm_reporting_tables.py
python code/plot_disconnected_sets.py \
  --output-dir output/experiments/disconnected_sets_2026-11-08 \
  --paper-output paper/figures/disconnected_sets_diagnostic.png
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Table regeneration reads the archived numerical results; it does not rerun
estimation or change accepted confidence-set components. Use the source
snapshot and recorded protocol for a fresh estimation run, and write any new
run to a separate directory.
