# Replication Guide

The repository separates a fast integrity check from expensive Monte Carlo
regeneration. The integrity check is suitable for every commit; the full runs
are submission-archive tasks and may take hours on CPU.

## 1. Install

Create a fresh Python environment, then run from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

`environment.json` records the exact reference environment used for the
repository-level audit. It is descriptive rather than a cross-platform lock.

## 2. Fast integrity check

```bash
make smoke
```

This imports every experiment, verifies code/config seed agreement, tests the
fold partition and core score/root/interval helpers on deterministic toy data,
checks submission files, and verifies every committed table and figure against
`artifact_manifest.json`. It does not train a WGAN or overwrite paper results.

## 3. Full result regeneration

Run the experiments in dependency order:

```bash
python3 code/baseline_direct_inversion.py --reps 500
python3 code/wgan_gp_baseline.py --reps 5 --steps 1200 --device cpu
python3 code/crossfit_wgan_pilot.py --reps 5 --folds 3 --steps 1200 --device cpu
python3 code/orthogonal_wgan_dml.py --reps 5 --folds 3 --steps 1200 --bandwidth 0.15 --device cpu
python3 code/local_nuisance_ablation.py --reps 200 --folds 3 --bandwidth 0.15
python3 code/bandwidth_path.py --reps 200 --exponents 0 0.2 0.35 0.5
python3 code/bias_corrected_inference.py --reps 200 --exponents 0.2 0.35
python3 code/root_robust_inference.py --reps 200 --exponents 0.2 0.35
python3 code/external_validity_monte_carlo.py --reps 200
```

The WGAN outputs feed the cross-fit and orthogonal-score scripts, so their
order matters. All scripts write directly to `paper/tables/` and
`paper/figures/`. Regenerate the checksum manifest only after reviewing an
intentional full or partial result update:

```bash
python3 code/build_artifact_manifest.py --write
make smoke
```

## 4. Manuscript

With a LaTeX installation containing `latexmk`:

```bash
make paper
make paper-anonymous
```

The anonymous entry point removes the author identity and public repository URL.

## Reproducibility boundary

The seeds and deterministic offset policy are recorded in `seeds.json` and in
each experiment's generated configuration file. CPU execution is the reference
for WGAN results. Deep-learning kernels and floating-point libraries can still
produce small platform-dependent differences, so the checksum manifest verifies
the archived artifacts; regenerated results should be assessed using Monte Carlo
precision rather than byte identity alone.
