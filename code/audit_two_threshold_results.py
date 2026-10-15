#!/usr/bin/env python3
"""Read-only audit of the independently archived two-threshold experiment.

The default reads committed results, not ignored caches or output directories.
Pass --results-root output/tables to inspect the historical local output layout.
Comparison CSV/PNG regeneration is opt-in via --write-comparison DIRECTORY;
no results, metadata, or plots are written during the default audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments/two_threshold"
DEFAULT_RESULTS = ARCHIVE / "results"
DEFAULT_SOURCE = ARCHIVE / "provenance/two_threshold_pilot_2026_10_07.py"
EXPERIMENTS = (
    "two_threshold_pilot",
    "two_threshold_empirical_200",
    "two_threshold_n2_200",
)
GROUP_KEYS = ["auctions", "family", "method", "delta", "bandwidth"]


def assert_table_matches(actual, expected, keys):
    """Compare tables without depending on row order or float CSV round trips."""
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual, expected, check_dtype=False, check_exact=False,
        rtol=1e-10, atol=1e-12,
    )


def audit_result_folder(folder, source_hash):
    """Check a completed result set without loading models or writing files."""
    raw = pd.read_csv(folder / "two_threshold_raw.csv")
    summary = pd.read_csv(folder / "two_threshold_summary.csv")
    targets = pd.read_csv(folder / "two_threshold_targets.csv")
    cfg = json.loads((folder / "two_threshold_config.json").read_text())
    assert cfg["source_hash"] == source_hash, (folder.name, "source hash mismatch")
    assert np.isfinite(raw.estimate).all(), (folder.name, "nonfinite estimate")
    assert not raw.duplicated(["auctions", "repetition", "method", "delta", "bandwidth"]).any()
    assert not summary.duplicated(GROUP_KEYS).any()
    assert set(raw.auctions) == set(cfg["sample_sizes"])
    assert set(raw.repetition) == set(range(1, cfg["repetitions"] + 1))

    threshold_methods = {"Threshold empirical", "Threshold plug-in"}
    gaussian_methods = {"Gaussian empirical", "Gaussian plug-in"}
    if not cfg["no_gan"]:
        threshold_methods |= {"Threshold WGAN", "Threshold calibrated"}
        gaussian_methods |= {"Gaussian WGAN", "Gaussian calibrated"}
    if cfg["skip_kernel"]:
        gaussian_methods = set()
    per_rep = len(threshold_methods) * len(cfg["deltas"]) + len(gaussian_methods)
    expected_rows = len(cfg["sample_sizes"]) * cfg["repetitions"] * per_rep
    assert len(raw) == expected_rows, (folder.name, len(raw), expected_rows)
    for (n, rep), cell in raw.groupby(["auctions", "repetition"]):
        assert len(cell) == per_rep, (folder.name, n, rep)
        for delta in cfg["deltas"]:
            part = cell[(cell.family == "Two-threshold") & np.isclose(cell.delta, delta)]
            assert set(part.method) == threshold_methods
        gaussian = cell[cell.family == "Gaussian"]
        assert set(gaussian.method) == gaussian_methods
        if len(gaussian):
            np.testing.assert_allclose(gaussian.bandwidth, cfg["kernel_bandwidth"])

    plugin = raw[raw.method == "Threshold plug-in"]
    for field in ["covers_target", "covers_exact", "score_set_length",
                  "score_set_components", "score_set_empty", "score_set_boundary"]:
        assert plugin[field].isna().all(), (folder.name, "plug-in inference", field)
    assert (plugin.score_set_segments == "[]").all()
    np.testing.assert_allclose(raw.error_target, raw.estimate - raw.target, atol=1e-12)
    np.testing.assert_allclose(raw.error_exact, raw.estimate - raw.exact_reserve, atol=1e-12)
    np.testing.assert_allclose(raw.exact_reserve, cfg["exact_reserve"], atol=1e-12)
    assert set(targets.delta) == set(cfg["deltas"])
    assert not targets.duplicated("delta").any()
    np.testing.assert_allclose(targets.target_bias, targets.target - targets.exact_reserve)
    for _, row in targets.iterrows():
        selected = raw[(raw.family == "Two-threshold") & np.isclose(raw.delta, row.delta)]
        np.testing.assert_allclose(selected.target, row.target, atol=1e-12)

    expected_summary = []
    for keys, group in raw.groupby(GROUP_KEYS, dropna=False, sort=False):
        expected_summary.append(dict(
            zip(GROUP_KEYS, keys), repetitions=len(group),
            mean_estimate=group.estimate.mean(),
            bias_target=group.error_target.mean(), bias_exact=group.error_exact.mean(),
            rmse_target=np.sqrt(np.mean(group.error_target**2)),
            rmse_exact=np.sqrt(np.mean(group.error_exact**2)),
            coverage_target=group.covers_target.dropna().astype(float).mean(),
            coverage_exact=group.covers_exact.dropna().astype(float).mean(),
            mean_revenue_regret=group.revenue_regret.mean(),
            missing_root_rate=(group.roots == 0).mean(),
            multiple_root_rate=(group.roots > 1).mean(),
        ))
    expected_summary = pd.DataFrame(expected_summary)
    assert_table_matches(summary[expected_summary.columns], expected_summary, GROUP_KEYS)

    paired_path = folder / "two_threshold_paired.csv"
    if cfg["no_gan"]:
        # Original empirical-only runs legitimately contain a one-byte empty CSV:
        # the archived paired table concerns calibrated GAN versus other learners.
        assert not paired_path.read_text().strip()
    else:
        expected_pairs = []
        for (n, delta), group in raw[raw.family == "Two-threshold"].groupby(["auctions", "delta"]):
            squared = group.pivot(index="repetition", columns="method", values="error_exact")**2
            for benchmark in sorted(threshold_methods - {"Threshold calibrated"}):
                difference = squared["Threshold calibrated"] - squared[benchmark]
                expected_pairs.append(dict(
                    auctions=n, delta=delta, benchmark=benchmark,
                    mse_difference=difference.mean(),
                    paired_mcse=difference.std(ddof=1) / np.sqrt(len(difference)),
                ))
        assert_table_matches(pd.read_csv(paired_path), pd.DataFrame(expected_pairs),
                             ["auctions", "delta", "benchmark"])
    assert (folder / "two_threshold_pilot.png").is_file()
    return raw, cfg


def comparison_tables(new, new_cfg, legacy_root):
    old = pd.read_csv(legacy_root / "calibrated_wgan_raw.csv")
    old_cfg = json.loads((legacy_root / "calibrated_wgan_config.json").read_text())
    for key in ["seed", "sample_sizes", "repetitions", "bidders", "folds",
                "training_steps", "generated_maxima", "pseudo_observations"]:
        assert new_cfg[key] == old_cfg[key], key
    check = old[(old.procedure == "Fixed bandwidth") & (old.method == "Calibrated WGAN-DML")].merge(
        new[new.method == "Gaussian calibrated"], on=["auctions", "repetition"], suffixes=("_old", "_new"))
    assert len(check) == len(new_cfg["sample_sizes"]) * new_cfg["repetitions"]
    np.testing.assert_allclose(check.estimate_old, check.estimate_new, rtol=0, atol=1e-12)
    legacy = old[(old.procedure == "Richardson") & (old.method == "Calibrated WGAN-DML")]
    rows, pairs = [], []
    for (n, delta, method), group in new[new.family == "Two-threshold"].groupby(["auctions", "delta", "method"]):
        rows.append(dict(auctions=n, delta=delta, method=method, repetitions=len(group),
                         rmse_exact=float(np.sqrt(np.mean(group.error_exact**2))),
                         mean_revenue_regret=float(group.revenue_regret.mean())))
        if method == "Threshold calibrated":
            merged = group.merge(legacy[legacy.auctions == n], on=["auctions", "repetition"], suffixes=("_new", "_old"))
            assert len(merged) == len(group)
            differences = merged.error_exact_new**2 - merged.error_exact_old**2
            pairs.append(dict(auctions=n, delta=delta, benchmark="Existing calibrated Richardson",
                              mse_difference=float(differences.mean()),
                              paired_mcse=float(differences.std(ddof=1) / np.sqrt(len(differences)))))
    for n, group in legacy.groupby("auctions"):
        rows.append(dict(auctions=n, delta=np.nan, method="Existing calibrated Richardson",
                         repetitions=len(group), rmse_exact=float(np.sqrt(np.mean(group.error_exact**2))),
                         mean_revenue_regret=float(group.revenue_regret.mean())))
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def write_comparison(comparison, pairs, output):
    """Explicit optional regeneration; never called by the default audit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "comparison_exact_reserve.csv", index=False)
    pairs.to_csv(output / "comparison_paired_richardson.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, delta in zip(axes.flat, [.05, .10, .15, .20]):
        for method, color in [("Threshold WGAN", "#ad4a45"), ("Threshold calibrated", "#176d9c"),
                              ("Threshold empirical", "#3f846c")]:
            group = comparison[(comparison.method == method) & np.isclose(comparison.delta, delta)].sort_values("auctions")
            ax.plot(group.auctions, group.rmse_exact, "o-", color=color, label=method.replace("Threshold ", ""))
        group = comparison[comparison.method == "Existing calibrated Richardson"].sort_values("auctions")
        ax.plot(group.auctions, group.rmse_exact, "o--", color="#333333", label="existing, bias-corrected")
        ax.set_title(f"Price half-window = {delta:.2f}")
        ax.set_xscale("log")
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Number of auctions")
    fig.supylabel("RMSE relative to exact optimal reserve")
    fig.suptitle("Paired exploratory comparison: N=5, 10 replications per sample size")
    fig.tight_layout()
    fig.savefig(output / "comparison_all_deltas.png", dpi=180)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS,
                        help="Parent containing the three named experiment folders.")
    parser.add_argument("--source-snapshot", type=Path, default=DEFAULT_SOURCE,
                        help="Source whose SHA-256 must match the recorded run metadata.")
    parser.add_argument("--legacy-root", type=Path, default=ROOT / "paper/tables")
    parser.add_argument("--write-comparison", type=Path, default=None, metavar="DIRECTORY",
                        help="Opt in to regenerate comparison CSV/PNG files at this path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source_hash = hashlib.sha256(args.source_snapshot.read_bytes()).hexdigest()
    completed = {}
    for name in EXPERIMENTS:
        completed[name] = audit_result_folder(args.results_root / name, source_hash)
    new, new_cfg = completed["two_threshold_pilot"]
    comparison, pairs = comparison_tables(new, new_cfg, args.legacy_root)
    pilot = args.results_root / "two_threshold_pilot"
    assert_table_matches(pd.read_csv(pilot / "comparison_exact_reserve.csv"), comparison,
                         ["auctions", "delta", "method"])
    assert_table_matches(pd.read_csv(pilot / "comparison_paired_richardson.csv"), pairs,
                         ["auctions", "delta", "benchmark"])
    assert (pilot / "comparison_all_deltas.png").is_file()
    assert (pilot / "experiment_notes.txt").is_file()
    if args.write_comparison is not None:
        write_comparison(comparison, pairs, args.write_comparison)
    print("Audit passed: complete cells, no duplicates, valid estimates, no plug-in CI claims;")
    print("archived summaries/pairs match raw rows and source metadata matches the preserved snapshot.")
    print("Legacy and rerun fixed-bandwidth estimates match for all 30 replications.")
    print("Default audit is read-only; no GAN caches or training are required.")
    print(comparison[(comparison.delta == .15) | comparison.delta.isna()].to_string(index=False))


if __name__ == "__main__":
    main()
