#!/usr/bin/env python3
"""Paired, exploratory two-threshold DML experiment; no kernel density estimate.

The fixed-delta target is the maximizer of a uniform-window average of the
single-buyer-equivalent revenue, not the exact Myerson reserve. Existing WGAN
samples are reused by an exact metadata hash. No paper artifacts are overwritten.
All deltas and all learners are retained, including overlap and root failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq

from baseline_direct_inversion import (
    SimulationConfig, estimate_reserve_from_maxima, make_revenue_interpolator,
    true_reserve, valuation_distribution,
)
from calibrated_wgan_pilot import (
    FoldMoments, calibration_weight, estimate as estimate_gaussian,
    synthetic_for_fold, wilson,
)
from crossfit_wgan_pilot import make_folds
from orthogonal_wgan_dml import DMLConfig, true_regularized_target
from wgan_gp_baseline import WGANConfig

ROOT = Path(__file__).resolve().parents[1]
PROBABILITY_FLOOR = 1e-8  # Stabilizes negative powers; finite-sample effects are NOT negligible by definition.
THRESHOLD_METHODS = ["Threshold WGAN", "Threshold calibrated", "Threshold empirical", "Threshold plug-in"]


@dataclass(frozen=True)
class ThresholdConfig:
    sample_sizes: tuple[int, ...] = (500, 2000, 10000)
    repetitions: int = 10
    deltas: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    bidders: int = 5
    folds: int = 3
    pseudo_observations: float = 50.0
    training_steps: int = 1200
    generated_maxima: int = 50000
    seed: int = 20260928  # Match the existing paired calibration pilot/cache.
    threads: int = 1
    lower: float = 0.45
    upper: float = 1.15
    grid_size: int = 1401
    kernel_bandwidth: float = 0.15


def ecdf(points, sample):
    """Right-continuous raw empirical CDF: no pseudo-counts or kernel."""
    sample = np.sort(np.asarray(sample, dtype=float).reshape(-1))
    if not len(sample):
        raise ValueError("An empirical CDF requires a nonempty sample.")
    return _ecdf_sorted(points, sample)


def _ecdf_sorted(points, ordered):
    return np.searchsorted(ordered, np.atleast_1d(points), side="right") / len(ordered)


def threshold_moment_and_derivatives(theta, p_minus, p_plus, delta, bidders):
    theta = np.asarray(theta, dtype=float)
    pm = np.clip(np.asarray(p_minus, dtype=float), PROBABILITY_FLOOR, 1.0)
    pp = np.clip(np.asarray(p_plus, dtype=float), PROBABILITY_FLOOR, 1.0)
    alpha = 1.0 / bidders
    m = (theta + delta) * (1 - pp**alpha) - (theta - delta) * (1 - pm**alpha)
    dm = alpha * (theta - delta) * pm**(alpha - 1)
    dp = -alpha * (theta + delta) * pp**(alpha - 1)
    return m, dm, dp


def raw_threshold_moment(theta, p_minus, p_plus, delta, bidders):
    """The plug-in objective remains defined at zero; do not floor its CDF."""
    theta = np.asarray(theta, dtype=float)
    pm, pp = np.clip(p_minus, 0, 1), np.clip(p_plus, 0, 1)
    return (theta+delta)*(1-pp**(1/bidders)) - (theta-delta)*(1-pm**(1/bidders))


def threshold_scores(theta, observed, p_minus, p_plus, delta, bidders):
    pm = np.clip(p_minus, PROBABILITY_FLOOR, 1.0)
    pp = np.clip(p_plus, PROBABILITY_FLOOR, 1.0)
    m, dm, dp = threshold_moment_and_derivatives(theta, pm, pp, delta, bidders)
    observed = np.asarray(observed)
    return m + dm * ((observed <= theta - delta) - pm) + dp * ((observed <= theta + delta) - pp)


class ThresholdFold:
    def __init__(self, observed, training, synthetic, delta, bidders):
        self.observed = np.asarray(observed, dtype=float)
        self.training = np.asarray(training, dtype=float)
        self.synthetic = np.asarray(synthetic, dtype=float)
        self._observed = np.sort(self.observed)
        self._training = np.sort(self.training)
        self._synthetic = np.sort(self.synthetic)
        self.delta, self.bidders = delta, bidders

    def nuisance(self, theta, weight):
        theta = np.atleast_1d(theta)
        pm = _ecdf_sorted(theta - self.delta, self._training)
        pp = _ecdf_sorted(theta + self.delta, self._training)
        if weight:
            if not len(self._synthetic):
                raise ValueError("A positive GAN weight requires synthetic observations.")
            pm = (1 - weight) * pm + weight * _ecdf_sorted(theta - self.delta, self._synthetic)
            pp = (1 - weight) * pp + weight * _ecdf_sorted(theta + self.delta, self._synthetic)
        return pm, pp

    def score_stats(self, theta, weight):
        """Fold score mean and raw second moment, preserving I-/I+ covariance."""
        theta = np.atleast_1d(theta)
        pm, pp = self.nuisance(theta, weight)
        pm, pp = np.maximum(pm, PROBABILITY_FLOOR), np.maximum(pp, PROBABILITY_FLOOR)
        m, dm, dp = threshold_moment_and_derivatives(theta, pm, pp, self.delta, self.bidders)
        hm = _ecdf_sorted(theta - self.delta, self._observed)
        hp = _ecdf_sorted(theta + self.delta, self._observed)
        c = m - dm * pm - dp * pp
        mean = c + dm * hm + dp * hp
        # I- <= I+, so E[I- I+] = E[I-], not a product of marginal means.
        second = c*c + dm*dm*hm + dp*dp*hp + 2*c*(dm*hm + dp*hp) + 2*dm*dp*hm
        return mean, second

    def score_mean(self, theta, weight):
        return self.score_stats(theta, weight)[0]

    def individual_scores(self, theta, weight):
        pm, pp = self.nuisance(theta, weight)
        return threshold_scores(theta, self.observed, pm[0], pp[0], self.delta, self.bidders)


def pooled_stats(folds, weights, theta):
    sizes = np.array([len(f.observed) for f in folds])
    stats = [f.score_stats(theta, w) for f, w in zip(folds, weights)]
    mean = np.average(np.array([s[0] for s in stats]), axis=0, weights=sizes)
    second = np.average(np.array([s[1] for s in stats]), axis=0, weights=sizes)
    n = sizes.sum()
    var = np.maximum((second - mean*mean) * n / (n - 1), 0)
    return mean, np.sqrt(var / n)


def confidence_segments(grid, accepted):
    """Grid-cell approximation, not a guaranteed outer envelope; retain components."""
    indices = np.flatnonzero(accepted)
    if not len(indices):
        return []
    blocks = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    half = (grid[1] - grid[0]) / 2
    return [[float(max(grid[0], grid[b[0]] - half)),
             float(min(grid[-1], grid[b[-1]] + half))] for b in blocks]


def estimate_threshold(folds, weights, reference, config, plugin=False):
    grid = np.linspace(config.lower, config.upper, config.grid_size)

    def moment(theta):
        if plugin:
            f = folds[0]
            pm, pp = f.nuisance(theta, 0.0)
            return raw_threshold_moment(theta, pm, pp, f.delta, f.bidders)
        return pooled_stats(folds, weights, theta)[0]

    values = moment(grid)
    # Down-crossings correspond to local maxima of the regularized objective.
    brackets = np.flatnonzero((values[:-1] >= 0) & (values[1:] <= 0))
    if len(brackets):
        candidates = grid[brackets] - values[brackets] * (grid[1] - grid[0]) / (values[brackets+1] - values[brackets])
        j = int(brackets[np.argmin(np.abs(candidates - reference))])
        left, right = float(grid[j]), float(grid[j+1])
        # Indicators jump: bisection locates a crossing; it does NOT certify a zero.
        while right - left > 1e-9:
            mid = (left + right) / 2
            if float(moment(mid)[0]) >= 0:
                left = mid
            else:
                right = mid
        r = min([left, right], key=lambda x: abs(float(moment(x)[0])))
    else:
        r = float(grid[np.argmin(np.abs(values))])
    score_mean, score_se = pooled_stats(folds, weights, grid)
    accepted = np.abs(score_mean) <= 1.96 * score_se
    segments = confidence_segments(grid, accepted) if not plugin else []
    pm_raw = np.array([f.nuisance(r, w)[0][0] for f, w in zip(folds, weights)])
    pp_raw = np.array([f.nuisance(r, w)[1][0] for f, w in zip(folds, weights)])
    mean_r, se_r = pooled_stats(folds, weights, r)
    lower_counts = [int(np.sum(f.training <= r - f.delta)) for f in folds]
    return dict(estimate=r, roots=int(len(brackets)), score_residual=float(moment(r)[0]),
                standardized_score_residual=float(abs(mean_r[0]) / se_r[0]) if se_r[0] > 0 else np.nan,
                score_set_segments=segments, score_set_length=sum(b-a for a, b in segments) if not plugin else np.nan,
                score_set_components=len(segments) if not plugin else np.nan,
                score_set_empty=not bool(segments) if not plugin else np.nan,
                score_set_boundary=bool(accepted[0] or accepted[-1]) if not plugin else np.nan,
                zero_nuisance_fraction=float(np.mean((pm_raw == 0) | (pp_raw == 0))),
                clipped_nuisance_fraction=float(np.mean((pm_raw < PROBABILITY_FLOOR) | (pp_raw < PROBABILITY_FLOOR))) if not plugin else 0.0,
                mean_lower_training_count=float(np.mean(lower_counts)))


def true_threshold_target(delta, baseline, lower=0.45, upper=1.15):
    distribution = valuation_distribution(baseline)
    def q(x):
        return x * distribution.sf(x)
    return float(brentq(lambda r: q(r+delta) - q(r-delta), lower, upper))


def cached_synthetic(training, n, rep, fold, config, allow_training):
    wc = WGANConfig(training_steps=config.training_steps, bidders=config.bidders,
                    generated_values=config.generated_maxima, device="cpu")
    seed = config.seed + n*100 + rep + 1000000 + (fold+1)*10000
    metadata = json.dumps(dict(wgan=asdict(wc), seed=seed,
        generated_maxima=config.generated_maxima,
        source_hash=hashlib.sha256((ROOT/"code/wgan_gp_baseline.py").read_bytes()).hexdigest(),
        data_hash=hashlib.sha256(training.tobytes()).hexdigest()), sort_keys=True)
    path = ROOT/"output/tables/calibration_cache"/(hashlib.sha256(metadata.encode()).hexdigest()[:20]+".npz")
    if not path.exists() and not allow_training:
        raise FileNotFoundError(f"Matching GAN cache missing for n={n}, rep={rep+1}, fold={fold+1}. "
                                "Use --allow-training to train, or --no-gan for empirical diagnostics.")
    return synthetic_for_fold(training, n, rep, fold, config, path.parent)


def score_covers(folds, weights, target):
    mean, se = pooled_stats(folds, weights, target)
    return bool(abs(mean[0]) <= 1.96 * se[0])


def run(config, output, no_gan=False, allow_training=False, skip_kernel=False):
    torch.set_num_threads(config.threads)
    output.mkdir(parents=True, exist_ok=True)
    baseline = SimulationConfig(bidders=config.bidders)
    distribution = valuation_distribution(baseline)
    r0 = true_reserve(baseline)
    targets = {d: true_threshold_target(d, baseline, config.lower, config.upper) for d in config.deltas}
    rh = true_regularized_target(DMLConfig(bandwidth=config.kernel_bandwidth, bidders=config.bidders)) if not skip_kernel else np.nan
    revenue = make_revenue_interpolator(baseline)
    optimal_revenue = float(revenue(r0))
    records = []
    for n in config.sample_sizes:
        for rep in range(config.repetitions):
            rng = np.random.default_rng(config.seed+n*100+rep)
            observed = distribution.rvs(size=(n, config.bidders), random_state=rng).max(axis=1)
            holdouts = make_folds(n, config.folds, config.seed+n*100+rep+31415926)
            reference = estimate_reserve_from_maxima(observed, config.bidders)
            fold_data = []
            for k, holdout in enumerate(holdouts):
                mask = np.ones(n, dtype=bool); mask[holdout] = False
                training = observed[mask]
                synthetic = np.empty(0) if no_gan else cached_synthetic(training, n, rep, k, config, allow_training)
                fold_data.append((observed[holdout], training, synthetic))
            weights = [calibration_weight(len(t), config.pseudo_observations) for _, t, _ in fold_data]
            learner_specs = [(THRESHOLD_METHODS[2], [0.0]*config.folds)]
            if not no_gan:
                learner_specs = [(THRESHOLD_METHODS[0], [1.0]*config.folds),
                                 (THRESHOLD_METHODS[1], weights)] + learner_specs
            for delta in config.deltas:
                folds = [ThresholdFold(o, t, s, delta, config.bidders) for o, t, s in fold_data]
                full = [ThresholdFold(observed, observed, np.empty(0), delta, config.bidders)]
                specs = [(m, folds, w, False) for m, w in learner_specs] + [(THRESHOLD_METHODS[3], full, [0.0], True)]
                for method, fs, ws, plugin in specs:
                    fit = estimate_threshold(fs, ws, reference, config, plugin)
                    r = fit.pop("estimate")
                    fit["score_set_segments"] = json.dumps(fit["score_set_segments"])
                    target = targets[delta]
                    pm_target = [f.nuisance(target, w)[0][0] for f, w in zip(fs, ws)]
                    records.append(dict(auctions=n, repetition=rep+1, family="Two-threshold", method=method,
                        delta=delta, bandwidth=np.nan, estimate=r, target=target, exact_reserve=r0,
                        error_target=r-target, error_exact=r-r0,
                        covers_target=score_covers(fs, ws, target) if not plugin else np.nan,
                        covers_exact=score_covers(fs, ws, r0) if not plugin else np.nan,
                        inference="pointwise-score-test inversion" if not plugin else "not computed for raw CDF plug-in",
                        wgan_weight=float(np.mean(ws)),
                        revenue_regret=max(optimal_revenue-float(revenue(r)), 0),
                        zero_lower_at_target=float(np.mean(np.array(pm_target) == 0)), **fit))
            if not skip_kernel:
                fs = [FoldMoments(o, t, s, config.kernel_bandwidth, config.bidders) for o, t, s in fold_data]
                specs = [("Gaussian empirical", fs, [0.0]*config.folds, False)]
                if not no_gan:
                    specs = [("Gaussian WGAN", fs, [1.0]*config.folds, False),
                             ("Gaussian calibrated", fs, weights, False)] + specs
                full = [FoldMoments(observed, observed, np.empty(0), config.kernel_bandwidth, config.bidders)]
                specs += [("Gaussian plug-in", full, [0.0], True)]
                for method, fs, ws, plugin in specs:
                    fit = estimate_gaussian(fs, ws, reference, plugin)
                    r, se = fit["estimate"], fit["standard_error"]
                    records.append(dict(auctions=n, repetition=rep+1, family="Gaussian", method=method,
                        delta=np.nan, bandwidth=config.kernel_bandwidth, estimate=r, target=rh, exact_reserve=r0,
                        error_target=r-rh, error_exact=r-r0, covers_target=bool(abs(r-rh)<=1.96*se),
                        covers_exact=bool(abs(r-r0)<=1.96*se), inference="existing Wald interval",
                        standard_error=se, interval_length=3.92*se, wgan_weight=float(np.mean(ws)),
                        revenue_regret=max(optimal_revenue-float(revenue(r)),0), roots=fit["roots"],
                        score_residual=fit["score_residual"]))
            pd.DataFrame(records).to_csv(output/"two_threshold_raw.csv", index=False)
            print(f"n={n}, replication={rep+1}/{config.repetitions}; paired data/folds/GAN samples", flush=True)
    raw = pd.DataFrame(records)
    summary = summarize(raw, output)
    oracle_rows = []
    for delta, target in targets.items():
        h_minus = float(distribution.cdf(target-delta)**config.bidders)
        oracle_rows.append(dict(delta=delta, target=target, exact_reserve=r0, target_bias=target-r0,
                                lower_threshold_probability=h_minus,
                                upper_threshold_probability=float(distribution.cdf(target+delta)**config.bidders)))
    pd.DataFrame(oracle_rows).to_csv(output/"two_threshold_targets.csv", index=False)
    metadata = dict(**asdict(config), no_gan=no_gan, allow_training=allow_training, skip_kernel=skip_kernel,
        probability_floor=PROBABILITY_FLOOR, exact_reserve=r0, kernel_target=rh,
        target="fixed-delta uniform-window average of single-buyer-equivalent revenue",
        root_selection="down-crossing nearest full-sample direct-inversion reference; no true target used",
        confidence="score-test inversion; all connected components retained; coverage evaluated by exact-target score test",
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_head=__import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        environment=dict(python=sys.version, numpy=np.__version__, scipy=__import__("scipy").__version__,
                         torch=torch.__version__, platform=platform.platform()))
    (output/"two_threshold_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
    draw(summary, output)
    return raw, summary


def summarize(raw, output):
    rows = []
    for keys, g in raw.groupby(["auctions", "family", "method", "delta", "bandwidth"], dropna=False, sort=False):
        n, family, method, delta, bandwidth = keys
        valid_coverage = g.covers_target.dropna().astype(float)
        lo, hi = wilson(valid_coverage.sum(), len(valid_coverage)) if len(valid_coverage) else (np.nan, np.nan)
        row = dict(auctions=n, family=family, method=method, delta=delta, bandwidth=bandwidth,
            repetitions=len(g), mean_estimate=g.estimate.mean(), bias_target=g.error_target.mean(),
            bias_exact=g.error_exact.mean(), rmse_target=np.sqrt(np.mean(g.error_target**2)),
            rmse_exact=np.sqrt(np.mean(g.error_exact**2)), coverage_target=valid_coverage.mean(),
            coverage_exact=g.covers_exact.dropna().astype(float).mean(), coverage_low=lo, coverage_high=hi,
            mean_revenue_regret=g.revenue_regret.mean(), missing_root_rate=(g.roots==0).mean(),
            multiple_root_rate=(g.roots>1).mean())
        for field in ["score_set_length", "score_set_components", "score_set_empty", "score_set_boundary",
                      "clipped_nuisance_fraction", "zero_lower_at_target", "mean_lower_training_count", "interval_length"]:
            if field in g:
                row["mean_"+field] = g[field].mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output/"two_threshold_summary.csv", index=False)
    paired = []
    for (n, delta), g in raw[raw.family=="Two-threshold"].groupby(["auctions", "delta"]):
        if THRESHOLD_METHODS[1] not in set(g.method):
            continue
        pivot = g.pivot(index="repetition", columns="method", values="error_exact")**2
        for benchmark in THRESHOLD_METHODS:
            if benchmark == THRESHOLD_METHODS[1]:
                continue
            difference = pivot[THRESHOLD_METHODS[1]] - pivot[benchmark]
            paired.append(dict(auctions=n, delta=delta, benchmark=benchmark,
                               mse_difference=difference.mean(), paired_mcse=difference.std(ddof=1)/np.sqrt(len(difference))))
    pd.DataFrame(paired).to_csv(output/"two_threshold_paired.csv", index=False)
    return summary


def draw(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    colors = ["#ad4a45", "#176d9c", "#3f846c", "#7a6b8d"]
    selected = summary[(summary.family=="Two-threshold") & np.isclose(summary.delta, .15)]
    for method, color in zip(THRESHOLD_METHODS, colors):
        g = selected[selected.method==method].sort_values("auctions")
        if len(g):
            axes[0].plot(g.auctions, g.rmse_exact, marker="o", color=color, label=method.replace("Threshold ", ""))
            axes[1].plot(g.auctions, g.coverage_target, marker="o", color=color)
    gaussian = summary[summary.method=="Gaussian calibrated"].sort_values("auctions")
    if len(gaussian):
        axes[0].plot(gaussian.auctions, gaussian.rmse_exact, "--o", color="#333333", label="existing Gaussian calibrated")
    axes[0].set_ylabel("RMSE relative to exact Myerson reserve")
    axes[1].set_ylabel("Coverage of each method's fixed-delta target")
    axes[1].axhline(.95, ls="--", color="gray", lw=1)
    axes[1].set_ylim(0, 1.03)
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Number of auctions")
        ax.grid(alpha=.2)
    fig.suptitle("Two-threshold pilot: delta=0.15; all results exploratory")
    fig.tight_layout()
    fig.savefig(output/"two_threshold_pilot.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 2000, 10000])
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--deltas", nargs="+", type=float, default=[.05, .10, .15, .20])
    parser.add_argument("--bidders", type=int, default=5)
    parser.add_argument("--no-gan", action="store_true")
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--skip-kernel", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT/"output/tables/two_threshold_pilot")
    args = parser.parse_args()
    config = ThresholdConfig(sample_sizes=tuple(args.sizes), repetitions=args.reps,
                             deltas=tuple(args.deltas), bidders=args.bidders)
    run(config, args.output, args.no_gan, args.allow_training, args.skip_kernel)


if __name__ == "__main__":
    main()
