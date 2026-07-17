#!/usr/bin/env python3
"""Three-bandwidth bias envelopes and root-robust reserve inference."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bandwidth_path import bandwidth_for, population_target
from baseline_direct_inversion import (
    SimulationConfig,
    estimate_reserve_from_maxima,
    true_reserve,
    valuation_distribution,
)
from crossfit_wgan_pilot import make_folds
from local_nuisance_ablation import AblationConfig, true_maximum_quantiles
from orthogonal_wgan_dml import orthogonal_scores, score_grid_for_fold


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "paper" / "tables"
FIGURE_DIR = ROOT / "paper" / "figures"
Z_95 = 1.96
METHOD_ORDER = [
    "Richardson Wald",
    "Selected-root bias envelope",
    "Root-robust bias envelope",
]


@dataclass(frozen=True)
class RootRobustConfig:
    bidders: int = 5
    sample_sizes: tuple[int, ...] = (500, 2_000, 10_000)
    repetitions: int = 200
    folds: int = 3
    reference_bandwidth: float = 0.15
    reference_sample_size: int = 500
    exponents: tuple[float, ...] = (0.2, 0.35)
    bandwidth_ratio: float = math.sqrt(2.0)
    theta_lower: float = 0.40
    theta_upper: float = 1.20
    theta_grid_size: int = 161
    jacobian_step: float = 0.0025
    target_integration_draws: int = 100_000
    seed: int = 20_260_716


def local_config(
    sample_size: int,
    bandwidth: float,
    config: RootRobustConfig,
) -> AblationConfig:
    return AblationConfig(
        bidders=config.bidders,
        sample_sizes=(sample_size,),
        repetitions=config.repetitions,
        folds=config.folds,
        bandwidth=bandwidth,
        theta_lower=config.theta_lower,
        theta_upper=config.theta_upper,
        theta_grid_size=config.theta_grid_size,
        jacobian_step=config.jacobian_step,
        target_integration_draws=config.target_integration_draws,
        seed=config.seed,
    )


def enumerate_roots(
    theta_grid: np.ndarray, score_grid: np.ndarray
) -> list[float]:
    """Return distinct linearly interpolated roots on a fixed grid."""
    roots: list[float] = []
    for index in np.flatnonzero(score_grid[:-1] * score_grid[1:] <= 0.0):
        left = float(theta_grid[index])
        right = float(theta_grid[index + 1])
        left_score = float(score_grid[index])
        right_score = float(score_grid[index + 1])
        if np.isclose(left_score, right_score):
            root = 0.5 * (left + right)
        else:
            root = left - left_score * (right - left) / (
                right_score - left_score
            )
        if not roots or not np.isclose(root, roots[-1], atol=1e-10):
            roots.append(float(root))
    return roots


def root_candidates(
    maxima: np.ndarray,
    folds: list[np.ndarray],
    theta_grid: np.ndarray,
    bandwidth: float,
    config: RootRobustConfig,
) -> tuple[list[dict[str, object]], int]:
    """Evaluate influence values at every empirical score root."""
    all_indices = np.arange(maxima.size)
    observed = [maxima[holdout] for holdout in folds]
    nuisance = [
        maxima[np.setdiff1d(all_indices, holdout, assume_unique=True)]
        for holdout in folds
    ]
    fold_grids = [
        score_grid_for_fold(
            theta_grid,
            observed_fold,
            nuisance_fold,
            bandwidth,
            config.bidders,
        )
        for observed_fold, nuisance_fold in zip(observed, nuisance)
    ]
    weights = np.asarray([fold.size for fold in observed], dtype=float)
    weights /= weights.sum()
    mean_grid = np.average(np.vstack(fold_grids), axis=0, weights=weights)
    roots = enumerate_roots(theta_grid, mean_grid)
    exact_root_count = len(roots)
    if not roots:
        roots = [float(theta_grid[np.argmin(np.abs(mean_grid))])]

    def mean_score(theta: float) -> float:
        return float(
            np.mean(
                np.concatenate(
                    [
                        orthogonal_scores(
                            theta,
                            observed_fold,
                            nuisance_fold,
                            bandwidth,
                            config.bidders,
                        )
                        for observed_fold, nuisance_fold in zip(
                            observed, nuisance
                        )
                    ]
                )
            )
        )

    candidates: list[dict[str, object]] = []
    for root in roots:
        score_blocks = [
            orthogonal_scores(
                root,
                observed_fold,
                nuisance_fold,
                bandwidth,
                config.bidders,
            )
            for observed_fold, nuisance_fold in zip(observed, nuisance)
        ]
        scores = np.concatenate(score_blocks)
        jacobian = (
            mean_score(root + config.jacobian_step)
            - mean_score(root - config.jacobian_step)
        ) / (2.0 * config.jacobian_step)
        influence = -scores / jacobian
        candidates.append(
            {
                "root": root,
                "jacobian": float(jacobian),
                "influence": influence,
            }
        )
    return candidates, exact_root_count


def nearest_candidate(
    candidates: list[dict[str, object]], reference: float
) -> dict[str, object]:
    return min(candidates, key=lambda item: abs(float(item["root"]) - reference))


def chain_statistics(
    fine: dict[str, object],
    medium: dict[str, object],
    coarse: dict[str, object],
    config: RootRobustConfig,
) -> dict[str, float]:
    ratio_squared = config.bandwidth_ratio**2
    denominator = ratio_squared - 1.0
    fine_root = float(fine["root"])
    medium_root = float(medium["root"])
    coarse_root = float(coarse["root"])
    fine_if = np.asarray(fine["influence"], dtype=float)
    medium_if = np.asarray(medium["influence"], dtype=float)
    coarse_if = np.asarray(coarse["influence"], dtype=float)
    richardson = (ratio_squared * fine_root - medium_root) / denominator
    coarse_richardson = (
        ratio_squared * medium_root - coarse_root
    ) / denominator
    richardson_if = (ratio_squared * fine_if - medium_if) / denominator
    coarse_richardson_if = (
        ratio_squared * medium_if - coarse_if
    ) / denominator
    difference = coarse_richardson - richardson
    difference_if = coarse_richardson_if - richardson_if
    standard_error = float(
        richardson_if.std(ddof=1) / np.sqrt(richardson_if.size)
    )
    difference_se = float(
        difference_if.std(ddof=1) / np.sqrt(difference_if.size)
    )
    bias_scale = config.bandwidth_ratio**4 - 1.0
    bias_point = abs(difference) / bias_scale
    bias_upper = (abs(difference) + Z_95 * difference_se) / bias_scale
    return {
        "fine_root": fine_root,
        "medium_root": medium_root,
        "coarse_root": coarse_root,
        "estimate": richardson,
        "standard_error": standard_error,
        "difference": difference,
        "difference_se": difference_se,
        "bias_point": bias_point,
        "bias_upper": bias_upper,
        "wald_lower": richardson - Z_95 * standard_error,
        "wald_upper": richardson + Z_95 * standard_error,
        "bias_lower": richardson - Z_95 * standard_error - bias_upper,
        "bias_upper_endpoint": richardson
        + Z_95 * standard_error
        + bias_upper,
    }


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for lower, upper in sorted(intervals):
        if not merged or lower > merged[-1][1]:
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    return [(lower, upper) for lower, upper in merged]


def contains(intervals: list[tuple[float, float]], value: float) -> bool:
    return any(lower <= value <= upper for lower, upper in intervals)


def record(
    *,
    sample_size: int,
    repetition: int,
    exponent: float,
    method: str,
    bandwidth: float,
    exact_reserve: float,
    population_target: float,
    selected: dict[str, float],
    intervals: list[tuple[float, float]],
    fine_roots: int,
    medium_roots: int,
    coarse_roots: int,
    chain_count: int,
) -> dict[str, object]:
    merged = merge_intervals(intervals)
    set_length = sum(upper - lower for lower, upper in merged)
    hull_lower = min(lower for lower, _ in merged)
    hull_upper = max(upper for _, upper in merged)
    return {
        "auctions": sample_size,
        "repetition": repetition,
        "bandwidth_exponent": exponent,
        "method": method,
        "bandwidth": bandwidth,
        "true_reserve": exact_reserve,
        "population_target": population_target,
        "population_target_bias": population_target - exact_reserve,
        "reserve_estimate": selected["estimate"],
        "reserve_error": selected["estimate"] - exact_reserve,
        "standard_error": selected["standard_error"],
        "ci_lower": hull_lower,
        "ci_upper": hull_upper,
        "confidence_set_length": set_length,
        "confidence_hull_length": hull_upper - hull_lower,
        "confidence_set_components": len(merged),
        "covers_exact_reserve": contains(merged, exact_reserve),
        "covers_population_target": contains(merged, population_target),
        "bias_point": selected["bias_point"],
        "bias_upper_bound": selected["bias_upper"],
        "richardson_difference": selected["difference"],
        "richardson_difference_se": selected["difference_se"],
        "fine_bandwidth_roots": fine_roots,
        "medium_bandwidth_roots": medium_roots,
        "coarse_bandwidth_roots": coarse_roots,
        "chain_count": chain_count,
        "any_multiple_roots": max(fine_roots, medium_roots, coarse_roots) > 1,
        "any_missing_root": min(fine_roots, medium_roots, coarse_roots) == 0,
    }


def run_experiment(config: RootRobustConfig) -> pd.DataFrame:
    baseline = SimulationConfig(bidders=config.bidders)
    distribution = valuation_distribution(baseline)
    exact_reserve = true_reserve(baseline)
    true_maxima = true_maximum_quantiles(
        AblationConfig(
            bidders=config.bidders,
            target_integration_draws=config.target_integration_draws,
        )
    )
    theta_grid = np.linspace(
        config.theta_lower, config.theta_upper, config.theta_grid_size
    )
    ratio = config.bandwidth_ratio
    ratio_squared = ratio**2
    target_lookup: dict[tuple[int, float], tuple[float, float]] = {}
    for sample_size in config.sample_sizes:
        for exponent in config.exponents:
            bandwidth = bandwidth_for(sample_size, exponent, config)
            targets = []
            for multiplier in (1.0, ratio, ratio_squared):
                target, roots = population_target(
                    theta_grid,
                    true_maxima,
                    local_config(
                        sample_size, multiplier * bandwidth, config
                    ),
                    exact_reserve,
                )
                if roots != 1:
                    raise RuntimeError(
                        "Population score must have one root at all bandwidths."
                    )
                targets.append(target)
            corrected_target = (
                ratio_squared * targets[0] - targets[1]
            ) / (ratio_squared - 1.0)
            target_lookup[(sample_size, exponent)] = (
                bandwidth,
                corrected_target,
            )

    records: list[dict[str, object]] = []
    start_time = time.perf_counter()
    for sample_size in config.sample_sizes:
        for repetition in range(1, config.repetitions + 1):
            data_seed = config.seed + sample_size * 100 + repetition - 1
            rng = np.random.default_rng(data_seed)
            values = distribution.rvs(
                size=(sample_size, config.bidders), random_state=rng
            )
            maxima = values.max(axis=1)
            direct_reference = estimate_reserve_from_maxima(
                maxima, config.bidders
            )
            folds = make_folds(
                sample_size, config.folds, data_seed + 31_415_926
            )
            for exponent in config.exponents:
                bandwidth, corrected_target = target_lookup[
                    (sample_size, exponent)
                ]
                fine, fine_count = root_candidates(
                    maxima, folds, theta_grid, bandwidth, config
                )
                medium, medium_count = root_candidates(
                    maxima, folds, theta_grid, ratio * bandwidth, config
                )
                coarse, coarse_count = root_candidates(
                    maxima,
                    folds,
                    theta_grid,
                    ratio_squared * bandwidth,
                    config,
                )

                selected_coarse = nearest_candidate(coarse, direct_reference)
                selected_medium = nearest_candidate(
                    medium, float(selected_coarse["root"])
                )
                selected_fine = nearest_candidate(
                    fine, float(selected_medium["root"])
                )
                selected = chain_statistics(
                    selected_fine, selected_medium, selected_coarse, config
                )

                chains = []
                for fine_candidate in fine:
                    medium_candidate = nearest_candidate(
                        medium, float(fine_candidate["root"])
                    )
                    coarse_candidate = nearest_candidate(
                        coarse, float(medium_candidate["root"])
                    )
                    chains.append(
                        chain_statistics(
                            fine_candidate,
                            medium_candidate,
                            coarse_candidate,
                            config,
                        )
                    )

                common = {
                    "sample_size": sample_size,
                    "repetition": repetition,
                    "exponent": exponent,
                    "bandwidth": bandwidth,
                    "exact_reserve": exact_reserve,
                    "population_target": corrected_target,
                    "selected": selected,
                    "fine_roots": fine_count,
                    "medium_roots": medium_count,
                    "coarse_roots": coarse_count,
                    "chain_count": len(chains),
                }
                records.append(
                    record(
                        method="Richardson Wald",
                        intervals=[
                            (selected["wald_lower"], selected["wald_upper"])
                        ],
                        **common,
                    )
                )
                records.append(
                    record(
                        method="Selected-root bias envelope",
                        intervals=[
                            (
                                selected["bias_lower"],
                                selected["bias_upper_endpoint"],
                            )
                        ],
                        **common,
                    )
                )
                records.append(
                    record(
                        method="Root-robust bias envelope",
                        intervals=[
                            (
                                chain["bias_lower"],
                                chain["bias_upper_endpoint"],
                            )
                            for chain in chains
                        ],
                        **common,
                    )
                )
            if repetition == 1 or repetition % 25 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"n={sample_size:>6,} rep={repetition:>3}/"
                    f"{config.repetitions} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return pd.DataFrame.from_records(records)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(
            ["auctions", "bandwidth_exponent", "method"],
            observed=True,
            sort=True,
        )
        .agg(
            repetitions=("repetition", "count"),
            bandwidth=("bandwidth", "first"),
            population_target_bias=("population_target_bias", "first"),
            bias_exact=("reserve_error", "mean"),
            rmse_exact=(
                "reserve_error", lambda values: np.sqrt(np.mean(values**2))
            ),
            mean_standard_error=("standard_error", "mean"),
            exact_coverage=("covers_exact_reserve", "mean"),
            target_coverage=("covers_population_target", "mean"),
            mean_set_length=("confidence_set_length", "mean"),
            mean_hull_length=("confidence_hull_length", "mean"),
            mean_components=("confidence_set_components", "mean"),
            mean_bias_point=("bias_point", "mean"),
            mean_bias_upper=("bias_upper_bound", "mean"),
            multiple_root_rate=("any_multiple_roots", "mean"),
            missing_root_rate=("any_missing_root", "mean"),
            mean_chain_count=("chain_count", "mean"),
        )
        .reset_index()
    )
    summary["method"] = pd.Categorical(
        summary["method"], categories=METHOD_ORDER, ordered=True
    )
    summary = summary.sort_values(
        ["auctions", "bandwidth_exponent", "method"]
    ).reset_index(drop=True)
    summary["method"] = summary["method"].astype(str)
    return summary


def write_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    config: RootRobustConfig,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TABLE_DIR / "root_robust_inference_raw.csv", index=False)
    summary.to_csv(TABLE_DIR / "root_robust_inference.csv", index=False)
    (TABLE_DIR / "root_robust_inference_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    displayed_source = summary[
        ~(
            (summary["auctions"] == config.reference_sample_size)
            & (summary["bandwidth_exponent"] != config.exponents[0])
        )
    ]
    displayed = displayed_source[
        [
            "auctions",
            "bandwidth_exponent",
            "method",
            "exact_coverage",
            "target_coverage",
            "mean_set_length",
            "mean_bias_upper",
            "multiple_root_rate",
        ]
    ].copy()
    displayed.columns = [
        "$n$",
        r"$\alpha$",
        "Confidence set",
        r"Coverage of $r_0$",
        "Target coverage",
        "Mean length",
        "Mean bias bound",
        "Multi-root",
    ]
    latex = displayed.to_latex(
        index=False,
        float_format=lambda value: f"{value:.4f}",
        escape=False,
        column_format="rrlrrrrr",
    )
    (TABLE_DIR / "root_robust_inference.tex").write_text(
        latex, encoding="utf-8"
    )

    methods = METHOD_ORDER
    colors = ["#4472C4", "#C55A11", "#70AD47"]
    markers = ["o", "s", "D"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x_values = np.arange(len(config.sample_sizes))
    for exponent, linestyle in zip(config.exponents, ["-", "--"]):
        for method, color, marker in zip(methods, colors, markers):
            selected = summary[
                (summary["bandwidth_exponent"] == exponent)
                & (summary["method"] == method)
            ]
            label = rf"$\alpha={exponent:g}$, {method}"
            axes[0].plot(
                x_values,
                selected["exact_coverage"],
                marker=marker,
                linestyle=linestyle,
                lw=1.8,
                color=color,
                label=label,
            )
            axes[1].plot(
                x_values,
                selected["mean_set_length"],
                marker=marker,
                linestyle=linestyle,
                lw=1.8,
                color=color,
                label=label,
            )
    axes[0].axhline(0.95, color="black", linestyle=":", lw=1.2)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel(r"Coverage of exact reserve $r_0$")
    axes[0].set_title("Coverage after bias and root protection")
    axes[1].set_ylabel("Mean confidence-set length")
    axes[1].set_title("Cost of robustification")
    for axis in axes:
        axis.set_xticks(x_values)
        axis.set_xticklabels([f"{size:,}" for size in config.sample_sizes])
        axis.set_xlabel("Number of auctions")
    axes[0].legend(frameon=False, fontsize=6.7, ncol=1)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "root_robust_inference.png", dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--reference-bandwidth", type=float, default=0.15)
    parser.add_argument("--reference-sample-size", type=int, default=500)
    parser.add_argument(
        "--exponents", type=float, nargs="+", default=[0.2, 0.35]
    )
    parser.add_argument("--bandwidth-ratio", type=float, default=math.sqrt(2.0))
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[500, 2_000, 10_000]
    )
    parser.add_argument("--grid-size", type=int, default=161)
    parser.add_argument("--target-draws", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RootRobustConfig(
        sample_sizes=tuple(args.sample_sizes),
        repetitions=args.reps,
        folds=args.folds,
        reference_bandwidth=args.reference_bandwidth,
        reference_sample_size=args.reference_sample_size,
        exponents=tuple(args.exponents),
        bandwidth_ratio=args.bandwidth_ratio,
        theta_grid_size=args.grid_size,
        target_integration_draws=args.target_draws,
        seed=args.seed,
    )
    raw = run_experiment(config)
    summary = summarize(raw)
    write_outputs(raw, summary, config)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
