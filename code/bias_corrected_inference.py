#!/usr/bin/env python3
"""Covariance-aware Richardson correction for exact-reserve inference."""

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
from local_nuisance_ablation import (
    AblationConfig,
    estimate_method,
    true_maximum_quantiles,
)
from orthogonal_wgan_dml import orthogonal_scores


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "paper" / "tables"
FIGURE_DIR = ROOT / "paper" / "figures"


@dataclass(frozen=True)
class BiasCorrectionConfig:
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
    config: BiasCorrectionConfig,
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


def empirical_influence_values(
    estimate: float,
    jacobian: float,
    maxima: np.ndarray,
    folds: list[np.ndarray],
    bandwidth: float,
    config: BiasCorrectionConfig,
) -> np.ndarray:
    all_indices = np.arange(maxima.size)
    blocks = []
    for holdout in folds:
        training = maxima[
            np.setdiff1d(all_indices, holdout, assume_unique=True)
        ]
        blocks.append(
            orthogonal_scores(
                estimate,
                maxima[holdout],
                training,
                bandwidth,
                config.bidders,
            )
        )
    return -np.concatenate(blocks) / jacobian


def interval_record(
    *,
    sample_size: int,
    repetition: int,
    exponent: float,
    method: str,
    bandwidth: float,
    estimate: float,
    standard_error: float,
    exact_reserve: float,
    population_target_value: float,
    small_roots: int,
    large_roots: int,
    root_gap: float,
    theta_lower: float,
    theta_upper: float,
) -> dict[str, float | int | str | bool]:
    ci_lower = estimate - 1.96 * standard_error
    ci_upper = estimate + 1.96 * standard_error
    return {
        "auctions": sample_size,
        "repetition": repetition,
        "bandwidth_exponent": exponent,
        "method": method,
        "bandwidth": bandwidth,
        "true_reserve": exact_reserve,
        "population_target": population_target_value,
        "population_target_bias": population_target_value - exact_reserve,
        "reserve_estimate": estimate,
        "reserve_error": estimate - exact_reserve,
        "target_error": estimate - population_target_value,
        "standard_error": standard_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "covers_exact_reserve": bool(ci_lower <= exact_reserve <= ci_upper),
        "covers_population_target": bool(
            ci_lower <= population_target_value <= ci_upper
        ),
        "small_bandwidth_roots": small_roots,
        "large_bandwidth_roots": large_roots,
        "any_multiple_roots": bool(small_roots > 1 or large_roots > 1),
        "any_missing_root": bool(small_roots == 0 or large_roots == 0),
        "root_gap": root_gap,
        "estimate_outside_grid": bool(
            estimate < theta_lower or estimate > theta_upper
        ),
    }


def run_experiment(config: BiasCorrectionConfig) -> pd.DataFrame:
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
    ratio_squared = config.bandwidth_ratio**2
    target_lookup: dict[
        tuple[int, float], tuple[float, float, float, float]
    ] = {}
    for sample_size in config.sample_sizes:
        for exponent in config.exponents:
            small_bandwidth = bandwidth_for(sample_size, exponent, config)
            large_bandwidth = config.bandwidth_ratio * small_bandwidth
            small_target, small_target_roots = population_target(
                theta_grid,
                true_maxima,
                local_config(sample_size, small_bandwidth, config),
                exact_reserve,
            )
            large_target, large_target_roots = population_target(
                theta_grid,
                true_maxima,
                local_config(sample_size, large_bandwidth, config),
                exact_reserve,
            )
            if small_target_roots != 1 or large_target_roots != 1:
                raise RuntimeError(
                    "Population score must have one root at both bandwidths."
                )
            corrected_target = (
                ratio_squared * small_target - large_target
            ) / (ratio_squared - 1.0)
            target_lookup[(sample_size, exponent)] = (
                small_bandwidth,
                small_target,
                large_target,
                corrected_target,
            )

    records: list[dict[str, float | int | str | bool]] = []
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
                (
                    small_bandwidth,
                    small_target,
                    _,
                    corrected_target,
                ) = target_lookup[(sample_size, exponent)]
                large_bandwidth = config.bandwidth_ratio * small_bandwidth
                large_result = estimate_method(
                    "Empirical-local DML",
                    maxima,
                    folds,
                    true_maxima,
                    theta_grid,
                    direct_reference,
                    local_config(sample_size, large_bandwidth, config),
                )
                large_estimate = float(large_result["reserve_estimate"])
                small_result = estimate_method(
                    "Empirical-local DML",
                    maxima,
                    folds,
                    true_maxima,
                    theta_grid,
                    large_estimate,
                    local_config(sample_size, small_bandwidth, config),
                )
                small_estimate = float(small_result["reserve_estimate"])
                small_influence = empirical_influence_values(
                    small_estimate,
                    float(small_result["score_jacobian"]),
                    maxima,
                    folds,
                    small_bandwidth,
                    config,
                )
                large_influence = empirical_influence_values(
                    large_estimate,
                    float(large_result["score_jacobian"]),
                    maxima,
                    folds,
                    large_bandwidth,
                    config,
                )
                corrected_estimate = (
                    ratio_squared * small_estimate - large_estimate
                ) / (ratio_squared - 1.0)
                corrected_influence = (
                    ratio_squared * small_influence - large_influence
                ) / (ratio_squared - 1.0)
                corrected_se = float(
                    corrected_influence.std(ddof=1)
                    / np.sqrt(corrected_influence.size)
                )
                small_se = float(small_result["standard_error"])
                small_roots = int(small_result["score_roots_on_grid"])
                large_roots = int(large_result["score_roots_on_grid"])
                root_gap = abs(small_estimate - large_estimate)
                exponent_label = f"{exponent:g}"
                records.append(
                    interval_record(
                        sample_size=sample_size,
                        repetition=repetition,
                        exponent=exponent,
                        method=rf"$n^{{-{exponent_label}}}$ DML",
                        bandwidth=small_bandwidth,
                        estimate=small_estimate,
                        standard_error=small_se,
                        exact_reserve=exact_reserve,
                        population_target_value=small_target,
                        small_roots=small_roots,
                        large_roots=large_roots,
                        root_gap=root_gap,
                        theta_lower=config.theta_lower,
                        theta_upper=config.theta_upper,
                    )
                )
                records.append(
                    interval_record(
                        sample_size=sample_size,
                        repetition=repetition,
                        exponent=exponent,
                        method=rf"$n^{{-{exponent_label}}}$ Richardson",
                        bandwidth=small_bandwidth,
                        estimate=corrected_estimate,
                        standard_error=corrected_se,
                        exact_reserve=exact_reserve,
                        population_target_value=corrected_target,
                        small_roots=small_roots,
                        large_roots=large_roots,
                        root_gap=root_gap,
                        theta_lower=config.theta_lower,
                        theta_upper=config.theta_upper,
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
    return (
        raw.groupby(
            ["auctions", "bandwidth_exponent", "method"],
            observed=True,
            sort=True,
        )
        .agg(
            repetitions=("repetition", "count"),
            bandwidth=("bandwidth", "first"),
            population_target_bias=("population_target_bias", "first"),
            mean_estimate=("reserve_estimate", "mean"),
            bias_exact=("reserve_error", "mean"),
            rmse_exact=(
                "reserve_error", lambda values: np.sqrt(np.mean(values**2))
            ),
            mean_standard_error=("standard_error", "mean"),
            exact_coverage=("covers_exact_reserve", "mean"),
            target_coverage=("covers_population_target", "mean"),
            multiple_root_rate=("any_multiple_roots", "mean"),
            missing_root_rate=("any_missing_root", "mean"),
            mean_root_gap=("root_gap", "mean"),
            outside_grid_rate=("estimate_outside_grid", "mean"),
        )
        .reset_index()
    )


def write_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    config: BiasCorrectionConfig,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TABLE_DIR / "bias_corrected_inference_raw.csv", index=False)
    summary.to_csv(TABLE_DIR / "bias_corrected_inference.csv", index=False)
    (TABLE_DIR / "bias_corrected_inference_config.json").write_text(
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
            "method",
            "bandwidth",
            "population_target_bias",
            "rmse_exact",
            "mean_standard_error",
            "exact_coverage",
            "target_coverage",
            "multiple_root_rate",
        ]
    ].copy()
    displayed.columns = [
        "$n$",
        "Estimator",
        "$h_n$",
        r"Target $-r_0$",
        r"RMSE to $r_0$",
        "Mean SE",
        r"Coverage of $r_0$",
        "Target coverage",
        "Multi-root",
    ]
    latex = displayed.to_latex(
        index=False,
        float_format=lambda value: f"{value:.4f}",
        escape=False,
        column_format="rlrrrrrrr",
    )
    (TABLE_DIR / "bias_corrected_inference.tex").write_text(
        latex, encoding="utf-8"
    )

    methods = list(summary["method"].drop_duplicates())
    colors = ["#4472C4", "#70AD47", "#C55A11", "#7030A0"]
    markers = ["o", "s", "^", "D"]
    x_positions = np.arange(len(config.sample_sizes))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for method, color, marker in zip(methods, colors, markers):
        selected = summary[summary["method"] == method]
        axes[0].plot(
            x_positions,
            selected["exact_coverage"],
            marker=marker,
            lw=2,
            color=color,
            label=method,
        )
        axes[1].plot(
            x_positions,
            selected["rmse_exact"],
            marker=marker,
            lw=2,
            color=color,
            label=method,
        )
    axes[0].axhline(0.95, color="black", linestyle="--", lw=1.2)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel(r"Coverage of exact reserve $r_0$")
    axes[0].set_title("Bias-corrected interval coverage")
    axes[1].set_ylabel(r"RMSE relative to $r_0$")
    axes[1].set_title("Cost of bias correction")
    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels([f"{size:,}" for size in config.sample_sizes])
        axis.set_xlabel("Number of auctions")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "bias_corrected_inference.png", dpi=220)
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
    config = BiasCorrectionConfig(
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
