#!/usr/bin/env python3
"""Shrinking-bandwidth paths for empirical-local reserve inference."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from orthogonal_wgan_dml import score_grid_for_fold, select_score_root


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "paper" / "tables"
FIGURE_DIR = ROOT / "paper" / "figures"


@dataclass(frozen=True)
class BandwidthPathConfig:
    bidders: int = 5
    sample_sizes: tuple[int, ...] = (500, 2_000, 10_000)
    repetitions: int = 200
    folds: int = 3
    reference_bandwidth: float = 0.15
    reference_sample_size: int = 500
    exponents: tuple[float, ...] = (0.0, 0.2, 0.35, 0.5)
    theta_lower: float = 0.45
    theta_upper: float = 1.15
    theta_grid_size: int = 141
    jacobian_step: float = 0.0025
    target_integration_draws: int = 100_000
    seed: int = 20_260_716


def bandwidth_for(
    sample_size: int, exponent: float, config: BandwidthPathConfig
) -> float:
    return float(
        config.reference_bandwidth
        * (sample_size / config.reference_sample_size) ** (-exponent)
    )


def ablation_config(
    sample_size: int,
    bandwidth: float,
    config: BandwidthPathConfig,
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


def population_target(
    theta_grid: np.ndarray,
    true_maxima: np.ndarray,
    config: AblationConfig,
    exact_reserve: float,
) -> tuple[float, int]:
    population_score = score_grid_for_fold(
        theta_grid,
        true_maxima,
        true_maxima,
        config.bandwidth,
        config.bidders,
    )
    return select_score_root(theta_grid, population_score, exact_reserve)


def run_paths(config: BandwidthPathConfig) -> pd.DataFrame:
    baseline = SimulationConfig(bidders=config.bidders)
    distribution = valuation_distribution(baseline)
    exact_reserve = true_reserve(baseline)
    common_config = AblationConfig(
        bidders=config.bidders,
        target_integration_draws=config.target_integration_draws,
    )
    true_maxima = true_maximum_quantiles(common_config)
    theta_grid = np.linspace(
        config.theta_lower, config.theta_upper, config.theta_grid_size
    )
    target_lookup: dict[tuple[int, float], tuple[float, int, float]] = {}
    for sample_size in config.sample_sizes:
        for exponent in config.exponents:
            bandwidth = bandwidth_for(sample_size, exponent, config)
            local_config = ablation_config(sample_size, bandwidth, config)
            target, roots = population_target(
                theta_grid, true_maxima, local_config, exact_reserve
            )
            target_lookup[(sample_size, exponent)] = (target, roots, bandwidth)
            if roots != 1:
                raise RuntimeError(
                    f"Population target has {roots} roots for n={sample_size}, "
                    f"alpha={exponent}."
                )

    records: list[dict[str, float | int | bool | str]] = []
    start_time = time.perf_counter()
    for sample_size in config.sample_sizes:
        for repetition in range(config.repetitions):
            data_seed = config.seed + sample_size * 100 + repetition
            rng = np.random.default_rng(data_seed)
            values = distribution.rvs(
                size=(sample_size, config.bidders), random_state=rng
            )
            maxima = values.max(axis=1)
            reference = estimate_reserve_from_maxima(maxima, config.bidders)
            folds = make_folds(sample_size, config.folds, data_seed + 31_415_926)

            for exponent in config.exponents:
                target, _, bandwidth = target_lookup[(sample_size, exponent)]
                local_config = ablation_config(sample_size, bandwidth, config)
                result = estimate_method(
                    "Empirical-local DML",
                    maxima,
                    folds,
                    true_maxima,
                    theta_grid,
                    reference,
                    local_config,
                )
                estimate = float(result["reserve_estimate"])
                standard_error = float(result["standard_error"])
                ci_lower = estimate - 1.96 * standard_error
                ci_upper = estimate + 1.96 * standard_error
                rule = "Fixed" if exponent == 0 else rf"$n^{{-{exponent:g}}}$"
                records.append(
                    {
                        "auctions": sample_size,
                        "repetition": repetition + 1,
                        "bandwidth_rule": rule,
                        "bandwidth_exponent": exponent,
                        "bandwidth": bandwidth,
                        "true_reserve": exact_reserve,
                        "regularized_target": target,
                        "regularization_bias": target - exact_reserve,
                        "reserve_estimate": estimate,
                        "reserve_error": estimate - exact_reserve,
                        "regularized_error": estimate - target,
                        "standard_error": standard_error,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "covers_exact_reserve": bool(
                            ci_lower <= exact_reserve <= ci_upper
                        ),
                        "covers_regularized_target": bool(
                            ci_lower <= target <= ci_upper
                        ),
                        "score_at_estimate": result["score_at_estimate"],
                        "score_jacobian": result["score_jacobian"],
                        "score_roots_on_grid": result["score_roots_on_grid"],
                    }
                )
            if repetition == 0 or (repetition + 1) % 25 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"n={sample_size:>6,} rep={repetition + 1:>3}/"
                    f"{config.repetitions} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return pd.DataFrame.from_records(records)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(
            ["auctions", "bandwidth_exponent", "bandwidth_rule"],
            observed=True,
            sort=True,
        )
        .agg(
            repetitions=("repetition", "count"),
            bandwidth=("bandwidth", "first"),
            regularized_target=("regularized_target", "first"),
            regularization_bias=("regularization_bias", "first"),
            mean_estimate=("reserve_estimate", "mean"),
            bias_exact=("reserve_error", "mean"),
            rmse_exact=("reserve_error", lambda values: np.sqrt(np.mean(values**2))),
            bias_regularized=("regularized_error", "mean"),
            rmse_regularized=(
                "regularized_error", lambda values: np.sqrt(np.mean(values**2))
            ),
            mean_standard_error=("standard_error", "mean"),
            exact_coverage=("covers_exact_reserve", "mean"),
            regularized_coverage=("covers_regularized_target", "mean"),
            no_root_rate=(
                "score_roots_on_grid", lambda values: np.mean(values == 0)
            ),
            multiple_root_rate=(
                "score_roots_on_grid", lambda values: np.mean(values > 1)
            ),
        )
        .reset_index()
    )


def write_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    config: BandwidthPathConfig,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TABLE_DIR / "bandwidth_path_raw.csv", index=False)
    summary.to_csv(TABLE_DIR / "bandwidth_path.csv", index=False)
    (TABLE_DIR / "bandwidth_path_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    displayed = summary[
        [
            "auctions",
            "bandwidth_rule",
            "bandwidth",
            "regularization_bias",
            "rmse_exact",
            "mean_standard_error",
            "exact_coverage",
            "regularized_coverage",
            "multiple_root_rate",
        ]
    ].copy()
    displayed.columns = [
        "$n$",
        "Rule",
        "$h_n$",
        r"$r_{h_n}-r_0$",
        r"RMSE to $r_0$",
        "Mean SE",
        r"Coverage of $r_0$",
        r"Coverage of $r_{h_n}$",
        "Multi-root",
    ]
    latex = displayed.to_latex(
        index=False,
        float_format=lambda value: f"{value:.4f}",
        escape=False,
        column_format="rlrrrrrrr",
    )
    (TABLE_DIR / "bandwidth_path.tex").write_text(latex, encoding="utf-8")

    colors = ["#7F7F7F", "#4472C4", "#70AD47", "#C55A11", "#7030A0"]
    x_positions = np.arange(len(config.sample_sizes))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for color, exponent in zip(colors, config.exponents):
        selected = summary[summary["bandwidth_exponent"] == exponent]
        label = "Fixed $h=0.15$" if exponent == 0 else rf"$h_n\propto n^{{-{exponent:g}}}$"
        axes[0].plot(
            x_positions,
            selected["exact_coverage"],
            marker="o",
            lw=2,
            color=color,
            label=label,
        )
        axes[1].plot(
            x_positions,
            selected["rmse_exact"],
            marker="o",
            lw=2,
            color=color,
            label=label,
        )
    axes[0].axhline(0.95, color="black", linestyle="--", lw=1.2)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel(r"Coverage of exact reserve $r_0$")
    axes[0].set_title("Exact-target interval coverage")
    axes[1].set_ylabel(r"RMSE relative to $r_0$")
    axes[1].set_title("Bias-variance trade-off")
    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels([f"{size:,}" for size in config.sample_sizes])
        axis.set_xlabel("Number of auctions")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "bandwidth_path.png", dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--reference-bandwidth", type=float, default=0.15)
    parser.add_argument("--reference-sample-size", type=int, default=500)
    parser.add_argument(
        "--exponents", type=float, nargs="+", default=[0.0, 0.2, 0.35, 0.5]
    )
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[500, 2_000, 10_000]
    )
    parser.add_argument("--grid-size", type=int, default=141)
    parser.add_argument("--target-draws", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BandwidthPathConfig(
        sample_sizes=tuple(args.sample_sizes),
        repetitions=args.reps,
        folds=args.folds,
        reference_bandwidth=args.reference_bandwidth,
        reference_sample_size=args.reference_sample_size,
        exponents=tuple(args.exponents),
        theta_grid_size=args.grid_size,
        target_integration_draws=args.target_draws,
        seed=args.seed,
    )
    raw = run_paths(config)
    summary = summarize(raw)
    write_outputs(raw, summary, config)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
