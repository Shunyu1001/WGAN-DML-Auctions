#!/usr/bin/env python3
"""Oracle and empirical-local ablations for the reserve-price score."""

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
    make_revenue_interpolator,
    true_reserve,
    valuation_distribution,
)
from crossfit_wgan_pilot import make_folds
from orthogonal_wgan_dml import (
    gaussian_kernel_moments,
    orthogonal_scores,
    score_grid_for_fold,
    select_score_root,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "paper" / "tables"
FIGURE_DIR = ROOT / "paper" / "figures"
DEFAULT_WGAN_DML_RAW = TABLE_DIR / "orthogonal_wgan_dml_raw.csv"


@dataclass(frozen=True)
class AblationConfig:
    bidders: int = 5
    sample_sizes: tuple[int, ...] = (500, 2_000, 10_000)
    repetitions: int = 200
    folds: int = 3
    bandwidth: float = 0.15
    theta_lower: float = 0.45
    theta_upper: float = 1.15
    theta_grid_size: int = 141
    jacobian_step: float = 0.0025
    target_integration_draws: int = 100_000
    seed: int = 20_260_716


def true_maximum_quantiles(config: AblationConfig) -> np.ndarray:
    distribution = valuation_distribution(SimulationConfig(bidders=config.bidders))
    probabilities = (
        np.arange(config.target_integration_draws, dtype=float) + 0.5
    ) / config.target_integration_draws
    return distribution.ppf(probabilities ** (1.0 / config.bidders))


def score_inference(
    estimate: float,
    observed_by_fold: list[np.ndarray],
    nuisance_by_fold: list[np.ndarray],
    config: AblationConfig,
) -> tuple[float, float, float, float]:
    score_blocks = [
        orthogonal_scores(
            estimate,
            observed,
            nuisance,
            config.bandwidth,
            config.bidders,
        )
        for observed, nuisance in zip(observed_by_fold, nuisance_by_fold)
    ]
    scores = np.concatenate(score_blocks)

    def mean_score(theta: float) -> float:
        return float(
            np.mean(
                np.concatenate(
                    [
                        orthogonal_scores(
                            theta,
                            observed,
                            nuisance,
                            config.bandwidth,
                            config.bidders,
                        )
                        for observed, nuisance in zip(
                            observed_by_fold, nuisance_by_fold
                        )
                    ]
                )
            )
        )

    jacobian = (
        mean_score(estimate + config.jacobian_step)
        - mean_score(estimate - config.jacobian_step)
    ) / (2.0 * config.jacobian_step)
    standard_error = float(
        scores.std(ddof=1) / (np.sqrt(scores.size) * abs(jacobian))
    )
    return float(scores.mean()), float(jacobian), standard_error, scores.size


def estimate_method(
    method: str,
    maxima: np.ndarray,
    folds: list[np.ndarray],
    true_maxima: np.ndarray,
    theta_grid: np.ndarray,
    reference: float,
    config: AblationConfig,
) -> dict[str, float | int]:
    all_indices = np.arange(maxima.size)
    if method == "Oracle-local score":
        observed_by_fold = [maxima]
        nuisance_by_fold = [true_maxima]
    elif method == "Empirical-local DML":
        observed_by_fold = [maxima[holdout] for holdout in folds]
        nuisance_by_fold = [
            maxima[np.setdiff1d(all_indices, holdout, assume_unique=True)]
            for holdout in folds
        ]
    else:
        raise ValueError(f"Unknown method: {method}")

    fold_score_grids = [
        score_grid_for_fold(
            theta_grid,
            observed,
            nuisance,
            config.bandwidth,
            config.bidders,
        )
        for observed, nuisance in zip(observed_by_fold, nuisance_by_fold)
    ]
    fold_weights = np.array([sample.size for sample in observed_by_fold], dtype=float)
    fold_weights /= fold_weights.sum()
    mean_score_grid = np.average(
        np.vstack(fold_score_grids), axis=0, weights=fold_weights
    )
    estimate, root_count = select_score_root(theta_grid, mean_score_grid, reference)
    score_mean, jacobian, standard_error, observations = score_inference(
        estimate, observed_by_fold, nuisance_by_fold, config
    )
    return {
        "reserve_estimate": estimate,
        "score_at_estimate": score_mean,
        "score_jacobian": jacobian,
        "standard_error": standard_error,
        "score_roots_on_grid": root_count,
        "score_observations": observations,
    }


def run_ablation(
    config: AblationConfig,
) -> tuple[pd.DataFrame, float, float, float]:
    baseline = SimulationConfig(bidders=config.bidders)
    distribution = valuation_distribution(baseline)
    exact_reserve = true_reserve(baseline)
    revenue_at = make_revenue_interpolator(baseline)
    optimal_revenue = float(revenue_at(exact_reserve))
    true_maxima = true_maximum_quantiles(config)
    theta_grid = np.linspace(
        config.theta_lower, config.theta_upper, config.theta_grid_size
    )
    true_cdf_grid, true_density_grid = gaussian_kernel_moments(
        theta_grid, true_maxima, config.bandwidth
    )
    true_score_grid = score_grid_for_fold(
        theta_grid,
        true_maxima,
        true_maxima,
        config.bandwidth,
        config.bidders,
    )
    regularized_target, target_root_count = select_score_root(
        theta_grid, true_score_grid, exact_reserve
    )
    if target_root_count != 1:
        raise RuntimeError("The population regularized moment must have one grid root.")
    target_cdf = float(np.interp(regularized_target, theta_grid, true_cdf_grid))
    target_density = float(
        np.interp(regularized_target, theta_grid, true_density_grid)
    )
    records: list[dict[str, float | int | str | bool]] = []
    start_time = time.perf_counter()

    for sample_size in config.sample_sizes:
        for repetition in range(config.repetitions):
            data_seed = config.seed + sample_size * 100 + repetition
            rng = np.random.default_rng(data_seed)
            values = distribution.rvs(
                size=(sample_size, config.bidders), random_state=rng
            )
            maxima = values.max(axis=1)
            direct_reserve = estimate_reserve_from_maxima(maxima, config.bidders)
            folds = make_folds(sample_size, config.folds, data_seed + 31_415_926)

            for method in ["Empirical-local DML", "Oracle-local score"]:
                result = estimate_method(
                    method,
                    maxima,
                    folds,
                    true_maxima,
                    theta_grid,
                    direct_reserve,
                    config,
                )
                estimate = float(result["reserve_estimate"])
                standard_error = float(result["standard_error"])
                ci_lower = estimate - 1.96 * standard_error
                ci_upper = estimate + 1.96 * standard_error
                regret = max(optimal_revenue - float(revenue_at(estimate)), 0.0)
                if method == "Empirical-local DML":
                    all_indices = np.arange(sample_size)
                    nuisance_samples = [
                        maxima[
                            np.setdiff1d(all_indices, holdout, assume_unique=True)
                        ]
                        for holdout in folds
                    ]
                    nuisance_moments = [
                        gaussian_kernel_moments(
                            regularized_target, sample, config.bandwidth
                        )
                        for sample in nuisance_samples
                    ]
                    nuisance_cdf_error = float(
                        np.mean([moments[0][0] for moments in nuisance_moments])
                        - target_cdf
                    )
                    nuisance_density_error = float(
                        np.mean([moments[1][0] for moments in nuisance_moments])
                        - target_density
                    )
                else:
                    nuisance_cdf_error = 0.0
                    nuisance_density_error = 0.0
                records.append(
                    {
                        "auctions": sample_size,
                        "repetition": repetition + 1,
                        "bidders": config.bidders,
                        "method": method,
                        "true_reserve": exact_reserve,
                        "regularized_target": regularized_target,
                        "reserve_estimate": estimate,
                        "reserve_error": estimate - exact_reserve,
                        "regularized_error": estimate - regularized_target,
                        "revenue_regret": regret,
                        "standard_error": standard_error,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "covers_regularized_target": bool(
                            ci_lower <= regularized_target <= ci_upper
                        ),
                        "covers_exact_reserve": bool(
                            ci_lower <= exact_reserve <= ci_upper
                        ),
                        "score_at_estimate": result["score_at_estimate"],
                        "score_jacobian": result["score_jacobian"],
                        "score_roots_on_grid": result["score_roots_on_grid"],
                        "nuisance_cdf_error_at_target": nuisance_cdf_error,
                        "nuisance_density_error_at_target": nuisance_density_error,
                        "maxima_below_exact_reserve": int(
                            np.sum(maxima <= exact_reserve)
                        ),
                        "maxima_below_regularized_target": int(
                            np.sum(maxima <= regularized_target)
                        ),
                    }
                )
            if repetition == 0 or (repetition + 1) % 25 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"n={sample_size:>6,} rep={repetition + 1:>3}/"
                    f"{config.repetitions} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    return (
        pd.DataFrame.from_records(records),
        regularized_target,
        target_cdf,
        target_density,
    )


def load_wgan_rows(path: Path, config: AblationConfig) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"WGAN-DML results not found at {path}. Run orthogonal_wgan_dml.py first."
        )
    raw = pd.read_csv(path)
    selected = raw[
        raw["auctions"].isin(config.sample_sizes)
        & (raw["method"] == "Orthogonal WGAN-DML")
    ].copy()
    expected = len(config.sample_sizes) * 5
    if len(selected) != expected:
        raise ValueError("The stored WGAN-DML pilot must contain five seeds per n.")
    selected["nuisance_cdf_error_at_target"] = np.nan
    selected["nuisance_density_error_at_target"] = np.nan
    selected["maxima_below_exact_reserve"] = np.nan
    selected["maxima_below_regularized_target"] = np.nan
    return selected


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["auctions", "method"], observed=True, sort=True)
        .agg(
            repetitions=("repetition", "count"),
            mean_estimate=("reserve_estimate", "mean"),
            bias_regularized=("regularized_error", "mean"),
            rmse_regularized=(
                "regularized_error", lambda values: np.sqrt(np.mean(values**2))
            ),
            mean_standard_error=("standard_error", "mean"),
            regularized_coverage=("covers_regularized_target", "mean"),
            no_root_rate=(
                "score_roots_on_grid", lambda values: np.mean(values == 0)
            ),
            mean_nuisance_cdf_error=("nuisance_cdf_error_at_target", "mean"),
            mean_nuisance_density_error=(
                "nuisance_density_error_at_target", "mean"
            ),
            mean_maxima_below_exact=("maxima_below_exact_reserve", "mean"),
        )
        .reset_index()
    )


def write_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    config: AblationConfig,
    regularized_target: float,
    target_cdf: float,
    target_density: float,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TABLE_DIR / "local_nuisance_ablation_raw.csv", index=False)
    summary.to_csv(TABLE_DIR / "local_nuisance_ablation.csv", index=False)
    config_payload = {
        **asdict(config),
        "regularized_target": regularized_target,
        "target_smoothed_maximum_cdf": target_cdf,
        "target_smoothed_maximum_density": target_density,
    }
    (TABLE_DIR / "local_nuisance_ablation_config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    displayed = summary[
        [
            "auctions",
            "method",
            "repetitions",
            "mean_estimate",
            "bias_regularized",
            "rmse_regularized",
            "mean_standard_error",
            "regularized_coverage",
        ]
    ].copy()
    displayed.columns = [
        "$n$",
        "Score nuisance",
        "Reps",
        r"Mean $\widehat r_h$",
        r"Bias to $r_h$",
        r"RMSE to $r_h$",
        "Mean SE",
        "Coverage",
    ]
    displayed["Score nuisance"] = displayed["Score nuisance"].astype("object").replace(
        {
            "Orthogonal WGAN-DML": "WGAN generated",
            "Empirical-local DML": "Training-fold empirical",
            "Oracle-local score": "Oracle",
        }
    )
    latex = displayed.to_latex(
        index=False,
        float_format=lambda value: f"{value:.4f}",
        escape=False,
        column_format="rlrrrrrr",
    )
    (TABLE_DIR / "local_nuisance_ablation.tex").write_text(
        latex, encoding="utf-8"
    )

    methods = [
        "Orthogonal WGAN-DML",
        "Empirical-local DML",
        "Oracle-local score",
    ]
    colors = {
        "Orthogonal WGAN-DML": "#C55A11",
        "Empirical-local DML": "#4472C4",
        "Oracle-local score": "#70AD47",
    }
    labels = {
        "Orthogonal WGAN-DML": "WGAN nuisance (5 reps)",
        "Empirical-local DML": "Training-fold empirical",
        "Oracle-local score": "Oracle nuisance",
    }
    x_positions = np.arange(len(config.sample_sizes))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for method in methods:
        selected = summary[summary["method"] == method]
        axes[0].plot(
            x_positions,
            selected["regularized_coverage"],
            marker="o",
            lw=2,
            color=colors[method],
            label=labels[method],
        )
        axes[1].plot(
            x_positions,
            selected["rmse_regularized"],
            marker="o",
            lw=2,
            color=colors[method],
            label=labels[method],
        )
    axes[0].axhline(0.95, color="black", linestyle="--", lw=1.2)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel(r"Coverage of $r_h$")
    axes[0].set_title("Nominal 95% interval coverage")
    axes[1].set_ylabel(r"RMSE relative to $r_h$")
    axes[1].set_title("Regularized-target estimation error")
    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels([f"{size:,}" for size in config.sample_sizes])
        axis.set_xlabel("Number of auctions")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "local_nuisance_ablation.png", dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bandwidth", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[500, 2_000, 10_000]
    )
    parser.add_argument("--grid-size", type=int, default=141)
    parser.add_argument("--target-draws", type=int, default=100_000)
    parser.add_argument("--wgan-raw", type=Path, default=DEFAULT_WGAN_DML_RAW)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AblationConfig(
        sample_sizes=tuple(args.sample_sizes),
        repetitions=args.reps,
        folds=args.folds,
        bandwidth=args.bandwidth,
        theta_grid_size=args.grid_size,
        target_integration_draws=args.target_draws,
        seed=args.seed,
    )
    ablation_raw, target, target_cdf, target_density = run_ablation(config)
    wgan_rows = load_wgan_rows(args.wgan_raw, config)
    combined = pd.concat([wgan_rows, ablation_raw], ignore_index=True, sort=False)
    method_order = [
        "Orthogonal WGAN-DML",
        "Empirical-local DML",
        "Oracle-local score",
    ]
    combined["method"] = pd.Categorical(
        combined["method"], categories=method_order, ordered=True
    )
    combined = combined.sort_values(
        ["auctions", "method", "repetition"]
    ).reset_index(drop=True)
    summary = summarize(combined)
    write_outputs(
        combined, summary, config, target, target_cdf, target_density
    )
    print(f"Regularized target: {target:.6f}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
