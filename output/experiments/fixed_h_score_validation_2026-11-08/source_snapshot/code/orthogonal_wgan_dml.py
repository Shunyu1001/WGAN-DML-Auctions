#!/usr/bin/env python3
"""Cross-fitted orthogonal WGAN-DML pilot for Myerson reserve estimation."""

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
from scipy.optimize import brentq
from scipy.special import ndtr

from baseline_direct_inversion import (
    SimulationConfig,
    make_revenue_interpolator,
    true_reserve,
    valuation_distribution,
)
from crossfit_wgan_pilot import make_folds
from wgan_gp_baseline import (
    WGANConfig,
    choose_device,
    estimate_reserve_from_values,
    sample_maxima,
    sample_values,
    set_seed,
    train_wgan_gp,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "paper" / "tables"
FIGURE_DIR = ROOT / "paper" / "figures"
DEFAULT_CROSSFIT_RAW = TABLE_DIR / "crossfit_wgan_pilot_raw.csv"
INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class DMLConfig:
    bidders: int = 5
    sample_sizes: tuple[int, ...] = (500, 2_000, 10_000)
    repetitions: int = 5
    folds: int = 3
    bandwidth: float = 0.15
    theta_lower: float = 0.45
    theta_upper: float = 1.15
    theta_grid_size: int = 281
    jacobian_step: float = 0.0025
    training_steps: int = 1_200
    critic_steps: int = 5
    batch_size: int = 512
    latent_dim: int = 8
    hidden_dim: int = 64
    learning_rate: float = 1e-4
    gradient_penalty: float = 0.1
    generated_values_per_fold: int = 50_000
    generated_maxima_per_fold: int = 50_000
    target_integration_draws: int = 100_000
    seed: int = 20_260_716
    device: str = "cpu"


def gaussian_kernel_moments(
    theta: np.ndarray | float,
    sample: np.ndarray,
    bandwidth: float,
    block_size: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed CDF and density moments at one or more theta values."""
    points = np.atleast_1d(np.asarray(theta, dtype=float))
    observations = np.asarray(sample, dtype=float).reshape(-1)
    cdf = np.empty(points.size)
    density = np.empty(points.size)
    for start in range(0, points.size, block_size):
        stop = min(start + block_size, points.size)
        z = (points[start:stop, None] - observations[None, :]) / bandwidth
        cdf[start:stop] = ndtr(z).mean(axis=1)
        density[start:stop] = (
            INV_SQRT_2PI * np.exp(-0.5 * z**2) / bandwidth
        ).mean(axis=1)
    return cdf, density


def reserve_moment_and_derivatives(
    theta: np.ndarray | float,
    smoothed_cdf: np.ndarray | float,
    smoothed_density: np.ndarray | float,
    bidders: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regularized reserve moment and derivatives with respect to its nuisances."""
    theta_array = np.asarray(theta, dtype=float)
    cdf = np.clip(np.asarray(smoothed_cdf, dtype=float), 1e-8, 1.0)
    density = np.asarray(smoothed_density, dtype=float)
    inverse_bidders = 1.0 / bidders
    cdf_power = cdf ** (inverse_bidders - 1.0)
    moment = (
        1.0
        - cdf**inverse_bidders
        - theta_array * inverse_bidders * cdf_power * density
    )
    derivative_cdf = (
        -inverse_bidders * cdf_power
        - theta_array
        * inverse_bidders
        * (inverse_bidders - 1.0)
        * cdf ** (inverse_bidders - 2.0)
        * density
    )
    derivative_density = -theta_array * inverse_bidders * cdf_power
    return moment, derivative_cdf, derivative_density


def orthogonal_scores(
    theta: float,
    observed_maxima: np.ndarray,
    generated_maxima_sample: np.ndarray,
    bandwidth: float,
    bidders: int,
) -> np.ndarray:
    """Evaluate the locally robust score for a held-out fold."""
    nuisance_cdf, nuisance_density = gaussian_kernel_moments(
        theta, generated_maxima_sample, bandwidth
    )
    moment, derivative_cdf, derivative_density = reserve_moment_and_derivatives(
        theta, nuisance_cdf[0], nuisance_density[0], bidders
    )
    z = (theta - np.asarray(observed_maxima, dtype=float)) / bandwidth
    observed_cdf_signal = ndtr(z)
    observed_density_signal = INV_SQRT_2PI * np.exp(-0.5 * z**2) / bandwidth
    return (
        float(moment)
        + float(derivative_cdf) * (observed_cdf_signal - nuisance_cdf[0])
        + float(derivative_density)
        * (observed_density_signal - nuisance_density[0])
    )


def score_grid_for_fold(
    theta_grid: np.ndarray,
    observed_maxima: np.ndarray,
    generated_maxima_sample: np.ndarray,
    bandwidth: float,
    bidders: int,
) -> np.ndarray:
    nuisance_cdf, nuisance_density = gaussian_kernel_moments(
        theta_grid, generated_maxima_sample, bandwidth
    )
    observed_cdf, observed_density = gaussian_kernel_moments(
        theta_grid, observed_maxima, bandwidth
    )
    moment, derivative_cdf, derivative_density = reserve_moment_and_derivatives(
        theta_grid, nuisance_cdf, nuisance_density, bidders
    )
    return (
        moment
        + derivative_cdf * (observed_cdf - nuisance_cdf)
        + derivative_density * (observed_density - nuisance_density)
    )


def select_score_root(
    theta_grid: np.ndarray,
    score_grid: np.ndarray,
    reference: float,
) -> tuple[float, int]:
    """Select the score root nearest the fold-averaged plug-in reference."""
    roots: list[float] = []
    for index in np.flatnonzero(score_grid[:-1] * score_grid[1:] <= 0.0):
        left = theta_grid[index]
        right = theta_grid[index + 1]
        left_score = score_grid[index]
        right_score = score_grid[index + 1]
        if np.isclose(left_score, right_score):
            roots.append(float(0.5 * (left + right)))
        else:
            roots.append(
                float(
                    left
                    - left_score * (right - left) / (right_score - left_score)
                )
            )
    if roots:
        selected = min(roots, key=lambda candidate: abs(candidate - reference))
        return selected, len(roots)
    return float(theta_grid[np.argmin(np.abs(score_grid))]), 0


def true_regularized_target(config: DMLConfig) -> float:
    """Compute the fixed-bandwidth target by deterministic quantile integration."""
    distribution = valuation_distribution(SimulationConfig(bidders=config.bidders))
    probabilities = (
        np.arange(config.target_integration_draws, dtype=float) + 0.5
    ) / config.target_integration_draws
    maximum_quantiles = distribution.ppf(probabilities ** (1.0 / config.bidders))

    def equation(theta: float) -> float:
        cdf, density = gaussian_kernel_moments(
            theta, maximum_quantiles, config.bandwidth
        )
        moment, _, _ = reserve_moment_and_derivatives(
            theta, cdf[0], density[0], config.bidders
        )
        return float(moment)

    return float(brentq(equation, config.theta_lower, config.theta_upper))


def load_comparison_rows(path: Path, config: DMLConfig) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cross-fit results not found at {path}. Run code/crossfit_wgan_pilot.py first."
        )
    raw = pd.read_csv(path)
    selected = raw[
        raw["auctions"].isin(config.sample_sizes)
        & raw["repetition"].between(1, config.repetitions)
        & raw["method"].isin(
            ["Direct inversion", "WGAN-GP", "Fold-averaged WGAN"]
        )
    ].copy()
    expected = len(config.sample_sizes) * config.repetitions * 3
    if len(selected) != expected:
        raise ValueError("Comparison rows do not match the requested pilot design.")
    return selected


def run_dml(
    config: DMLConfig,
    crossfit_raw: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    baseline = SimulationConfig(bidders=config.bidders)
    distribution = valuation_distribution(baseline)
    exact_reserve = true_reserve(baseline)
    regularized_reserve = true_regularized_target(config)
    revenue_at = make_revenue_interpolator(baseline)
    optimal_revenue = float(revenue_at(exact_reserve))
    comparison_rows = load_comparison_rows(crossfit_raw, config)
    device = choose_device(config.device)
    wgan_config = WGANConfig(
        bidders=config.bidders,
        sample_sizes=config.sample_sizes,
        repetitions=config.repetitions,
        training_steps=config.training_steps,
        critic_steps=config.critic_steps,
        batch_size=config.batch_size,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        gradient_penalty=config.gradient_penalty,
        generated_values=config.generated_values_per_fold,
        diagnostic_auctions=config.generated_maxima_per_fold,
        seed=config.seed,
        device=config.device,
    )
    theta_grid = np.linspace(
        config.theta_lower, config.theta_upper, config.theta_grid_size
    )
    records: list[dict[str, float | int | str | bool]] = []
    fold_records: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

    for sample_size in config.sample_sizes:
        for repetition in range(config.repetitions):
            data_seed = config.seed + sample_size * 100 + repetition
            rng = np.random.default_rng(data_seed)
            values = distribution.rvs(
                size=(sample_size, config.bidders), random_state=rng
            )
            maxima = values.max(axis=1)
            folds = make_folds(sample_size, config.folds, data_seed + 31_415_926)
            all_indices = np.arange(sample_size)
            generated_by_fold: list[np.ndarray] = []
            fold_score_grids: list[np.ndarray] = []
            fold_reserves: list[float] = []

            for fold_index, holdout in enumerate(folds):
                training = np.setdiff1d(all_indices, holdout, assume_unique=True)
                fold_seed = data_seed + 1_000_000 + (fold_index + 1) * 10_000
                generator, diagnostics = train_wgan_gp(
                    maxima[training], wgan_config, fold_seed
                )
                set_seed(fold_seed + 101)
                generated_values = sample_values(
                    generator, config.generated_values_per_fold, device
                )
                generated_maximum_sample = sample_maxima(
                    generator,
                    config.generated_maxima_per_fold,
                    config.bidders,
                    device,
                )
                fold_reserve = estimate_reserve_from_values(generated_values)
                fold_score_grid = score_grid_for_fold(
                    theta_grid,
                    maxima[holdout],
                    generated_maximum_sample,
                    config.bandwidth,
                    config.bidders,
                )
                generated_by_fold.append(generated_maximum_sample)
                fold_score_grids.append(fold_score_grid)
                fold_reserves.append(fold_reserve)
                fold_records.append(
                    {
                        "auctions": sample_size,
                        "repetition": repetition + 1,
                        "fold": fold_index + 1,
                        "training_auctions": training.size,
                        "heldout_auctions": holdout.size,
                        "fold_plugin_reserve": fold_reserve,
                        "training_checkpoint_w1": float(
                            diagnostics["best_max_bid_w1"]
                        ),
                    }
                )

            fold_average = float(np.mean(fold_reserves))
            recorded_fold_average = comparison_rows[
                (comparison_rows["auctions"] == sample_size)
                & (comparison_rows["repetition"] == repetition + 1)
                & (comparison_rows["method"] == "Fold-averaged WGAN")
            ]["reserve_estimate"].iloc[0]
            if not np.isclose(fold_average, recorded_fold_average, atol=1e-12):
                raise RuntimeError("The DML nuisance fits no longer reproduce the fold pilot.")

            fold_weights = np.array([len(fold) for fold in folds], dtype=float)
            fold_weights /= fold_weights.sum()
            mean_score_grid = np.average(
                np.vstack(fold_score_grids), axis=0, weights=fold_weights
            )
            estimate, root_count = select_score_root(
                theta_grid, mean_score_grid, fold_average
            )

            score_blocks = [
                orthogonal_scores(
                    estimate,
                    maxima[holdout],
                    generated_by_fold[fold_index],
                    config.bandwidth,
                    config.bidders,
                )
                for fold_index, holdout in enumerate(folds)
            ]
            scores = np.concatenate(score_blocks)

            def mean_score(theta: float) -> float:
                return float(
                    np.mean(
                        np.concatenate(
                            [
                                orthogonal_scores(
                                    theta,
                                    maxima[holdout],
                                    generated_by_fold[fold_index],
                                    config.bandwidth,
                                    config.bidders,
                                )
                                for fold_index, holdout in enumerate(folds)
                            ]
                        )
                    )
                )

            jacobian = (
                mean_score(estimate + config.jacobian_step)
                - mean_score(estimate - config.jacobian_step)
            ) / (2.0 * config.jacobian_step)
            standard_error = float(
                scores.std(ddof=1) / (np.sqrt(sample_size) * abs(jacobian))
            )
            ci_lower = estimate - 1.96 * standard_error
            ci_upper = estimate + 1.96 * standard_error
            regret = max(optimal_revenue - float(revenue_at(estimate)), 0.0)
            records.append(
                {
                    "auctions": sample_size,
                    "repetition": repetition + 1,
                    "bidders": config.bidders,
                    "method": "Orthogonal WGAN-DML",
                    "true_reserve": exact_reserve,
                    "regularized_target": regularized_reserve,
                    "reserve_estimate": estimate,
                    "reserve_error": estimate - exact_reserve,
                    "regularized_error": estimate - regularized_reserve,
                    "absolute_reserve_error": abs(estimate - exact_reserve),
                    "revenue_regret": regret,
                    "standard_error": standard_error,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "covers_regularized_target": bool(
                        ci_lower <= regularized_reserve <= ci_upper
                    ),
                    "covers_exact_reserve": bool(
                        ci_lower <= exact_reserve <= ci_upper
                    ),
                    "score_at_estimate": float(scores.mean()),
                    "score_jacobian": jacobian,
                    "score_roots_on_grid": root_count,
                    "fold_average_reference": fold_average,
                    "orthogonal_correction": estimate - fold_average,
                }
            )
            elapsed = time.perf_counter() - start_time
            print(
                f"n={sample_size:>6,} rep={repetition + 1:>2}/"
                f"{config.repetitions} fold={fold_average:.4f} "
                f"dml={estimate:.4f} se={standard_error:.4f} "
                f"roots={root_count} elapsed={elapsed:.1f}s",
                flush=True,
            )

    dml_raw = pd.DataFrame.from_records(records)
    comparison = comparison_rows.copy()
    comparison["regularized_target"] = regularized_reserve
    comparison["regularized_error"] = (
        comparison["reserve_estimate"] - regularized_reserve
    )
    for column in [
        "standard_error",
        "ci_lower",
        "ci_upper",
        "covers_regularized_target",
        "covers_exact_reserve",
        "score_at_estimate",
        "score_jacobian",
        "score_roots_on_grid",
        "fold_average_reference",
        "orthogonal_correction",
    ]:
        comparison[column] = np.nan
    combined = pd.concat([comparison, dml_raw], ignore_index=True, sort=False)
    combined["method"] = pd.Categorical(
        combined["method"],
        categories=[
            "Direct inversion",
            "WGAN-GP",
            "Fold-averaged WGAN",
            "Orthogonal WGAN-DML",
        ],
        ordered=True,
    )
    combined = combined.sort_values(
        ["auctions", "method", "repetition"]
    ).reset_index(drop=True)
    return combined, pd.DataFrame.from_records(fold_records), regularized_reserve


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["auctions", "method"], observed=True, sort=True)
        .agg(
            repetitions=("repetition", "count"),
            true_reserve=("true_reserve", "first"),
            regularized_target=("regularized_target", "first"),
            mean_estimate=("reserve_estimate", "mean"),
            bias_exact=("reserve_error", "mean"),
            rmse_exact=("reserve_error", lambda x: np.sqrt(np.mean(x**2))),
            bias_regularized=("regularized_error", "mean"),
            rmse_regularized=(
                "regularized_error", lambda x: np.sqrt(np.mean(x**2))
            ),
            mean_revenue_regret=("revenue_regret", "mean"),
            mean_standard_error=("standard_error", "mean"),
            regularized_coverage=("covers_regularized_target", "mean"),
        )
        .reset_index()
    )


def write_outputs(
    raw: pd.DataFrame,
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    config: DMLConfig,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TABLE_DIR / "orthogonal_wgan_dml_raw.csv", index=False)
    folds.to_csv(TABLE_DIR / "orthogonal_wgan_dml_folds.csv", index=False)
    summary.to_csv(TABLE_DIR / "orthogonal_wgan_dml.csv", index=False)
    (TABLE_DIR / "orthogonal_wgan_dml_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    displayed = summary[
        [
            "auctions",
            "method",
            "mean_estimate",
            "bias_exact",
            "rmse_exact",
            "rmse_regularized",
            "mean_standard_error",
            "regularized_coverage",
        ]
    ].copy()
    displayed.columns = [
        "$n$",
        "Estimator",
        r"Mean $\widehat r$",
        r"Bias to $r_0$",
        r"RMSE to $r_0$",
        r"RMSE to $r_h$",
        "Mean SE",
        r"Coverage of $r_h$",
    ]
    displayed["Estimator"] = displayed["Estimator"].astype("object").replace(
        {
            "Direct inversion": "Direct",
            "WGAN-GP": "Full WGAN",
            "Fold-averaged WGAN": "Fold average",
            "Orthogonal WGAN-DML": "Orthogonal DML",
        }
    )
    latex = displayed.to_latex(
        index=False,
        float_format=lambda value: "--" if pd.isna(value) else f"{value:.4f}",
        escape=False,
        column_format="rlrrrrrr",
        na_rep="--",
    )
    (TABLE_DIR / "orthogonal_wgan_dml.tex").write_text(latex, encoding="utf-8")

    methods = ["Fold-averaged WGAN", "Orthogonal WGAN-DML"]
    colors = {"Fold-averaged WGAN": "#70AD47", "Orthogonal WGAN-DML": "#C55A11"}
    labels = {"Fold-averaged WGAN": "Fold average", "Orthogonal WGAN-DML": "Orthogonal DML"}
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x_positions = np.arange(len(config.sample_sizes))
    for method in methods:
        selected = summary[summary["method"] == method]
        axes[0].plot(
            x_positions,
            selected["rmse_exact"],
            marker="o",
            lw=2,
            color=colors[method],
            label=labels[method],
        )
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels([f"{size:,}" for size in config.sample_sizes])
    axes[0].set_xlabel("Number of auctions")
    axes[0].set_ylabel(r"RMSE relative to exact reserve $r_0$")
    axes[0].set_title("Plug-in averaging and orthogonal correction")
    axes[0].legend(frameon=False)

    dml = raw[raw["method"] == "Orthogonal WGAN-DML"].copy()
    jitter = np.linspace(-0.13, 0.13, config.repetitions)
    for x_position, sample_size in zip(x_positions, config.sample_sizes):
        selected = dml[dml["auctions"] == sample_size].sort_values("repetition")
        axes[1].errorbar(
            x_position + jitter[: len(selected)],
            selected["reserve_estimate"],
            yerr=1.96 * selected["standard_error"],
            fmt="o",
            ms=4,
            color=colors["Orthogonal WGAN-DML"],
            ecolor="#ED7D31",
            elinewidth=1,
            capsize=2,
        )
    exact_target = float(dml["true_reserve"].iloc[0])
    regularized_target = float(dml["regularized_target"].iloc[0])
    axes[1].axhline(
        exact_target, color="black", linestyle="--", lw=1.4, label=r"Exact $r_0$"
    )
    axes[1].axhline(
        regularized_target,
        color="#4472C4",
        linestyle=":",
        lw=1.8,
        label=r"Regularized $r_h$",
    )
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([f"{size:,}" for size in config.sample_sizes])
    axes[1].set_xlabel("Number of auctions")
    axes[1].set_ylabel("Reserve estimate and 95% interval")
    axes[1].set_title("Cross-fitted orthogonal-score inference")
    axes[1].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "orthogonal_wgan_dml.png", dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bandwidth", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gp-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu"
    )
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[500, 2_000, 10_000]
    )
    parser.add_argument("--generated-values", type=int, default=50_000)
    parser.add_argument("--generated-maxima", type=int, default=50_000)
    parser.add_argument("--target-draws", type=int, default=100_000)
    parser.add_argument("--crossfit-raw", type=Path, default=DEFAULT_CROSSFIT_RAW)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DMLConfig(
        sample_sizes=tuple(args.sample_sizes),
        repetitions=args.reps,
        folds=args.folds,
        bandwidth=args.bandwidth,
        training_steps=args.steps,
        critic_steps=args.critic_steps,
        batch_size=args.batch_size,
        gradient_penalty=args.gp_weight,
        generated_values_per_fold=args.generated_values,
        generated_maxima_per_fold=args.generated_maxima,
        target_integration_draws=args.target_draws,
        seed=args.seed,
        device=args.device,
    )
    raw, folds, regularized_target = run_dml(config, args.crossfit_raw)
    summary = summarize(raw)
    write_outputs(raw, folds, summary, config)
    print(f"Regularized target: {regularized_target:.6f}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
