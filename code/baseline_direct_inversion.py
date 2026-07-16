#!/usr/bin/env python3
"""Monte Carlo benchmark for reserve-price estimation from auction maxima."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from scipy.stats import lognorm


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "paper" / "tables"
FIGURE_DIR = ROOT / "paper" / "figures"


@dataclass(frozen=True)
class SimulationConfig:
    mu: float = 0.0
    sigma: float = 0.5
    bidders: int = 5
    sample_sizes: tuple[int, ...] = (200, 500, 2_000, 10_000)
    repetitions: int = 500
    seed: int = 20_260_716


def valuation_distribution(config: SimulationConfig):
    return lognorm(s=config.sigma, scale=np.exp(config.mu))


def true_reserve(config: SimulationConfig) -> float:
    distribution = valuation_distribution(config)

    def virtual_value_equation(value: float) -> float:
        return 1.0 - distribution.cdf(value) - value * distribution.pdf(value)

    lower = distribution.ppf(1e-6)
    upper = distribution.ppf(1.0 - 1e-6)
    return float(brentq(virtual_value_equation, lower, upper))


def estimate_reserve_from_maxima(maxima: np.ndarray, bidders: int) -> float:
    """Invert the empirical CDF of maxima and maximize empirical monopoly revenue."""
    ordered = np.sort(np.asarray(maxima, dtype=float))
    # Use the right-continuous empirical CDF and its observed support points.
    empirical_h = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    empirical_f = empirical_h ** (1.0 / bidders)
    objective = ordered * (1.0 - empirical_f)
    return float(ordered[np.argmax(objective)])


def make_revenue_interpolator(config: SimulationConfig):
    """Construct true expected second-price revenue as a function of the reserve."""
    distribution = valuation_distribution(config)
    lower = distribution.ppf(1e-7)
    upper = distribution.ppf(1.0 - 1e-9)
    grid = np.linspace(lower, upper, 200_000)
    cdf = distribution.cdf(grid)
    n = config.bidders
    second_highest_survival = 1.0 - cdf**n - n * (1.0 - cdf) * cdf ** (n - 1)
    tail_integral = -cumulative_trapezoid(
        second_highest_survival[::-1], grid[::-1], initial=0.0
    )[::-1]
    revenue = grid * (1.0 - cdf**n) + tail_integral

    def interpolate(reserve: np.ndarray | float) -> np.ndarray:
        return np.interp(reserve, grid, revenue)

    return interpolate


def simulate(config: SimulationConfig) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rng = np.random.default_rng(config.seed)
    distribution = valuation_distribution(config)
    reserve = true_reserve(config)
    revenue_at = make_revenue_interpolator(config)
    optimal_revenue = float(revenue_at(reserve))
    estimates: dict[int, np.ndarray] = {}
    records: list[dict[str, float | int]] = []

    for sample_size in config.sample_sizes:
        reserve_estimates = np.empty(config.repetitions)
        for repetition in range(config.repetitions):
            values = distribution.rvs(
                size=(sample_size, config.bidders), random_state=rng
            )
            maxima = values.max(axis=1)
            reserve_estimates[repetition] = estimate_reserve_from_maxima(
                maxima, config.bidders
            )

        regrets = np.maximum(
            optimal_revenue - revenue_at(reserve_estimates), 0.0
        )
        errors = reserve_estimates - reserve
        estimates[sample_size] = reserve_estimates
        records.append(
            {
                "auctions": sample_size,
                "repetitions": config.repetitions,
                "bidders": config.bidders,
                "true_reserve": reserve,
                "mean_estimate": reserve_estimates.mean(),
                "bias": errors.mean(),
                "standard_deviation": reserve_estimates.std(ddof=1),
                "rmse": np.sqrt(np.mean(errors**2)),
                "median_absolute_error": np.median(np.abs(errors)),
                "mean_revenue_regret": regrets.mean(),
                "p90_revenue_regret": np.quantile(regrets, 0.9),
            }
        )

    results = pd.DataFrame.from_records(records)
    assert reserve > 0.0
    assert np.isfinite(results.select_dtypes(include=[np.number]).to_numpy()).all()
    return results, estimates


def example_distribution_estimates(
    config: SimulationConfig,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(config.seed + 1)
    distribution = valuation_distribution(config)
    examples: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sample_size in (config.sample_sizes[0], config.sample_sizes[-1]):
        values = distribution.rvs(
            size=(sample_size, config.bidders), random_state=rng
        )
        maxima = np.sort(values.max(axis=1))
        empirical_h = np.arange(1, sample_size + 1) / sample_size
        examples[sample_size] = (maxima, empirical_h ** (1.0 / config.bidders))
    return examples


def write_table(results: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLE_DIR / "baseline_monte_carlo.csv", index=False)

    displayed = results[
        [
            "auctions",
            "mean_estimate",
            "bias",
            "standard_deviation",
            "rmse",
            "mean_revenue_regret",
        ]
    ].copy()
    displayed.columns = [
        "$n$",
        r"Mean $\widehat r$",
        "Bias",
        "SD",
        "RMSE",
        "Mean regret",
    ]
    latex = displayed.to_latex(
        index=False,
        float_format=lambda value: f"{value:.4f}",
        escape=False,
        column_format="rrrrrr",
    )
    (TABLE_DIR / "baseline_monte_carlo.tex").write_text(latex, encoding="utf-8")


def write_figure(
    config: SimulationConfig, estimates: dict[int, np.ndarray]
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    distribution = valuation_distribution(config)
    reserve = true_reserve(config)
    examples = example_distribution_estimates(config)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    grid = np.linspace(distribution.ppf(0.005), distribution.ppf(0.995), 1_000)
    axes[0].plot(grid, distribution.cdf(grid), color="black", lw=2, label="True $F$")
    colors = ["#4472C4", "#C55A11"]
    for color, (sample_size, (maxima, empirical_f)) in zip(colors, examples.items()):
        axes[0].step(
            maxima,
            empirical_f,
            where="post",
            lw=1.4,
            color=color,
            label=fr"Direct inversion, $n={sample_size:,}$",
        )
    axes[0].set_xlim(grid[0], distribution.ppf(0.975))
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("Valuation")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("Latent distribution recovery")
    axes[0].legend(frameon=False, fontsize=9)

    positions = np.arange(len(config.sample_sizes))
    axes[1].boxplot(
        [estimates[size] for size in config.sample_sizes],
        positions=positions,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#D9E2F3", "edgecolor": "#4472C4"},
        medianprops={"color": "#C55A11", "linewidth": 1.5},
        whiskerprops={"color": "#4472C4"},
        capprops={"color": "#4472C4"},
    )
    axes[1].axhline(reserve, color="black", linestyle="--", lw=1.5, label="True reserve")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"{size:,}" for size in config.sample_sizes])
    axes[1].set_xlabel("Number of auctions")
    axes[1].set_ylabel("Estimated reserve")
    axes[1].set_title("Sampling distribution of direct estimator")
    axes[1].legend(frameon=False, fontsize=9)

    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "baseline_direct_inversion.png", dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=500, help="Monte Carlo repetitions")
    parser.add_argument("--seed", type=int, default=20_260_716, help="Random seed")
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[200, 500, 2_000, 10_000],
        help="Auction sample sizes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        repetitions=args.reps,
        seed=args.seed,
        sample_sizes=tuple(args.sample_sizes),
    )
    results, estimates = simulate(config)
    write_table(results)
    write_figure(config, estimates)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.5f}"))


if __name__ == "__main__":
    main()
