#!/usr/bin/env python3
"""External-validity Monte Carlo for fixed-bandwidth reserve inference."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import lognorm

from baseline_direct_inversion import estimate_reserve_from_maxima
from crossfit_wgan_pilot import make_folds
from local_nuisance_ablation import AblationConfig, estimate_method
from orthogonal_wgan_dml import score_grid_for_fold, select_score_root


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "paper" / "tables"


@dataclass(frozen=True)
class ExternalValidityConfig:
    bidders: tuple[int, ...] = (3, 5, 10)
    sample_sizes: tuple[int, ...] = (2_000, 10_000)
    repetitions: int = 200
    folds: int = 3
    bandwidth: float = 0.15
    theta_grid_size: int = 121
    jacobian_step: float = 0.0025
    target_integration_draws: int = 100_000
    seed: int = 20_260_717


class MixtureLognormal:
    """Two-component lognormal mixture with deterministic interpolated quantiles."""

    def __init__(
        self,
        weight: float,
        mu_1: float,
        sigma_1: float,
        mu_2: float,
        sigma_2: float,
    ) -> None:
        self.weight = weight
        self.first = lognorm(s=sigma_1, scale=np.exp(mu_1))
        self.second = lognorm(s=sigma_2, scale=np.exp(mu_2))
        lower = min(self.first.ppf(1e-8), self.second.ppf(1e-8))
        upper = max(self.first.ppf(1.0 - 1e-8), self.second.ppf(1.0 - 1e-8))
        self._grid = np.geomspace(lower, upper, 300_000)
        self._cdf_grid = self.cdf(self._grid)

    def cdf(self, values: np.ndarray | float) -> np.ndarray:
        return (
            self.weight * self.first.cdf(values)
            + (1.0 - self.weight) * self.second.cdf(values)
        )

    def ppf(self, probabilities: np.ndarray | float) -> np.ndarray:
        return np.interp(probabilities, self._cdf_grid, self._grid)

    def rvs(self, size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        choose_first = rng.random(size) < self.weight
        first_draws = self.first.rvs(size=size, random_state=rng)
        second_draws = self.second.rvs(size=size, random_state=rng)
        return np.where(choose_first, first_draws, second_draws)


def distributions() -> dict[str, object]:
    return {
        "Baseline": lognorm(s=0.5, scale=np.exp(0.0)),
        "Heavy tail": lognorm(s=0.8, scale=np.exp(0.0)),
        "Mixture": MixtureLognormal(
            weight=0.7,
            mu_1=-0.15,
            sigma_1=0.35,
            mu_2=0.55,
            sigma_2=0.25,
        ),
    }


def sample_values(
    distribution: object,
    size: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    if isinstance(distribution, MixtureLognormal):
        return distribution.rvs(size, rng)
    return distribution.rvs(size=size, random_state=rng)


def exact_reserve(distribution: object) -> float:
    """Return the global monopoly-price maximizer, allowing nonregular mixtures."""
    lower = float(distribution.ppf(1e-5))
    upper = float(distribution.ppf(1.0 - 1e-5))
    grid = np.geomspace(lower, upper, 200_000)
    revenue = grid * (1.0 - distribution.cdf(grid))
    index = int(np.argmax(revenue))
    left = grid[max(index - 2, 0)]
    right = grid[min(index + 2, grid.size - 1)]
    refined = minimize_scalar(
        lambda reserve: -reserve * (1.0 - float(distribution.cdf(reserve))),
        bounds=(left, right),
        method="bounded",
    )
    return float(refined.x)


def maximum_quantiles(
    distribution: object,
    bidders: int,
    draws: int,
) -> np.ndarray:
    probabilities = (np.arange(draws, dtype=float) + 0.5) / draws
    return distribution.ppf(probabilities ** (1.0 / bidders))


def target_and_grid(
    distribution: object,
    bidders: int,
    exact: float,
    config: ExternalValidityConfig,
) -> tuple[np.ndarray, np.ndarray, float, int, AblationConfig]:
    true_maxima = maximum_quantiles(
        distribution, bidders, config.target_integration_draws
    )
    theta_lower = max(float(distribution.ppf(0.01)), 0.4 * exact)
    theta_upper = max(float(distribution.ppf(0.85)), 1.8 * exact)
    theta_grid = np.linspace(theta_lower, theta_upper, config.theta_grid_size)
    local_config = AblationConfig(
        bidders=bidders,
        sample_sizes=config.sample_sizes,
        repetitions=config.repetitions,
        folds=config.folds,
        bandwidth=config.bandwidth,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
        theta_grid_size=config.theta_grid_size,
        jacobian_step=config.jacobian_step,
        target_integration_draws=config.target_integration_draws,
        seed=config.seed,
    )
    score = score_grid_for_fold(
        theta_grid,
        true_maxima,
        true_maxima,
        config.bandwidth,
        bidders,
    )
    target, root_count = select_score_root(theta_grid, score, exact)
    if root_count == 0:
        raise RuntimeError(
            f"Population score has no root for N={bidders}; expand the grid."
        )
    return true_maxima, theta_grid, target, root_count, local_config


def run_experiment(config: ExternalValidityConfig) -> pd.DataFrame:
    records: list[dict[str, float | int | str | bool]] = []
    start = time.perf_counter()
    for dgp_index, (dgp, distribution) in enumerate(distributions().items()):
        exact = exact_reserve(distribution)
        for bidders in config.bidders:
            true_maxima, theta_grid, target, target_roots, local_config = (
                target_and_grid(distribution, bidders, exact, config)
            )
            print(
                f"{dgp:>10s} N={bidders:>2}: r0={exact:.5f}, "
                f"rh={target:.5f}, population roots={target_roots}",
                flush=True,
            )
            for sample_size in config.sample_sizes:
                for repetition in range(config.repetitions):
                    data_seed = (
                        config.seed
                        + dgp_index * 100_000_000
                        + bidders * 1_000_000
                        + sample_size * 100
                        + repetition
                    )
                    rng = np.random.default_rng(data_seed)
                    values = sample_values(
                        distribution, (sample_size, bidders), rng
                    )
                    maxima = values.max(axis=1)
                    direct = estimate_reserve_from_maxima(maxima, bidders)
                    folds = make_folds(
                        sample_size, config.folds, data_seed + 31_415_926
                    )
                    result = estimate_method(
                        "Empirical-local DML",
                        maxima,
                        folds,
                        true_maxima,
                        theta_grid,
                        direct,
                        local_config,
                    )
                    estimate = float(result["reserve_estimate"])
                    standard_error = float(result["standard_error"])
                    ci_lower = estimate - 1.96 * standard_error
                    ci_upper = estimate + 1.96 * standard_error
                    records.append(
                        {
                            "dgp": dgp,
                            "bidders": bidders,
                            "auctions": sample_size,
                            "repetition": repetition + 1,
                            "exact_reserve": exact,
                            "regularized_target": target,
                            "population_root_count": target_roots,
                            "dml_estimate": estimate,
                            "dml_target_error": estimate - target,
                            "dml_exact_error": estimate - exact,
                            "standard_error": standard_error,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "covers_regularized_target": bool(
                                ci_lower <= target <= ci_upper
                            ),
                            "covers_exact_reserve": bool(
                                ci_lower <= exact <= ci_upper
                            ),
                            "sample_root_count": int(
                                result["score_roots_on_grid"]
                            ),
                            "direct_estimate": direct,
                            "direct_exact_error": direct - exact,
                            "maxima_below_exact_reserve": int(
                                np.sum(maxima <= exact)
                            ),
                        }
                    )
                    if repetition == 0 or (repetition + 1) % 50 == 0:
                        elapsed = time.perf_counter() - start
                        print(
                            f"  n={sample_size:>6,} rep={repetition + 1:>3}/"
                            f"{config.repetitions} elapsed={elapsed:.1f}s",
                            flush=True,
                        )
    raw = pd.DataFrame.from_records(records)
    expected = (
        len(distributions())
        * len(config.bidders)
        * len(config.sample_sizes)
        * config.repetitions
    )
    if len(raw) != expected:
        raise RuntimeError(f"Expected {expected} records, received {len(raw)}.")
    return raw


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["dgp", "bidders", "auctions"], sort=False, observed=True)
        .agg(
            repetitions=("repetition", "count"),
            exact_reserve=("exact_reserve", "first"),
            regularized_target=("regularized_target", "first"),
            regularization_bias=(
                "regularized_target",
                lambda values: values.iloc[0]
                - raw.loc[values.index[0], "exact_reserve"],
            ),
            dml_target_bias=("dml_target_error", "mean"),
            dml_target_rmse=(
                "dml_target_error", lambda values: np.sqrt(np.mean(values**2))
            ),
            target_coverage=("covers_regularized_target", "mean"),
            dml_exact_rmse=(
                "dml_exact_error", lambda values: np.sqrt(np.mean(values**2))
            ),
            direct_exact_rmse=(
                "direct_exact_error", lambda values: np.sqrt(np.mean(values**2))
            ),
            multiple_root_rate=(
                "sample_root_count", lambda values: np.mean(values > 1)
            ),
            missing_root_rate=(
                "sample_root_count", lambda values: np.mean(values == 0)
            ),
            mean_maxima_below_exact=("maxima_below_exact_reserve", "mean"),
        )
        .reset_index()
    )


def write_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    config: ExternalValidityConfig,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TABLE_DIR / "external_validity_raw.csv", index=False)
    summary.to_csv(TABLE_DIR / "external_validity.csv", index=False)
    (TABLE_DIR / "external_validity_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    displayed = summary[
        [
            "dgp",
            "bidders",
            "auctions",
            "regularization_bias",
            "dml_target_rmse",
            "target_coverage",
            "dml_exact_rmse",
            "direct_exact_rmse",
            "multiple_root_rate",
        ]
    ].copy()
    displayed.columns = [
        "DGP",
        "$N$",
        "$n$",
        r"$r_h-r_0$",
        r"DML RMSE to $r_h$",
        r"Coverage of $r_h$",
        r"DML RMSE to $r_0$",
        r"Direct RMSE to $r_0$",
        "Multi-root",
    ]
    latex = displayed.to_latex(
        index=False,
        float_format=lambda value: f"{value:.3f}",
        escape=False,
        column_format="lrrrrrrrr",
    )
    (TABLE_DIR / "external_validity.tex").write_text(latex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bandwidth", type=float, default=0.15)
    parser.add_argument("--grid-size", type=int, default=121)
    parser.add_argument("--target-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_717)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[2_000, 10_000]
    )
    parser.add_argument("--bidders", type=int, nargs="+", default=[3, 5, 10])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExternalValidityConfig(
        bidders=tuple(args.bidders),
        sample_sizes=tuple(args.sample_sizes),
        repetitions=args.reps,
        folds=args.folds,
        bandwidth=args.bandwidth,
        theta_grid_size=args.grid_size,
        target_integration_draws=args.target_draws,
        seed=args.seed,
    )
    raw = run_experiment(config)
    summary = summarize(raw)
    write_outputs(raw, summary, config)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
