#!/usr/bin/env python3
"""K-fold WGAN plug-in precursor for the auction-reserve DML design."""

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
from scipy.stats import wasserstein_distance

from baseline_direct_inversion import (
    SimulationConfig,
    estimate_reserve_from_maxima,
    make_revenue_interpolator,
    true_reserve,
    valuation_distribution,
)
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
DEFAULT_BASELINE_RAW = TABLE_DIR / "wgan_gp_pilot_raw.csv"


@dataclass(frozen=True)
class CrossFitConfig:
    bidders: int = 5
    sample_sizes: tuple[int, ...] = (500, 2_000, 10_000)
    repetitions: int = 5
    folds: int = 3
    training_steps: int = 1_200
    critic_steps: int = 5
    batch_size: int = 512
    latent_dim: int = 8
    hidden_dim: int = 64
    learning_rate: float = 1e-4
    gradient_penalty: float = 0.1
    generated_values_per_fold: int = 50_000
    diagnostic_auctions_per_fold: int = 20_000
    seed: int = 20_260_716
    device: str = "cpu"


def make_folds(sample_size: int, folds: int, seed: int) -> list[np.ndarray]:
    if folds < 2:
        raise ValueError("Cross-fitting requires at least two folds.")
    if folds > sample_size:
        raise ValueError("The number of folds cannot exceed the sample size.")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(sample_size)
    return [np.sort(index) for index in np.array_split(permutation, folds)]


def load_baseline_rows(path: Path, config: CrossFitConfig) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline results not found at {path}. Run code/wgan_gp_baseline.py first."
        )
    baseline = pd.read_csv(path)
    selected = baseline[
        baseline["auctions"].isin(config.sample_sizes)
        & baseline["repetition"].between(1, config.repetitions)
        & baseline["method"].isin(["Direct inversion", "WGAN-GP"])
    ].copy()
    expected = len(config.sample_sizes) * config.repetitions * 2
    if len(selected) != expected:
        raise ValueError(
            "Baseline raw results do not match the requested sample sizes and repetitions."
        )
    selected["validation_max_bid_w1"] = np.nan
    selected["fold_reserve_sd"] = np.nan
    selected["pooled_reserve"] = np.nan
    selected["training_auctions"] = selected["auctions"]
    return selected


def run_crossfit(
    config: CrossFitConfig,
    baseline_raw: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = SimulationConfig(bidders=config.bidders)
    distribution = valuation_distribution(baseline)
    reserve = true_reserve(baseline)
    revenue_at = make_revenue_interpolator(baseline)
    optimal_revenue = float(revenue_at(reserve))
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
        diagnostic_auctions=config.diagnostic_auctions_per_fold,
        seed=config.seed,
        device=config.device,
    )
    baseline_rows = load_baseline_rows(baseline_raw, config)
    crossfit_records: list[dict[str, float | int | str]] = []
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
            direct_reserve = estimate_reserve_from_maxima(maxima, config.bidders)
            recorded_direct = baseline_rows[
                (baseline_rows["auctions"] == sample_size)
                & (baseline_rows["repetition"] == repetition + 1)
                & (baseline_rows["method"] == "Direct inversion")
            ]["reserve_estimate"].iloc[0]
            if not np.isclose(direct_reserve, recorded_direct, atol=1e-12):
                raise RuntimeError("The cross-fit DGP no longer matches the baseline pilot.")

            heldout_folds = make_folds(
                sample_size, config.folds, data_seed + 31_415_926
            )
            all_indices = np.arange(sample_size)
            fold_reserves: list[float] = []
            generated_value_blocks: list[np.ndarray] = []
            validation_distances: list[float] = []
            latent_distances: list[float] = []

            for fold_index, holdout in enumerate(heldout_folds):
                training = np.setdiff1d(all_indices, holdout, assume_unique=True)
                fold_seed = data_seed + 1_000_000 + (fold_index + 1) * 10_000
                generator, diagnostics = train_wgan_gp(
                    maxima[training], wgan_config, fold_seed
                )
                set_seed(fold_seed + 101)
                generated_values = sample_values(
                    generator, config.generated_values_per_fold, device
                )
                generated_maxima = sample_maxima(
                    generator,
                    config.diagnostic_auctions_per_fold,
                    config.bidders,
                    device,
                )
                fold_reserve = estimate_reserve_from_values(generated_values)
                validation_w1 = wasserstein_distance(
                    maxima[holdout], generated_maxima
                )
                true_values = distribution.rvs(
                    size=config.generated_values_per_fold,
                    random_state=np.random.default_rng(fold_seed + 202),
                )
                latent_w1 = wasserstein_distance(true_values, generated_values)
                fold_reserves.append(fold_reserve)
                generated_value_blocks.append(generated_values)
                validation_distances.append(float(validation_w1))
                latent_distances.append(float(latent_w1))
                fold_records.append(
                    {
                        "auctions": sample_size,
                        "repetition": repetition + 1,
                        "fold": fold_index + 1,
                        "training_auctions": training.size,
                        "heldout_auctions": holdout.size,
                        "reserve_estimate": fold_reserve,
                        "reserve_error": fold_reserve - reserve,
                        "training_checkpoint_w1": float(
                            diagnostics["best_max_bid_w1"]
                        ),
                        "validation_max_bid_w1": validation_w1,
                        "latent_value_w1": latent_w1,
                        "best_step": int(diagnostics["best_step"]),
                    }
                )

            estimate = float(np.mean(fold_reserves))
            pooled_reserve = estimate_reserve_from_values(
                np.concatenate(generated_value_blocks)
            )
            regret = max(optimal_revenue - float(revenue_at(estimate)), 0.0)
            crossfit_records.append(
                {
                    "auctions": sample_size,
                    "repetition": repetition + 1,
                    "bidders": config.bidders,
                    "true_reserve": reserve,
                    "method": "Fold-averaged WGAN",
                    "reserve_estimate": estimate,
                    "reserve_error": estimate - reserve,
                    "absolute_reserve_error": abs(estimate - reserve),
                    "revenue_regret": regret,
                    "max_bid_w1": np.nan,
                    "latent_value_w1": float(np.mean(latent_distances)),
                    "validation_max_bid_w1": float(
                        np.mean(validation_distances)
                    ),
                    "fold_reserve_sd": float(np.std(fold_reserves, ddof=1)),
                    "pooled_reserve": pooled_reserve,
                    "training_auctions": sample_size - len(heldout_folds[0]),
                }
            )
            elapsed = time.perf_counter() - start_time
            print(
                f"n={sample_size:>6,} rep={repetition + 1:>2}/"
                f"{config.repetitions} fold-mean={estimate:.4f} "
                f"fold-sd={np.std(fold_reserves, ddof=1):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    raw = pd.concat(
        [baseline_rows, pd.DataFrame.from_records(crossfit_records)],
        ignore_index=True,
        sort=False,
    )
    raw["method"] = pd.Categorical(
        raw["method"],
        categories=["Direct inversion", "WGAN-GP", "Fold-averaged WGAN"],
        ordered=True,
    )
    raw = raw.sort_values(["auctions", "method", "repetition"]).reset_index(
        drop=True
    )
    return raw, pd.DataFrame.from_records(fold_records)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["auctions", "method"], observed=True, sort=True)
        .agg(
            repetitions=("repetition", "count"),
            true_reserve=("true_reserve", "first"),
            mean_estimate=("reserve_estimate", "mean"),
            bias=("reserve_error", "mean"),
            standard_deviation=("reserve_estimate", "std"),
            rmse=("reserve_error", lambda x: np.sqrt(np.mean(x**2))),
            mean_revenue_regret=("revenue_regret", "mean"),
            mean_validation_max_bid_w1=("validation_max_bid_w1", "mean"),
            mean_fold_reserve_sd=("fold_reserve_sd", "mean"),
        )
        .reset_index()
    )


def write_outputs(
    raw: pd.DataFrame,
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    config: CrossFitConfig,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TABLE_DIR / "crossfit_wgan_pilot_raw.csv", index=False)
    folds.to_csv(TABLE_DIR / "crossfit_wgan_folds.csv", index=False)
    summary.to_csv(TABLE_DIR / "crossfit_wgan_pilot.csv", index=False)
    (TABLE_DIR / "crossfit_wgan_pilot_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    displayed = summary[
        [
            "auctions",
            "method",
            "mean_estimate",
            "bias",
            "rmse",
            "mean_revenue_regret",
            "mean_validation_max_bid_w1",
            "mean_fold_reserve_sd",
        ]
    ].copy()
    displayed.columns = [
        "$n$",
        "Estimator",
        r"Mean $\widehat r$",
        "Bias",
        "RMSE",
        "Mean regret",
        r"Held-out $W_1$",
        "Fold SD",
    ]
    displayed["Estimator"] = displayed["Estimator"].astype("object").replace(
        {
            "Direct inversion": "Direct",
            "WGAN-GP": "Full WGAN",
            "Fold-averaged WGAN": "Fold average",
        }
    )
    latex = displayed.to_latex(
        index=False,
        float_format=lambda value: "--" if pd.isna(value) else f"{value:.4f}",
        escape=False,
        column_format="rlrrrrrr",
        na_rep="--",
    )
    (TABLE_DIR / "crossfit_wgan_pilot.tex").write_text(
        latex, encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    methods = ["WGAN-GP", "Fold-averaged WGAN"]
    colors = {"WGAN-GP": "#4472C4", "Fold-averaged WGAN": "#70AD47"}
    offsets = {"WGAN-GP": -0.13, "Fold-averaged WGAN": 0.13}
    x_positions = np.arange(len(config.sample_sizes))
    jitter_rng = np.random.default_rng(config.seed + 808)
    for method in methods:
        for x_position, sample_size in zip(x_positions, config.sample_sizes):
            selected = raw[
                (raw["method"] == method) & (raw["auctions"] == sample_size)
            ]["reserve_estimate"].to_numpy()
            jitter = jitter_rng.normal(0.0, 0.018, selected.size)
            axes[0].scatter(
                x_position + offsets[method] + jitter,
                selected,
                s=30,
                alpha=0.8,
                color=colors[method],
                label=method if x_position == 0 else None,
            )
            axes[0].plot(
                [x_position + offsets[method] - 0.07, x_position + offsets[method] + 0.07],
                [selected.mean(), selected.mean()],
                color=colors[method],
                lw=2,
            )
    reserve = true_reserve(SimulationConfig(bidders=config.bidders))
    axes[0].axhline(reserve, color="black", linestyle="--", lw=1.5)
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels([f"{size:,}" for size in config.sample_sizes])
    axes[0].set_xlabel("Number of auctions")
    axes[0].set_ylabel("Estimated reserve")
    axes[0].set_title("Full-sample and fold-averaged WGAN")
    axes[0].legend(frameon=False, fontsize=9)

    for x_position, sample_size in zip(x_positions, config.sample_sizes):
        selected_folds = folds[folds["auctions"] == sample_size]
        jitter = jitter_rng.normal(0.0, 0.055, len(selected_folds))
        axes[1].scatter(
            x_position + jitter,
            selected_folds["validation_max_bid_w1"],
            color="#70AD47",
            s=28,
            alpha=0.8,
        )
        mean_distance = selected_folds["validation_max_bid_w1"].mean()
        axes[1].plot(
            [x_position - 0.14, x_position + 0.14],
            [mean_distance, mean_distance],
            color="#70AD47",
            lw=2,
        )
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([f"{size:,}" for size in config.sample_sizes])
    axes[1].set_xlabel("Number of auctions")
    axes[1].set_ylabel(r"Held-out maximum-bid $W_1$")
    axes[1].set_title("Out-of-fold observable fit")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "crossfit_wgan_pilot.png", dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
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
    parser.add_argument("--diagnostic-auctions", type=int, default=20_000)
    parser.add_argument(
        "--baseline-raw", type=Path, default=DEFAULT_BASELINE_RAW
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CrossFitConfig(
        sample_sizes=tuple(args.sample_sizes),
        repetitions=args.reps,
        folds=args.folds,
        training_steps=args.steps,
        critic_steps=args.critic_steps,
        batch_size=args.batch_size,
        gradient_penalty=args.gp_weight,
        seed=args.seed,
        device=args.device,
        generated_values_per_fold=args.generated_values,
        diagnostic_auctions_per_fold=args.diagnostic_auctions,
    )
    raw, folds = run_crossfit(config, args.baseline_raw)
    summary = summarize(raw)
    write_outputs(raw, folds, summary, config)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
