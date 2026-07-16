#!/usr/bin/env python3
"""Structural WGAN-GP pilot for reserve estimation from auction maxima."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from torch import nn

from baseline_direct_inversion import (
    SimulationConfig,
    estimate_reserve_from_maxima,
    make_revenue_interpolator,
    true_reserve,
    valuation_distribution,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "paper" / "tables"
FIGURE_DIR = ROOT / "paper" / "figures"


@dataclass(frozen=True)
class WGANConfig:
    bidders: int = 5
    sample_sizes: tuple[int, ...] = (500, 2_000, 10_000)
    repetitions: int = 5
    training_steps: int = 1_200
    critic_steps: int = 5
    batch_size: int = 512
    latent_dim: int = 8
    hidden_dim: int = 64
    learning_rate: float = 1e-4
    gradient_penalty: float = 0.1
    generated_values: int = 50_000
    diagnostic_auctions: int = 20_000
    seed: int = 20_260_716
    device: str = "cpu"


class ValueGenerator(nn.Module):
    """Map an independent bidder-level noise draw to one log valuation."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_center: float,
        output_half_width: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer(
            "output_center", torch.tensor(float(output_center), dtype=torch.float32)
        )
        self.register_buffer(
            "output_half_width",
            torch.tensor(float(output_half_width), dtype=torch.float32),
        )
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.output_center + self.output_half_width * torch.tanh(
            self.network(noise)
        )


class MaxBidCritic(nn.Module):
    """One-dimensional critic operating on standardized auction maxima."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, maximum: torch.Tensor) -> torch.Tensor:
        return self.network(maximum)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generated_maxima(
    generator: ValueGenerator,
    batch_size: int,
    bidders: int,
    device: torch.device,
) -> torch.Tensor:
    noise = torch.randn(
        batch_size * bidders, generator.latent_dim, device=device
    )
    log_values = generator(noise).reshape(batch_size, bidders)
    return log_values.max(dim=1, keepdim=True).values


def gradient_penalty(
    critic: MaxBidCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    weight = torch.rand(real.shape[0], 1, device=real.device)
    interpolated = weight * real + (1.0 - weight) * fake
    interpolated.requires_grad_(True)
    score = critic(interpolated)
    gradient = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((gradient.norm(2, dim=1) - 1.0) ** 2).mean()


def train_wgan_gp(
    observed_maxima: np.ndarray,
    config: WGANConfig,
    seed: int,
) -> tuple[ValueGenerator, dict[str, float | list[float]]]:
    set_seed(seed)
    device = choose_device(config.device)
    real_raw = torch.as_tensor(
        np.log(observed_maxima).reshape(-1, 1), dtype=torch.float32, device=device
    )
    location = real_raw.mean()
    scale = real_raw.std().clamp_min(1e-6)
    real = (real_raw - location) / scale

    generator = ValueGenerator(
        config.latent_dim,
        config.hidden_dim,
        output_center=float(location.detach().cpu()),
        output_half_width=float((4.0 * scale).detach().cpu()),
    ).to(device)
    critic = MaxBidCritic(config.hidden_dim).to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.2 * config.learning_rate, betas=(0.0, 0.9)
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.learning_rate, betas=(0.0, 0.9)
    )
    rng = np.random.default_rng(seed)
    batch_size = min(config.batch_size, real.shape[0])
    critic_history: list[float] = []
    generator_history: list[float] = []
    fit_history: list[float] = []
    best_fit = float("inf")
    best_step = 0
    best_generator_state = copy.deepcopy(generator.state_dict())
    evaluation_size = min(4_096, real.shape[0])
    evaluation_interval = max(config.training_steps // 60, 10)

    generator.train()
    critic.train()
    for step in range(config.training_steps):
        for _ in range(config.critic_steps):
            index = torch.as_tensor(
                rng.integers(0, real.shape[0], size=batch_size),
                dtype=torch.long,
                device=device,
            )
            real_batch = real[index]
            fake_log_maximum = generated_maxima(
                generator, batch_size, config.bidders, device
            )
            fake = (fake_log_maximum - location) / scale
            penalty = gradient_penalty(critic, real_batch, fake.detach())
            critic_loss = (
                critic(fake.detach()).mean()
                - critic(real_batch).mean()
                + config.gradient_penalty * penalty
            )
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_optimizer.step()

        fake_log_maximum = generated_maxima(
            generator, batch_size, config.bidders, device
        )
        fake = (fake_log_maximum - location) / scale
        generator_loss = -critic(fake).mean()
        generator_optimizer.zero_grad(set_to_none=True)
        generator_loss.backward()
        generator_optimizer.step()

        if step % max(config.training_steps // 100, 1) == 0:
            critic_history.append(float(critic_loss.detach().cpu()))
            generator_history.append(float(generator_loss.detach().cpu()))

        if step % evaluation_interval == 0 or step == config.training_steps - 1:
            generator.eval()
            with torch.no_grad():
                fake_log_maximum = generated_maxima(
                    generator, evaluation_size, config.bidders, device
                )
                fake_maximum = np.exp(
                    fake_log_maximum.cpu().numpy().reshape(-1)
                )
            current_fit = wasserstein_distance(
                observed_maxima, fake_maximum
            )
            fit_history.append(float(current_fit))
            if current_fit < best_fit:
                best_fit = float(current_fit)
                best_step = step + 1
                best_generator_state = copy.deepcopy(generator.state_dict())
            generator.train()

    generator.load_state_dict(best_generator_state)

    diagnostics: dict[str, float | list[float]] = {
        "critic_loss": float(critic_loss.detach().cpu()),
        "generator_loss": float(generator_loss.detach().cpu()),
        "critic_history": critic_history,
        "generator_history": generator_history,
        "fit_history": fit_history,
        "best_max_bid_w1": best_fit,
        "best_step": float(best_step),
    }
    generator.eval()
    return generator, diagnostics


def sample_values(
    generator: ValueGenerator,
    sample_size: int,
    device: torch.device,
    batch_size: int = 8_192,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, sample_size, batch_size):
            current = min(batch_size, sample_size - start)
            noise = torch.randn(current, generator.latent_dim, device=device)
            log_values = generator(noise)
            samples.append(np.exp(log_values.cpu().numpy().reshape(-1)))
    return np.concatenate(samples)


def sample_maxima(
    generator: ValueGenerator,
    auctions: int,
    bidders: int,
    device: torch.device,
    batch_size: int = 4_096,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, auctions, batch_size):
            current = min(batch_size, auctions - start)
            log_maximum = generated_maxima(generator, current, bidders, device)
            samples.append(np.exp(log_maximum.cpu().numpy().reshape(-1)))
    return np.concatenate(samples)


def estimate_reserve_from_values(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    empirical_f = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    objective = ordered * (1.0 - empirical_f)
    # Exclude only the most extreme generated tail, where a single neural
    # network draw can otherwise dominate the finite simulation objective.
    upper = max(2, int(np.floor(0.995 * ordered.size)))
    return float(ordered[:upper][np.argmax(objective[:upper])])


def run_pilot(
    config: WGANConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray | list[float] | int]]:
    baseline = SimulationConfig(bidders=config.bidders)
    distribution = valuation_distribution(baseline)
    reserve = true_reserve(baseline)
    revenue_at = make_revenue_interpolator(baseline)
    optimal_revenue = float(revenue_at(reserve))
    device = choose_device(config.device)
    records: list[dict[str, float | int | str]] = []
    example: dict[str, np.ndarray | list[float] | int] = {}
    start_time = time.perf_counter()

    for sample_size in config.sample_sizes:
        for repetition in range(config.repetitions):
            seed = config.seed + sample_size * 100 + repetition
            rng = np.random.default_rng(seed)
            values = distribution.rvs(
                size=(sample_size, config.bidders), random_state=rng
            )
            maxima = values.max(axis=1)
            direct_reserve = estimate_reserve_from_maxima(maxima, config.bidders)

            generator, training = train_wgan_gp(maxima, config, seed)
            generated_values = sample_values(
                generator, config.generated_values, device
            )
            generated_observations = sample_maxima(
                generator,
                config.diagnostic_auctions,
                config.bidders,
                device,
            )
            wgan_reserve = estimate_reserve_from_values(generated_values)
            true_values = distribution.rvs(
                size=config.generated_values, random_state=rng
            )

            common = {
                "auctions": sample_size,
                "repetition": repetition + 1,
                "bidders": config.bidders,
                "true_reserve": reserve,
            }
            for method, estimate in (
                ("Direct inversion", direct_reserve),
                ("WGAN-GP", wgan_reserve),
            ):
                regret = max(optimal_revenue - float(revenue_at(estimate)), 0.0)
                records.append(
                    {
                        **common,
                        "method": method,
                        "reserve_estimate": estimate,
                        "reserve_error": estimate - reserve,
                        "absolute_reserve_error": abs(estimate - reserve),
                        "revenue_regret": regret,
                        "max_bid_w1": (
                            wasserstein_distance(maxima, generated_observations)
                            if method == "WGAN-GP"
                            else np.nan
                        ),
                        "latent_value_w1": (
                            wasserstein_distance(true_values, generated_values)
                            if method == "WGAN-GP"
                            else np.nan
                        ),
                    }
                )

            if sample_size == config.sample_sizes[-1] and repetition == 0:
                example = {
                    "sample_size": sample_size,
                    "true_values": true_values,
                    "generated_values": generated_values,
                    "true_maxima": maxima,
                    "generated_maxima": generated_observations,
                    "critic_history": training["critic_history"],
                }

            elapsed = time.perf_counter() - start_time
            print(
                f"n={sample_size:>6,} rep={repetition + 1:>2}/"
                f"{config.repetitions} direct={direct_reserve:.4f} "
                f"wgan={wgan_reserve:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    return pd.DataFrame.from_records(records), example


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["auctions", "method"], sort=True)
    summary = grouped.agg(
        repetitions=("repetition", "count"),
        true_reserve=("true_reserve", "first"),
        mean_estimate=("reserve_estimate", "mean"),
        bias=("reserve_error", "mean"),
        standard_deviation=("reserve_estimate", "std"),
        rmse=("reserve_error", lambda values: np.sqrt(np.mean(values**2))),
        mean_revenue_regret=("revenue_regret", "mean"),
        mean_max_bid_w1=("max_bid_w1", "mean"),
        mean_latent_value_w1=("latent_value_w1", "mean"),
    ).reset_index()
    return summary


def write_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    example: dict[str, np.ndarray | list[float] | int],
    config: WGANConfig,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TABLE_DIR / "wgan_gp_pilot_raw.csv", index=False)
    summary.to_csv(TABLE_DIR / "wgan_gp_pilot.csv", index=False)
    (TABLE_DIR / "wgan_gp_pilot_config.json").write_text(
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
            "mean_max_bid_w1",
        ]
    ].copy()
    displayed.columns = [
        "$n$",
        "Estimator",
        r"Mean $\widehat r$",
        "Bias",
        "RMSE",
        "Mean regret",
        r"Max-bid $W_1$",
    ]
    displayed["Estimator"] = displayed["Estimator"].replace(
        {"Direct inversion": "Direct", "WGAN-GP": "WGAN-GP"}
    )
    latex = displayed.to_latex(
        index=False,
        float_format=lambda value: "--" if pd.isna(value) else f"{value:.4f}",
        escape=False,
        column_format="rlrrrrr",
        na_rep="--",
    )
    (TABLE_DIR / "wgan_gp_pilot.tex").write_text(latex, encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    distribution = valuation_distribution(SimulationConfig(bidders=config.bidders))
    grid = np.linspace(distribution.ppf(0.005), distribution.ppf(0.995), 1_000)
    true_values = np.sort(np.asarray(example["true_values"]))
    generated_values = np.sort(np.asarray(example["generated_values"]))
    axes[0].plot(grid, distribution.cdf(grid), color="black", lw=2, label="True latent $F$")
    axes[0].step(
        generated_values,
        np.arange(1, generated_values.size + 1) / generated_values.size,
        where="post",
        color="#4472C4",
        lw=1.5,
        label="WGAN-GP latent $F$",
    )
    axes[0].set_xlim(grid[0], distribution.ppf(0.985))
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("Valuation")
    axes[0].set_ylabel("CDF")
    axes[0].set_title(fr"Latent recovery, $n={int(example['sample_size']):,}$")
    axes[0].legend(frameon=False, fontsize=9)

    methods = ["Direct inversion", "WGAN-GP"]
    colors = {"Direct inversion": "#C55A11", "WGAN-GP": "#4472C4"}
    offsets = {"Direct inversion": -0.16, "WGAN-GP": 0.16}
    x_positions = np.arange(len(config.sample_sizes))
    rng = np.random.default_rng(config.seed + 99)
    for method in methods:
        for x_position, sample_size in zip(x_positions, config.sample_sizes):
            selected = raw[
                (raw["method"] == method) & (raw["auctions"] == sample_size)
            ]["reserve_estimate"].to_numpy()
            jitter = rng.normal(0.0, 0.025, size=selected.size)
            axes[1].scatter(
                x_position + offsets[method] + jitter,
                selected,
                s=28,
                alpha=0.75,
                color=colors[method],
                label=method if x_position == 0 else None,
            )
            axes[1].plot(
                [x_position + offsets[method] - 0.08, x_position + offsets[method] + 0.08],
                [selected.mean(), selected.mean()],
                color=colors[method],
                lw=2,
            )
    reserve = true_reserve(SimulationConfig(bidders=config.bidders))
    axes[1].axhline(reserve, color="black", linestyle="--", lw=1.5, label="True reserve")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([f"{size:,}" for size in config.sample_sizes])
    axes[1].set_xlabel("Number of auctions")
    axes[1].set_ylabel("Estimated reserve")
    axes[1].set_title("Pilot estimates across training seeds")
    axes[1].legend(frameon=False, fontsize=9)

    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "wgan_gp_pilot.png", dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gp-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[500, 2_000, 10_000])
    parser.add_argument("--generated-values", type=int, default=50_000)
    parser.add_argument("--diagnostic-auctions", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = WGANConfig(
        sample_sizes=tuple(args.sample_sizes),
        repetitions=args.reps,
        training_steps=args.steps,
        critic_steps=args.critic_steps,
        batch_size=args.batch_size,
        gradient_penalty=args.gp_weight,
        seed=args.seed,
        device=args.device,
        generated_values=args.generated_values,
        diagnostic_auctions=args.diagnostic_auctions,
    )
    raw, example = run_pilot(config)
    summary = summarize(raw)
    write_outputs(raw, summary, example, config)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))


if __name__ == "__main__":
    main()
