#!/usr/bin/env python3
"""Fast deterministic checks for the replication package."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

MODULES = (
    "baseline_direct_inversion",
    "wgan_gp_baseline",
    "crossfit_wgan_pilot",
    "orthogonal_wgan_dml",
    "local_nuisance_ablation",
    "bandwidth_path",
    "bias_corrected_inference",
    "root_robust_inference",
    "external_validity_monte_carlo",
)

CONFIG_FILES = {
    "wgan_gp_baseline": "wgan_gp_pilot_config.json",
    "crossfit_wgan_pilot": "crossfit_wgan_pilot_config.json",
    "orthogonal_wgan_dml": "orthogonal_wgan_dml_config.json",
    "local_nuisance_ablation": "local_nuisance_ablation_config.json",
    "bandwidth_path": "bandwidth_path_config.json",
    "bias_corrected_inference": "bias_corrected_inference_config.json",
    "root_robust_inference": "root_robust_inference_config.json",
    "external_validity_monte_carlo": "external_validity_config.json",
}

CONFIG_CLASSES = {
    "wgan_gp_baseline": "WGANConfig",
    "crossfit_wgan_pilot": "CrossFitConfig",
    "orthogonal_wgan_dml": "DMLConfig",
    "local_nuisance_ablation": "AblationConfig",
    "bandwidth_path": "BandwidthPathConfig",
    "bias_corrected_inference": "BiasCorrectionConfig",
    "root_robust_inference": "RootRobustConfig",
    "external_validity_monte_carlo": "ExternalValidityConfig",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_imports_and_recorded_seeds() -> dict[str, object]:
    seed_registry = json.loads(
        (ROOT / "replication" / "seeds.json").read_text(encoding="utf-8")
    )
    imported = {name: importlib.import_module(name) for name in MODULES}
    table_dir = ROOT / "paper" / "tables"

    for module_name, config_filename in CONFIG_FILES.items():
        module = imported[module_name]
        config_class = getattr(module, CONFIG_CLASSES[module_name])
        field_names = {field.name for field in fields(config_class)}
        require("seed" in field_names, f"{config_class.__name__} has no seed field")
        code_seed = config_class().seed
        recorded_config = json.loads(
            (table_dir / config_filename).read_text(encoding="utf-8")
        )
        require(recorded_config["seed"] == code_seed, f"Seed mismatch in {config_filename}")
        require(seed_registry["scripts"][module_name]["base_seed"] == code_seed, f"Seed registry mismatch for {module_name}")
    return imported


def check_cross_fitting(module: object) -> None:
    first = module.make_folds(23, 4, 12345)
    second = module.make_folds(23, 4, 12345)
    require(len(first) == 4, "Cross-fitting did not create four folds")
    require(all(np.array_equal(a, b) for a, b in zip(first, second)), "Fold assignment is not deterministic")
    combined = np.concatenate(first)
    require(np.array_equal(np.sort(combined), np.arange(23)), "Folds are not a partition")


def check_score_helpers(module: object) -> None:
    theta = np.linspace(0.5, 1.1, 9)
    sample = np.linspace(0.35, 1.8, 101)
    cdf, density = module.gaussian_kernel_moments(theta, sample, 0.15)
    require(np.all(np.diff(cdf) >= 0.0), "Smoothed CDF is not monotone")
    require(np.all(density > 0.0), "Smoothed density is not positive")
    root, count = module.select_score_root(
        np.array([0.0, 1.0, 2.0]), np.array([-1.0, 1.0, 3.0]), 0.4
    )
    require(count == 1 and np.isclose(root, 0.5), "Score-root interpolation failed")


def check_interval_helpers(module: object) -> None:
    merged = module.merge_intervals([(0.0, 1.0), (0.8, 1.4), (2.0, 3.0)])
    require(merged == [(0.0, 1.4), (2.0, 3.0)], "Interval union failed")
    require(module.contains(merged, 1.2), "Interval containment failed")
    require(not module.contains(merged, 1.7), "Interval exclusion failed")


def check_release_files() -> None:
    required = (
        "CITATION.cff",
        "paper/main.tex",
        "paper/main_anonymous.tex",
        "paper/references.bib",
        "docs/submission_strategy.md",
        "docs/working_paper_checklist.md",
        "docs/working_paper_metadata.md",
        "replication/environment.json",
        "replication/seeds.json",
    )
    for relative in required:
        path = ROOT / relative
        require(path.is_file() and path.stat().st_size > 0, f"Missing required file: {relative}")


def main() -> None:
    imported = check_imports_and_recorded_seeds()
    check_cross_fitting(imported["crossfit_wgan_pilot"])
    check_score_helpers(imported["orthogonal_wgan_dml"])
    check_interval_helpers(imported["root_robust_inference"])
    check_release_files()
    print(
        "Replication smoke checks passed: imports, seeds, folds, score helpers, "
        "interval helpers, and working-paper release files."
    )


if __name__ == "__main__":
    main()
