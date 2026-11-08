"""Independently verify a completed fixed-h 100-repetition experiment.

This script neither trains estimators nor changes experiment inputs.  It writes
one new summary/verification.json, refusing to replace an existing report.
All calculations below use the Python standard library, not the estimator or
aggregation modules being checked.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SIZES = (500, 2000)
REPETITIONS = 100
SHARD_SIZE = 10
SEED = 202611080
METHODS = ("Calibrated WGAN-DML", "Empirical-local DML", "Smooth plug-in")
TOLERANCE = 1e-12
OLD_RUN = ROOT / "output/experiments/fixed_h_small_2026-11-08"


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value):
    return float(value) if value not in ("", None) else math.nan


def close(left, right):
    left, right = number(left), number(right)
    return ((math.isnan(left) and math.isnan(right)) or
            (math.isfinite(left) and math.isfinite(right) and
             math.isclose(left, right, rel_tol=0.0, abs_tol=TOLERANCE)))


def boolean(value):
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"Expected a boolean, got {value!r}")


def mean(values):
    return statistics.fmean(values) if values else math.nan


def mcse(values):
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.nan


def wilson(successes, total):
    # Match the experiment's recorded normal critical value, including Wilson
    # intervals, rather than silently changing it during verification.
    z = 1.96
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def row_key(row):
    return int(row["auctions"]), int(row["repetition"]), row["method"]


def independent_statistics(rows):
    """Recompute from status, estimate, target and SE, never saved errors."""
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["auctions"]), row["method"]].append(row)
    summaries, paired, errors = {}, {}, {}
    for key, group in groups.items():
        valid = [row for row in group if row["status"] == "ok"]
        sample_errors = [number(row["estimate"]) - number(row["target"]) for row in valid]
        sample_squares = [error * error for error in sample_errors]
        covered = sum(abs(number(row["estimate"]) - number(row["target"])) <=
                      1.96 * number(row["standard_error"]) for row in valid)
        count, failures = len(group), len(group) - len(valid)
        failure_low, failure_high = wilson(failures, count)
        coverage_low, coverage_high = wilson(covered, count)
        rmse = math.sqrt(mean(sample_squares)) if sample_squares else math.nan
        summaries[key] = dict(
            repetitions=count, valid_repetitions=len(valid), failures=failures,
            failure_rate=failures/count, failure_low=failure_low, failure_high=failure_high,
            bias=mean(sample_errors), bias_mcse=mcse(sample_errors), rmse=rmse,
            rmse_mcse=mcse(sample_squares)/(2*rmse) if rmse > 0 else math.nan,
            coverage=covered/count, coverage_low=coverage_low, coverage_high=coverage_high,
            conditional_coverage=covered/len(valid) if valid else math.nan,
            mean_interval_length=mean([3.92 * number(row["standard_error"]) for row in valid]),
            failure_reasons=dict(Counter(row["failure_reason"] for row in group if row["status"] != "ok")),
        )
        for row in valid:
            errors[row_key(row)] = number(row["estimate"]) - number(row["target"])
    for size in sorted({int(row["auctions"]) for row in rows}):
        for reference in METHODS[1:]:
            differences = [errors[size, rep, METHODS[0]]**2 - errors[size, rep, reference]**2
                           for rep in range(1, REPETITIONS + 1)
                           if (size, rep, METHODS[0]) in errors and (size, rep, reference) in errors]
            paired[size, f"calibrated minus {reference}"] = dict(
                valid_pairs=len(differences), mse_difference=mean(differences),
                paired_mcse=mcse(differences))
    return summaries, paired


def seed_records(base_seed, repetitions):
    records = []
    for size in SIZES:
        for rep in range(repetitions):
            base = base_seed + size * 100 + rep
            records.append(dict(auctions=size, repetition=rep+1, data_seed=base,
                                fold_seed=base+31415926,
                                gan_seeds=[base+1000000+(k+1)*10000 for k in range(3)],
                                generation_seeds=[base+1000000+(k+1)*10000+102 for k in range(3)]))
    return records


def flatten_seeds(records):
    return [value for row in records for value in
            [row["data_seed"], row["fold_seed"], *row["gan_seeds"], *row["generation_seeds"]]]


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class Audit:
    def __init__(self):
        self.checks = {}
        self.failures = []

    def check(self, name, condition, detail=""):
        result = bool(condition)
        self.checks[name] = result
        if not result:
            self.failures.append(dict(check=name, detail=str(detail)))
        return result


def verify(run_dir):
    audit = Audit()
    details = {}
    manifest = read_json(run_dir / "run_manifest.json")
    audit.check("master_completed", manifest.get("completed") is True)
    audit.check("master_old_results_preserved", manifest.get("old_results_preserved") is True)
    audit.check("master_design", manifest.get("base_seed") == SEED and
                manifest.get("sample_sizes") == list(SIZES) and
                manifest.get("repetitions_per_size") == REPETITIONS and
                manifest.get("shard_size") == SHARD_SIZE)

    source = run_dir / "source_snapshot/code"
    source_hashes = manifest["source_sha256"]
    audit.check("snapshot_file_set", {path.name for path in source.glob("*.py")} == set(source_hashes))
    audit.check("snapshot_hashes", all((source/name).is_file() and sha256(source/name) == digest
                                      for name, digest in source_hashes.items()))

    expected_plan = seed_records(SEED, REPETITIONS)
    actual_plan = read_json(run_dir / "seed_plan.json")
    audit.check("complete_seed_plan", actual_plan == expected_plan)
    seeds = flatten_seeds(expected_plan)
    audit.check("all_seed_roles_unique", len(seeds) == len(set(seeds)) == 1600)
    audit.check("no_previous_seed_overlap", not set(seeds).intersection(flatten_seeds(seed_records(20261108, 10))))
    plan_index = {(row["auctions"], row["repetition"]): row for row in expected_plan}

    expected_paths = [run_dir/f"n{size}"/f"part_{start+1:03d}_{start+SHARD_SIZE:03d}"/
                      "protocol.json" for size in SIZES for start in range(0, REPETITIONS, SHARD_SIZE)]
    audit.check("exactly_twenty_shards", set(run_dir.glob("n*/part_*/protocol.json")) == set(expected_paths))
    protocols, shard_rows = [], []
    configuration = None
    for path in expected_paths:
        protocol = read_json(path)
        protocols.append(protocol)
        label = str(path.parent.relative_to(run_dir))
        size = int(path.parent.parent.name[1:])
        start, end = (int(part) for part in path.parent.name.removeprefix("part_").split("_"))
        config = dict(protocol["configuration"])
        audit.check(label+":completed", protocol.get("completed") is True)
        audit.check(label+":range", protocol.get("repetition_start") == start and
                    protocol.get("repetition_end") == end and end-start+1 == SHARD_SIZE)
        audit.check(label+":design", config.pop("sample_sizes") == [size] and
                    config.get("repetitions") == SHARD_SIZE and config.get("seed") == SEED and
                    protocol.get("methods") == list(METHODS) and
                    protocol.get("algorithm_version") == "fixed_h_unique_root_v2" and
                    protocol.get("target") == "theta_0,h")
        if configuration is None:
            configuration = config
        audit.check(label+":configuration", config == configuration)
        audit.check(label+":source_hashes", set(protocol.get("source_sha256", {})) ==
                    {"fixed_h_small_simulation.py", "calibrated_wgan_pilot.py",
                     "orthogonal_wgan_dml.py", "wgan_gp_baseline.py"} and
                    all(source_hashes.get(name) == digest for name, digest in protocol["source_sha256"].items()))
        local_rows = read_csv(path.parent/"raw_results.csv")
        expected_keys = {(size, rep, method) for rep in range(start, end+1) for method in METHODS}
        audit.check(label+":complete_rows", len(local_rows) == 30 and
                    {row_key(row) for row in local_rows} == expected_keys and protocol.get("rows") == 30)
        shard_rows.extend(local_rows)
    audit.check("common_target", all(close(p["regularized_target"], protocols[0]["regularized_target"])
                                     for p in protocols))
    saved_protocols = read_json(run_dir/"summary/protocols.json")
    protocol_key = lambda p: (p["configuration"]["sample_sizes"][0], p["repetition_start"])
    audit.check("aggregate_protocols_match_shards", len(saved_protocols) == 20 and
                sorted(saved_protocols, key=protocol_key) == sorted(protocols, key=protocol_key))

    rows = read_csv(run_dir/"summary/raw_results.csv")
    expected_keys = {(size, rep, method) for size in SIZES for rep in range(1, REPETITIONS+1) for method in METHODS}
    keys = [row_key(row) for row in rows]
    audit.check("exactly_600_rows", len(rows) == 600)
    audit.check("unique_complete_repetition_keys", len(keys) == len(set(keys)) and set(keys) == expected_keys)
    audit.check("100_per_sample_size_and_method", Counter((key[0], key[2]) for key in keys) ==
                Counter({(size, method): 100 for size in SIZES for method in METHODS}))
    shard_index = {row_key(row): row for row in shard_rows}
    same_rows = len(shard_rows) == len(rows) and set(shard_index) == set(keys)
    numeric_fields = ("estimate", "standard_error", "target", "error", "squared_error", "interval_length",
                      "roots", "jacobian", "score_residual", "wgan_weight", "estimation_seconds",
                      "training_seconds", "data_seed", "fold_seed")
    for row in rows:
        original = shard_index.get(row_key(row))
        same_rows = same_rows and original is not None and all(close(row[name], original[name]) for name in numeric_fields)
        same_rows = same_rows and original is not None and all(row[name] == original[name] for name in
                                                              ("status", "failure_reason", "covers_target"))
    audit.check("aggregate_preserves_all_shard_rows", same_rows)
    audit.check("paired_data_and_fold_seeds", all(
        int(row["data_seed"]) == plan_index[row_key(row)[:2]]["data_seed"] and
        int(row["fold_seed"]) == plan_index[row_key(row)[:2]]["fold_seed"] for row in rows))

    raw_valid = True
    for row in rows:
        valid = row["status"] == "ok"
        estimate, target, se = (number(row[name]) for name in ("estimate", "target", "standard_error"))
        error = estimate-target if valid else math.nan
        raw_valid = raw_valid and close(target, protocols[0]["regularized_target"])
        raw_valid = raw_valid and (not valid or (math.isfinite(estimate) and math.isfinite(se) and se > 0))
        raw_valid = raw_valid and close(row["error"], error) and close(row["squared_error"], error*error)
        raw_valid = raw_valid and boolean(row["covers_target"]) == bool(valid and abs(error) <= 1.96*se)
        raw_valid = raw_valid and close(row["interval_length"], 3.92*se if valid else math.nan)
        raw_valid = raw_valid and (valid or bool(row["failure_reason"]))
    audit.check("raw_metrics_recomputed_and_failures_retained", raw_valid)

    computed, computed_paired = independent_statistics(rows)
    saved = read_csv(run_dir/"summary/summary.csv")
    saved_keys = [(int(row["auctions"]), row["method"]) for row in saved]
    audit.check("six_summary_rows", len(saved) == 6 and set(saved_keys) == set(computed))
    summary_match = len(saved) == len(computed)
    differences = []
    for row in saved:
        key = int(row["auctions"]), row["method"]
        expected = computed.get(key)
        if expected is None:
            summary_match = False
            continue
        for name, value in expected.items():
            matched = json.loads(row[name]) == value if name == "failure_reasons" else close(row[name], value)
            if not matched:
                differences.append(dict(auctions=key[0], method=key[1], field=name,
                                        saved=row[name], recomputed=value))
            summary_match = summary_match and matched
    audit.check("summary_metrics_match_independent_recalculation", summary_match, differences)
    saved_paired = read_csv(run_dir/"summary/paired_comparisons.csv")
    paired_keys = [(int(row["auctions"]), row["comparison"]) for row in saved_paired]
    audit.check("four_paired_comparison_rows", len(saved_paired) == 4 and set(paired_keys) == set(computed_paired))
    audit.check("paired_mse_and_mcse_match", len(saved_paired) == len(computed_paired) and all(
        key in computed_paired and all(close(row[name], value) for name, value in computed_paired[key].items())
        for row, key in zip(saved_paired, paired_keys)))

    preservation = read_json(run_dir/"previous_run_sha256.json")
    old_protocol = read_json(OLD_RUN/"n500/protocol.json")
    old_configuration = {name: value for name, value in old_protocol["configuration"].items()
                         if name not in ("sample_sizes", "seed")}
    current_configuration = {name: value for name, value in configuration.items() if name != "seed"}
    audit.check("settings_unchanged_from_previous_run", current_configuration == old_configuration)
    audit.check("core_estimators_unchanged_from_previous_run", all(
        source_hashes.get(name) == old_protocol["source_sha256"].get(name)
        for name in ("calibrated_wgan_pilot.py", "orthogonal_wgan_dml.py", "wgan_gp_baseline.py")))
    old_files = {str(path.relative_to(OLD_RUN)): path for path in OLD_RUN.rglob("*") if path.is_file()}
    audit.check("previous_run_file_set_unchanged", set(old_files) == set(preservation))
    changed = [name for name, digest in preservation.items()
               if name not in old_files or sha256(old_files[name]) != digest]
    audit.check("previous_run_hashes_unchanged", not changed, changed)
    details.update(total_rows=len(rows), shard_count=len(protocols), seed_count=len(seeds),
                   previous_files_checked=len(preservation), previous_run=str(OLD_RUN),
                   independent_summary=[dict(auctions=size, method=method, **values)
                                        for (size, method), values in sorted(computed.items())],
                   independent_paired=[dict(auctions=size, comparison=label, **values)
                                       for (size, label), values in sorted(computed_paired.items())])
    return audit, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    destination = run_dir/"summary/verification.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing verification report: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Completed summary directory does not exist: {destination.parent}")
    try:
        audit, details = verify(run_dir)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        audit, details = Audit(), {}
        audit.check("verification_inputs_readable_and_valid", False, f"{type(exc).__name__}: {exc}")
    report = dict(verified_utc=datetime.now(timezone.utc).isoformat(), run_dir=str(run_dir),
                  verifier_sha256=sha256(Path(__file__).resolve()), absolute_tolerance=TOLERANCE,
                  passed=bool(audit.checks) and all(audit.checks.values()),
                  checks=audit.checks, failures=audit.failures, **details)
    with destination.open("x") as handle:
        json.dump(json_safe(report), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(audit.checks)} checks; {destination}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
