"""Independently audit the fresh-seed Wald versus score-inversion validation.

Only Python standard-library calculations are used. The runner calls this after
aggregation and before setting its own completed flag. The sole output is a new
summary/verification.json; an existing verification is never overwritten.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

SEED = 2026110800
SIZES = (500, 2000)
REPS = 100
METHODS = ("Calibrated WGAN-DML", "Empirical-local DML", "Smooth plug-in")
INTERVALS = ("Wald", "Score inversion")
TOL = 1e-12
CRITICAL = 1.96
BOUNDS = (0.45, 1.15)
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


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


def num(value):
    return float(value) if value not in (None, "") else math.nan


def close(a, b, tol=TOL):
    a, b = num(a), num(b)
    return ((math.isnan(a) and math.isnan(b)) or
            (math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol))


def truth(value):
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"Not a serialized Boolean: {value!r}")


def mean(values):
    return statistics.fmean(values) if values else math.nan


def mcse(values):
    return statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else math.nan


def wilson(successes, total):
    p = successes/total
    d = 1+CRITICAL**2/total
    center = (p+CRITICAL**2/(2*total))/d
    half = CRITICAL*math.sqrt(p*(1-p)/total+CRITICAL**2/(4*total**2))/d
    return max(0.0, center-half), min(1.0, center+half)


def key(row):
    return int(row["auctions"]), int(row["repetition"]), row["method"]


def seed_plan(base, reps):
    return [dict(auctions=n, repetition=r+1, data_seed=base+n*100+r,
                 fold_seed=base+n*100+r+31415926,
                 gan_seeds=[base+n*100+r+1000000+(k+1)*10000 for k in range(3)],
                 generation_seeds=[base+n*100+r+1000000+(k+1)*10000+102 for k in range(3)])
            for n in SIZES for r in range(reps)]


def all_seeds(plan):
    return [v for row in plan for v in
            [row["data_seed"], row["fold_seed"], *row["gan_seeds"], *row["generation_seeds"]]]


class Audit:
    def __init__(self):
        self.checks = {}
        self.failures = []

    def check(self, name, condition, detail=""):
        passed = bool(condition)
        self.checks[name] = passed
        if not passed:
            self.failures.append(dict(check=name, detail=str(detail)))
        return passed


def validate_rows(rows, audit, expected_target):
    """Check the confidence-set identity independently from saved coverage."""
    points_valid = True
    inversions_valid = True
    refinements = []
    for row in rows:
        n, rep, _ = key(row)
        ok = row["status"] == "ok"
        inv = row["inversion_status"] == "ok"
        est, se, target = [num(row[name]) for name in ("estimate", "standard_error", "target")]
        points_valid &= row["status"] in ("ok", "failed") and close(target, expected_target)
        if ok:
            points_valid &= (math.isfinite(est) and BOUNDS[0] < est < BOUNDS[1] and
                             math.isfinite(se) and se > 0 and
                             math.isfinite(num(row["jacobian"])) and num(row["jacobian"]) < -1e-6)
            points_valid &= close(row["error"], est-target) and close(row["wald_length"], 3.92*se)
        else:
            points_valid &= all(math.isnan(num(row[f])) for f in
                                ("estimate", "standard_error", "error", "wald_length"))
        points_valid &= truth(row["wald_coverage"]) == bool(ok and abs(est-target) <= CRITICAL*se)

        inversions_valid &= row["inversion_status"] in ("ok", "failed")
        components = json.loads(row["score_components"])
        if inv:
            inversions_valid &= ok and bool(components) and row["inversion_failure_reason"] == ""
            t = num(row["score_t_at_target"])
            inversions_valid &= math.isfinite(t)
            ordered = True
            last_right = -math.inf
            for left, right in components:
                ordered &= (math.isfinite(left) and math.isfinite(right) and
                            BOUNDS[0]-1e-9 <= left <= right <= BOUNDS[1]+1e-9 and
                            left >= last_right-1e-9)
                last_right = right
            inversions_valid &= ordered and int(row["component_count"]) == len(components)
            inversions_valid &= close(row["score_length"], sum(b-a for a,b in components))
            inversions_valid &= any(a-1e-9 <= est <= b+1e-9 for a,b in components)
            membership = any(a-1e-9 <= target <= b+1e-9 for a,b in components)
            inversions_valid &= truth(row["score_coverage"]) == (abs(t) <= CRITICAL)
            inversions_valid &= truth(row["score_coverage"]) == membership
            boundary = any(abs(a-BOUNDS[0]) < 1e-9 or abs(b-BOUNDS[1]) < 1e-9 for a,b in components)
            inversions_valid &= truth(row["boundary_truncated"]) == boundary
        else:
            inversions_valid &= bool(row["inversion_failure_reason"])
            inversions_valid &= not truth(row["score_coverage"])
            inversions_valid &= math.isnan(num(row["score_length"]))
            inversions_valid &= components == [] and int(row["component_count"]) == 0
        refinement = num(row["refinement_max_difference"])
        if rep in (1, 50, 100) and inv:
            refinements.append(refinement)
            inversions_valid &= math.isfinite(refinement) and refinement <= 1e-8
        elif math.isfinite(refinement):
            inversions_valid &= refinement <= 1e-8
    audit.check("point_estimates_and_wald_flags_valid", points_valid)
    audit.check("score_sets_and_coverage_identities_valid", inversions_valid)
    audit.check("selected_refinements_pass", all(math.isfinite(v) and v <= 1e-8 for v in refinements))
    return refinements


def independent_statistics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)[0], row["method"]].append(row)
    summaries = {}
    paired = {}
    for (n, method), group in groups.items():
        valid_points = [r["status"] == "ok" for r in group]
        valid_score = [r["status"] == "ok" and r["inversion_status"] == "ok" for r in group]
        point_failures = sum(not x for x in valid_points)
        for interval, valid, flag, length in (
            ("Wald", valid_points, "wald_coverage", "wald_length"),
            ("Score inversion", valid_score, "score_coverage", "score_length")):
            count = sum(truth(r[flag]) for r in group)
            low, high = wilson(count, len(group))
            total_invalid = sum(not x for x in valid)
            lengths = [num(r[length]) for r, v in zip(group, valid) if v]
            summaries[n, method, interval] = dict(
                repetitions=len(group), point_failures=point_failures,
                interval_failures=total_invalid,
                additional_interval_failures=total_invalid-point_failures,
                failure_rate=total_invalid/len(group), coverage=count/len(group),
                coverage_low=low, coverage_high=high, mean_length=mean(lengths))
        changes = [int(truth(r["score_coverage"]))-int(truth(r["wald_coverage"])) for r in group]
        length_diffs = [num(r["score_length"])-num(r["wald_length"])
                        for r, p, s in zip(group, valid_points, valid_score) if p and s]
        paired[n, method] = dict(gained=sum(c == 1 for c in changes), lost=sum(c == -1 for c in changes),
                                 coverage_change=mean(changes), paired_mcse=mcse(changes),
                                 valid_length_pairs=len(length_diffs), mean_length_change=mean(length_diffs))
    return summaries, paired


def function_asts(path):
    source = path.read_text()
    tree = ast.parse(source)
    names = ("score_statistics", "invert_statistic")
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names}


def verify(run_dir, project_root):
    audit = Audit()
    manifest = read_json(run_dir/"run_manifest.json")
    audit.check("master_design", manifest.get("base_seed") == SEED and
                manifest.get("sample_sizes") == list(SIZES) and manifest.get("repetitions_per_size") == REPS)
    audit.check("prior_results_preserved_flag", manifest.get("prior_results_preserved") is True)
    # completed=False is expected: successful verification is a prerequisite.
    snapshot = run_dir/"source_snapshot/code"
    snapshot_hashes = manifest["source_sha256"]
    audit.check("snapshot_hashes", all((snapshot/name).is_file() and sha256(snapshot/name) == digest
                                      for name, digest in snapshot_hashes.items()))
    audit.check("snapshot_file_set", {str(p.relative_to(snapshot)) for p in snapshot.rglob("*")
                                      if p.is_file() and "__pycache__" not in p.parts} == set(snapshot_hashes))
    previous_snapshot = project_root/"output/experiments/fixed_h_coverage_diagnostic_2026-11-08/source_snapshot/code"
    core_names = ("baseline_direct_inversion.py", "calibrated_wgan_pilot.py", "crossfit_wgan_pilot.py",
                  "orthogonal_wgan_dml.py", "wgan_gp_baseline.py")
    audit.check("frozen_core_hashes", all((snapshot/name).is_file() and
                sha256(snapshot/name) == sha256(previous_snapshot/name) for name in core_names))
    original_functions = function_asts(previous_snapshot/"diagnose_fixed_h_coverage.py")
    found = defaultdict(list)
    for path in snapshot.glob("*.py"):
        for name, definition in function_asts(path).items():
            found[name].append((path.name, definition))
    audit.check("frozen_inversion_function_semantics", all(found[name] and
                all(definition == original_functions[name] for _, definition in found[name])
                for name in original_functions))

    expected_plan = seed_plan(SEED, REPS)
    supplied_plan = read_json(run_dir/"seed_plan.json")
    order = lambda r: (r["auctions"], r["repetition"])
    audit.check("exact_seed_plan", sorted(supplied_plan, key=order) == sorted(expected_plan, key=order))
    seeds = all_seeds(expected_plan)
    audit.check("1600_distinct_new_seeds", len(seeds) == len(set(seeds)) == 1600)
    audit.check("no_overlap_prior_10_or_100", not set(seeds).intersection(
        all_seeds(seed_plan(20261108, 10))+all_seeds(seed_plan(202611080, 100))))
    plan_index = {(r["auctions"],r["repetition"]):r for r in expected_plan}

    paths = [run_dir/f"n{n}"/f"part_{start+1:03d}_{start+10:03d}"/"protocol.json"
             for n in SIZES for start in range(0,100,10)]
    audit.check("twenty_exact_shards", set(paths) == set(run_dir.glob("n*/part_*/protocol.json")))
    original_config = read_json(project_root/"output/experiments/fixed_h_100_2026-11-08/n500/part_001_010/protocol.json")["configuration"]
    shard_rows = []
    protocols = []
    for path in paths:
        p = read_json(path); protocols.append(p)
        label = str(path.parent.relative_to(run_dir))
        n = int(path.parent.parent.name[1:])
        start, end = map(int, path.parent.name.removeprefix("part_").split("_"))
        config = p["configuration"]
        audit.check(label+":complete", p.get("completed") is True and p.get("rows") == 30)
        audit.check(label+":range", p.get("repetition_start") == start and p.get("repetition_end") == end)
        audit.check(label+":frozen_configuration", config.get("sample_sizes") == [n] and config.get("seed") == SEED and
                    all(config.get(k) == v for k,v in original_config.items()
                        if k not in ("sample_sizes","seed","repetitions")))
        local = read_csv(path.parent/"raw_results.csv")
        audit.check(label+":rows", len(local) == 30 and {key(r) for r in local} ==
                    {(n,r,m) for r in range(start,end+1) for m in METHODS})
        shard_rows.extend(local)

    rows = read_csv(run_dir/"summary/raw_results.csv")
    expected_keys = {(n,r,m) for n in SIZES for r in range(1,REPS+1) for m in METHODS}
    keys = [key(r) for r in rows]
    audit.check("600_complete_unique_rows", len(rows) == len(keys) == len(set(keys)) == 600 and set(keys) == expected_keys)
    audit.check("100_per_size_and_method", Counter((n,m) for n,_,m in keys) ==
                Counter({(n,m):100 for n in SIZES for m in METHODS}))
    audit.check("row_seeds_match_plan", all(int(r["data_seed"]) == plan_index[key(r)[:2]]["data_seed"] and
                int(r["fold_seed"]) == plan_index[key(r)[:2]]["fold_seed"] for r in rows))
    local_index = {key(r):r for r in shard_rows}
    same = len(local_index) == 600 and len(shard_rows) == 600
    string_fields = {"method","status","failure_reason","inversion_status","inversion_failure_reason"}
    bool_fields = {"wald_coverage","score_coverage","boundary_truncated"}
    for row in rows:
        local = local_index.get(key(row), {})
        same &= set(row) == set(local)
        for field, value in row.items():
            if field not in local:
                same = False
            elif field in string_fields:
                same &= value == local[field]
            elif field in bool_fields:
                same &= truth(value) == truth(local[field])
            elif field == "score_components":
                same &= json.loads(value) == json.loads(local[field])
            else:
                same &= close(value, local[field])
    audit.check("aggregate_preserves_shard_rows", same)
    expected_target = 0.8318283572057762
    refinements = validate_rows(rows, audit, expected_target)

    calculated, paired = independent_statistics(rows)
    saved = read_csv(run_dir/"summary/summary.csv")
    summary_keys = [(int(r["auctions"]),r["method"],r["interval"]) for r in saved]
    audit.check("12_unique_summary_rows", len(saved) == len(set(summary_keys)) == 12 and set(summary_keys) == set(calculated))
    audit.check("summary_metrics_recomputed", all(
        k in calculated and all(close(r[name],value) for name,value in calculated[k].items())
        for r,k in zip(saved,summary_keys)))
    saved_paired = read_csv(run_dir/"summary/paired_comparisons.csv")
    paired_keys = [(int(r["auctions"]),r["method"]) for r in saved_paired]
    audit.check("6_unique_paired_rows", len(saved_paired) == len(set(paired_keys)) == 6 and set(paired_keys) == set(paired))
    audit.check("paired_metrics_recomputed", all(
        k in paired and all(close(r[name],value) for name,value in paired[k].items())
        for r,k in zip(saved_paired,paired_keys)))

    previous = read_json(run_dir/"previous_results_sha256.json")
    changed = [name for name,digest in previous.items()
               if not (project_root/name).is_file() or sha256(project_root/name) != digest]
    audit.check("all_prior_file_hashes_unchanged", bool(previous) and not changed, changed)
    details = dict(total_rows=len(rows),shard_count=len(protocols),seed_count=len(seeds),
                   previous_files_checked=len(previous),selected_refinements=len(refinements),
                   maximum_refinement_difference=max(refinements) if refinements else None,
                   independent_summary=[dict(auctions=n,method=m,interval=i,**v) for (n,m,i),v in calculated.items()],
                   independent_paired=[dict(auctions=n,method=m,**v) for (n,m),v in paired.items()])
    return audit, details


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [clean(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--project-root",type=Path,default=DEFAULT_ROOT)
    args = parser.parse_args()
    destination = args.run_dir/"summary/verification.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError("Aggregation must finish before verification")
    try:
        audit, details = verify(args.run_dir.resolve(),args.project_root.resolve())
    except (OSError,ValueError,KeyError,TypeError,ZeroDivisionError,SyntaxError) as exc:
        audit, details = Audit(), {}
        audit.check("verification_inputs_readable",False,f"{type(exc).__name__}: {exc}")
    report = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
                  passed=bool(audit.checks) and all(audit.checks.values()),checks=audit.checks,
                  failures=audit.failures,absolute_tolerance=TOL,verifier_sha256=sha256(Path(__file__)),**details)
    with destination.open("x") as handle:
        json.dump(clean(report),handle,indent=2,allow_nan=False)
        handle.write("\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(audit.checks)} checks; {destination}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
