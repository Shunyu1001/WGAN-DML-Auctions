"""Paired coverage diagnosis and score-test inversion; never retrain the WGAN."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import skew, kurtosis

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"output/experiments/fixed_h_100_2026-11-08"
# Import precisely the numerical code used in the completed experiment.
sys.path.insert(0, str(SOURCE/"source_snapshot/code"))
from baseline_direct_inversion import SimulationConfig, valuation_distribution
from calibrated_wgan_pilot import (CalibrationConfig, calibration_weight, estimate,
                                   density_derivative, scores, wilson)
from crossfit_wgan_pilot import make_folds
from fixed_h_small_simulation import CachedMoments, METHODS
from orthogonal_wgan_dml import gaussian_kernel_moments, reserve_moment_and_derivatives
from wgan_gp_baseline import WGANConfig

CRITICAL = 1.96
BOUNDS = (0.45, 1.15)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_statistics(points, moments, weights):
    """Pooled observation-level mean, variance, and signed test statistic."""
    points = np.atleast_1d(np.asarray(points, dtype=float))
    total = sum(len(f.observed) for f in moments)
    first = np.zeros(len(points)); second = np.zeros(len(points))
    for fold, weight in zip(moments, weights):
        a, g = fold.nuisance(points, weight)
        if (not np.all(np.isfinite(a)) or not np.all(np.isfinite(g))
                or np.any(a <= 1e-8) or np.any(a > 1)):
            raise ValueError("invalid_candidate_nuisance")
        m, da, dg = reserve_moment_and_derivatives(points, a, g, fold.bidders)
        z = (points[:, None]-fold.observed[None, :])/fold.h
        psi = (m[:, None] + da[:, None]*(ndtr(z)-a[:, None]) +
               dg[:, None]*(np.exp(-z*z/2)/(np.sqrt(2*np.pi)*fold.h)-g[:, None]))
        first += psi.sum(axis=1); second += np.square(psi).sum(axis=1)
    mean = first/total
    variance = (second-first*first/total)/(total-1)
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("invalid_candidate_variance")
    return mean, variance, np.sqrt(total)*mean/np.sqrt(variance)


def invert_statistic(statistic, root, grid_size=281):
    """Retain every grid-detected component and refine all crossing endpoints."""
    grid = np.unique(np.r_[np.linspace(*BOUNDS, grid_size), root])
    value = np.square(statistic(grid))-CRITICAL**2
    if not np.all(np.isfinite(value)):
        raise ValueError("nonfinite_test_statistic")
    crossings = []
    for j in np.flatnonzero(value[:-1]*value[1:] < 0):
        crossings.append(brentq(lambda x: float(statistic([x])[0]**2-CRITICAL**2),
                                grid[j], grid[j+1], xtol=1e-11))
    crossings += grid[value == 0].tolist()
    knots = sorted(set([*BOUNDS, *crossings]))
    components = []
    for left, right in zip(knots[:-1], knots[1:]):
        if abs(float(statistic([(left+right)/2])[0])) <= CRITICAL:
            if components and abs(components[-1][1]-left) < 1e-9:
                components[-1][1] = right
            else:
                components.append([left, right])
    for point in crossings:
        if not any(left <= point <= right for left, right in components):
            components.append([point, point])
    components.sort()
    if not components or not any(left-1e-9 <= root <= right+1e-9 for left, right in components):
        raise ValueError("inversion_does_not_contain_valid_root")
    return components


def load_replay(size, rep, source=SOURCE):
    config = CalibrationConfig(sample_sizes=(size,), repetitions=10, seed=202611080)
    distribution = valuation_distribution(SimulationConfig(bidders=config.bidders))
    seed = config.seed+size*100+rep
    observed = distribution.rvs(size=(size, config.bidders),
                                random_state=np.random.default_rng(seed)).max(axis=1)
    folds = make_folds(size, config.folds, seed+31415926)
    part = (rep//10)*10
    cache = source/f"n{size}"/f"part_{part+1:03d}_{part+10:03d}"/"synthetic_cache"
    wc = WGANConfig(training_steps=config.training_steps, bidders=config.bidders,
                    generated_values=config.generated_maxima, device="cpu")
    moments = []; weights = []; cache_hashes = {}
    for k, holdout in enumerate(folds):
        mask = np.ones(size, dtype=bool); mask[holdout] = False
        training = observed[mask]
        metadata = json.dumps(dict(wgan=asdict(wc), seed=seed+1000000+(k+1)*10000,
            generated_maxima=config.generated_maxima,
            source_hash=sha(source/"source_snapshot/code/wgan_gp_baseline.py"),
            data_hash=hashlib.sha256(training.tobytes()).hexdigest()), sort_keys=True)
        key = hashlib.sha256(metadata.encode()).hexdigest()[:20]
        path = cache/f"{key}.npz"
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["metadata"]) != metadata:
                raise ValueError(f"Cache metadata mismatch: {path}")
            synthetic = stored["synthetic"].copy()
        if synthetic.shape != (config.generated_maxima,):
            raise ValueError("Unexpected synthetic sample shape")
        cache_hashes[str(path.relative_to(source))] = sha(path)
        moments.append(CachedMoments(observed[holdout], training, synthetic,
                                      config.bandwidth, config.bidders))
        weights.append(calibration_weight(len(training), config.pseudo_observations))
    full = CachedMoments(observed, observed, np.empty(0), config.bandwidth, config.bidders)
    return config, observed, [(METHODS[0], moments, weights, False),
                            (METHODS[1], moments, [0.]*config.folds, False),
                            (METHODS[2], [full], [0.], True)], cache_hashes


def diagnose_raw(raw):
    output = []
    for (n, method), g in raw.groupby(["auctions", "method"], sort=False):
        g = g[g.status.eq("ok")]
        e = g.error.to_numpy(); se = g.standard_error.to_numpy(); z = e/se
        sd = np.std(e, ddof=1); bias = np.mean(e)
        output.append(dict(auctions=n, method=method, repetitions=len(g), bias=bias,
            error_sd=sd, mean_se=se.mean(), rms_se=np.sqrt(np.mean(se*se)),
            mean_se_over_sd=se.mean()/sd, rms_se_over_sd=np.sqrt(np.mean(se*se))/sd,
            error_skew=skew(e, bias=False), error_excess_kurtosis=kurtosis(e, bias=False),
            z_mean=z.mean(), z_sd=np.std(z, ddof=1), z_skew=skew(z, bias=False),
            z_q025=np.quantile(z,.025), z_q975=np.quantile(z,.975),
            lower_misses=int((z < -CRITICAL).sum()), upper_misses=int((z > CRITICAL).sum()),
            coverage=np.mean(abs(z)<=CRITICAL),
            corr_error_se=np.corrcoef(e,se)[0,1],
            corr_error_abs_jacobian=np.corrcoef(e,abs(g.jacobian))[0,1],
            corr_error_score_sd=np.corrcoef(e,se*np.sqrt(n)*abs(g.jacobian))[0,1],
            oracle_demeaned_coverage=np.mean(abs(e-bias)<=CRITICAL*se),
            oracle_constant_sd_coverage=np.mean(abs(e)<=CRITICAL*sd)))
    return pd.DataFrame(output)


def run_cell(size, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    original = pd.read_csv(SOURCE/"summary/raw_results.csv", keep_default_na=False)
    original = original[original.auctions.eq(size)].set_index(["repetition", "method"])
    target = float(original.target.iloc[0])
    dist = valuation_distribution(SimulationConfig(bidders=5))
    pop_sample = dist.ppf(((np.arange(100000)+.5)/100000)**.2)
    pop_a, pop_g = gaussian_kernel_moments(target, pop_sample, .15)
    rows = []; cache_hashes = {}; started = time.perf_counter()
    for rep in range(100):
        config, observed, tasks, hashes = load_replay(size, rep)
        cache_hashes.update(hashes)
        oracle = scores(target, observed, pop_a[0], pop_g[0], .15, 5)
        oracle_t = np.sqrt(size)*oracle.mean()/np.std(oracle, ddof=1)
        for method, moments, weights, plugin in tasks:
            fit = estimate(moments, weights, plugin=plugin,
                theta_bounds=BOUNDS, grid_size=141, jacobian_floor=1e-6, cdf_floor=1e-8)
            old = original.loc[(rep+1,method)]
            if fit["status"] != old.status or any(abs(fit[k]-old[k])>1e-12
                    for k in ("estimate", "standard_error", "jacobian")):
                raise AssertionError(f"Replay mismatch n={size} rep={rep+1} {method}")
            row = dict(auctions=size,repetition=rep+1,method=method,target=target,
                estimate=fit["estimate"],standard_error=fit["standard_error"],
                status=fit["status"],error=old.error,jacobian=fit["jacobian"],
                wald_coverage=bool(old.covers_target),wald_length=3.92*fit["standard_error"],
                oracle_score_t=oracle_t,replay_matched=True)
            try:
                mean, variance, target_t = score_statistics([target], moments, weights)
                row.update(score_mean_at_target=mean[0],score_sd_at_target=np.sqrt(variance[0]),
                           score_t_at_target=target_t[0])
                fn = lambda t: score_statistics(t,moments,weights)[2]
                components = invert_statistic(fn,fit["estimate"])
                coverage = abs(target_t[0]) <= CRITICAL
                membership = any(l-1e-9<=target<=r+1e-9 for l,r in components)
                if coverage != membership:
                    raise AssertionError("Coverage and component membership disagree")
                refinement = np.nan
                if rep in (0,49,99):
                    fine = invert_statistic(fn,fit["estimate"],561)
                    if np.shape(fine) != np.shape(components):
                        raise AssertionError("Refined grid has different topology")
                    refinement = float(np.max(abs(np.array(fine)-np.array(components))))
                    if refinement > 1e-8:
                        raise AssertionError("Inversion endpoints depend on grid")
                row.update(inversion_status="ok",inversion_failure_reason="",
                    score_coverage=bool(coverage),score_length=sum(r-l for l,r in components),
                    score_components=json.dumps(components),component_count=len(components),
                    boundary_truncated=any(abs(l-BOUNDS[0])<1e-9 or abs(r-BOUNDS[1])<1e-9
                                           for l,r in components),
                    refinement_max_difference=refinement,
                    score_lower_miss=all(target>r for l,r in components),
                    score_upper_miss=all(target<l for l,r in components))
            except ValueError as exc:
                row.update(inversion_status="failed",inversion_failure_reason=str(exc),
                    score_coverage=False,score_length=np.nan,score_components="[]",component_count=0,
                    boundary_truncated=False,refinement_max_difference=np.nan,
                    score_lower_miss=False,score_upper_miss=False)
            rows.append(row)
        pd.DataFrame(rows).to_csv(output/"raw_results.csv",index=False)
        print(f"n={size} replay {rep+1}/100; elapsed={time.perf_counter()-started:.1f}s",flush=True)
    (output/"cache_sha256.json").write_text(json.dumps(cache_hashes,indent=2)+"\n")
    (output/"completed.json").write_text(json.dumps(dict(completed=True,rows=len(rows),
        elapsed_seconds=time.perf_counter()-started),indent=2)+"\n")


def aggregate(output):
    raw = pd.concat([pd.read_csv(output/f"n{n}/raw_results.csv",keep_default_na=False,
                                na_values=[""]) for n in (500,2000)],ignore_index=True)
    original = pd.read_csv(SOURCE/"summary/raw_results.csv")
    diagnose_raw(original).to_csv(output/"diagnosis_summary.csv",index=False)
    result=[]
    for (n,method),g in raw.groupby(["auctions","method"],sort=False):
        for interval,coverage,length in [("Wald",g.wald_coverage,g.wald_length),
                ("Score inversion",g.score_coverage,g.score_length)]:
            count=int(coverage.sum()); lo,hi=wilson(count,len(g))
            result.append(dict(auctions=n,method=method,interval=interval,repetitions=len(g),
                point_failures=int((g.status!="ok").sum()),
                interval_failures=0 if interval=="Wald" else int((g.inversion_status!="ok").sum()),
                bias=g.error.mean(),rmse=np.sqrt(np.mean(g.error**2)),coverage=count/len(g),
                coverage_low=lo,coverage_high=hi,mean_length=length.mean(),
                disconnected_sets=0 if interval=="Wald" else int((g.component_count>1).sum()),
                boundary_truncated=0 if interval=="Wald" else int(g.boundary_truncated.sum())))
    raw.to_csv(output/"raw_results.csv",index=False)
    pd.DataFrame(result).to_csv(output/"comparison_summary.csv",index=False)
    tests=[]
    for (n,method),g in raw.groupby(["auctions","method"],sort=False):
        for name,col in [("Wald",g.error/g.standard_error),("Score at truth",g.score_t_at_target),
                         ("Oracle score at truth",g.oracle_score_t)]:
            tests.append(dict(auctions=n,method=method,statistic=name,mean=col.mean(),
                sd=col.std(ddof=1),skew=skew(col,bias=False),q025=col.quantile(.025),
                q975=col.quantile(.975),rejection=np.mean(abs(col)>CRITICAL)))
    pd.DataFrame(tests).to_csv(output/"test_statistics.csv",index=False)
    print(pd.DataFrame(result).to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size",type=int,choices=[500,2000])
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--aggregate",action="store_true")
    args=parser.parse_args()
    if args.aggregate: aggregate(args.output_dir)
    elif args.size: run_cell(args.size,args.output_dir)
    else: parser.error("--size or --aggregate required")
