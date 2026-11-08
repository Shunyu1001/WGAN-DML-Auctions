"""Small preregistered fixed-h comparison using the revised acceptance rules.

Only calibrated WGAN, empirical-local DML, and the same-kernel full-sample
plug-in are evaluated. No historical results or manuscript tables are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch

from baseline_direct_inversion import SimulationConfig, valuation_distribution
from calibrated_wgan_pilot import (
    CalibrationConfig, FoldMoments, calibration_weight, estimate,
    synthetic_for_fold, wilson,
)
from crossfit_wgan_pilot import make_folds
from orthogonal_wgan_dml import DMLConfig, true_regularized_target

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("Calibrated WGAN-DML", "Empirical-local DML", "Smooth plug-in")


class CachedMoments(FoldMoments):
    """Memoize exact nuisance evaluations; no interpolation or approximation."""
    def __init__(self, *args):
        super().__init__(*args)
        self.memo = {}

    def nuisance(self, theta, weight):
        points = np.atleast_1d(np.asarray(theta, dtype=float))
        key = (points.shape, points.tobytes(), float(weight))
        if key not in self.memo:
            self.memo[key] = super().nuisance(theta, weight)
        return self.memo[key]


def run_cell(size: int, reps: int, seed: int, output: Path, start_rep: int = 0):
    if size < 1 or reps < 1 or start_rep < 0:
        raise ValueError("size and reps must be positive, and start_rep nonnegative")
    if output.exists():
        raise FileExistsError(f"Refusing to reuse an existing result directory: {output}")
    output.mkdir(parents=True)
    cache = output / "synthetic_cache"
    cache.mkdir()
    config = CalibrationConfig(sample_sizes=(size,), repetitions=reps, seed=seed)
    torch.set_num_threads(1)
    distribution = valuation_distribution(SimulationConfig(bidders=config.bidders))
    dc = DMLConfig(bandwidth=config.bandwidth, bidders=config.bidders,
                   theta_lower=config.theta_lower, theta_upper=config.theta_upper)
    target = true_regularized_target(dc)
    target_refined = true_regularized_target(
        DMLConfig(**{**asdict(dc), "target_integration_draws": 2*dc.target_integration_draws}))
    source_files = ["fixed_h_small_simulation.py", "calibrated_wgan_pilot.py",
                    "orthogonal_wgan_dml.py", "wgan_gp_baseline.py"]
    protocol = dict(
        configuration=asdict(config), methods=METHODS, target="theta_0,h",
        regularized_target=target, refined_target=target_refined,
        target_refinement_difference=target_refined-target,
        design="same data, folds, generated maxima; fixed h only; no performance-dependent tuning",
        algorithm_version="fixed_h_unique_root_v2",
        source_sha256={f: hashlib.sha256((ROOT/"code"/f).read_bytes()).hexdigest()
                       for f in source_files},
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                         pandas=pd.__version__,torch=torch.__version__,platform=platform.platform()),
        repetition_start=start_rep + 1, repetition_end=start_rep + reps,
        completed=False,
    )
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    rows=[]
    started=time.perf_counter()
    solver=dict(theta_bounds=(config.theta_lower,config.theta_upper),
                grid_size=config.theta_grid_size,jacobian_floor=config.jacobian_floor,
                cdf_floor=config.cdf_floor,root_tolerance=config.root_tolerance,
                score_tolerance=config.score_tolerance)
    for rep in range(start_rep, start_rep + reps):
        print(f"n={size} repetition={rep+1} (shard {start_rep+1}-{start_rep+reps})",flush=True)
        rng=np.random.default_rng(seed+size*100+rep)
        observed=distribution.rvs(size=(size,config.bidders),random_state=rng).max(axis=1)
        folds=make_folds(size,config.folds,seed+size*100+rep+31415926)
        moments=[]; weights=[]; fitting_seconds=0.
        for k,holdout in enumerate(folds):
            mask=np.ones(size,dtype=bool); mask[holdout]=False
            training=observed[mask]
            t=time.perf_counter()
            synthetic=synthetic_for_fold(training,size,rep,k,config,cache)
            fitting_seconds+=time.perf_counter()-t
            moments.append(CachedMoments(observed[holdout],training,synthetic,
                                         config.bandwidth,config.bidders))
            weights.append(calibration_weight(len(training),config.pseudo_observations))
        full=CachedMoments(observed,observed,np.empty(0),config.bandwidth,config.bidders)
        tasks=[(METHODS[0],moments,weights,False),
               (METHODS[1],moments,[0.]*config.folds,False),
               (METHODS[2],[full],[0.],True)]
        for method,data,weight,plugin in tasks:
            t=time.perf_counter()
            fit=estimate(data,weight,plugin=plugin,**solver)
            valid=fit["status"]=="ok"
            error=fit["estimate"]-target if valid else np.nan
            row=dict(auctions=size,repetition=rep+1,method=method,
                     status=fit["status"],failure_reason=fit["failure_reason"],
                     estimate=fit["estimate"],standard_error=fit["standard_error"],
                     target=target,error=error,squared_error=error*error,
                     covers_target=bool(valid and abs(error)<=1.96*fit["standard_error"]),
                     interval_length=3.92*fit["standard_error"] if valid else np.nan,
                     roots=fit["roots"],jacobian=fit["jacobian"],
                     score_residual=fit["score_residual"],
                     wgan_weight=float(np.mean(weights)) if method==METHODS[0] else 0.,
                     estimation_seconds=time.perf_counter()-t,
                     training_seconds=fitting_seconds if method==METHODS[0] else 0.,
                     data_seed=seed+size*100+rep,
                     fold_seed=seed+size*100+rep+31415926)
            rows.append(row)
            print(f"  {method}: {fit['status']} estimate={fit['estimate']:.6f} "
                  f"se={fit['standard_error']:.6f} reason={fit['failure_reason']}",flush=True)
        pd.DataFrame(rows).to_csv(output/"raw_results.csv",index=False)
        print(f"  cumulative seconds={time.perf_counter()-started:.1f}",flush=True)
    protocol.update(completed=True,elapsed_seconds=time.perf_counter()-started,rows=len(rows))
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")


def aggregate(cells: list[Path], output: Path):
    if output.exists():
        raise FileExistsError(f"Refusing to replace aggregate: {output}")
    output.mkdir(parents=True)
    protocols=[json.loads((p/"protocol.json").read_text()) for p in cells]
    if not all(p["completed"] for p in protocols):
        raise ValueError("Every cell must finish before aggregation")
    raw=pd.concat([pd.read_csv(p/"raw_results.csv",keep_default_na=False,
                              na_values=[""]) for p in cells],ignore_index=True)
    if raw.duplicated(["auctions","repetition","method"]).any():
        raise ValueError("Duplicate simulation rows")
    out=[]
    for (n,method),g in raw.groupby(["auctions","method"],sort=False):
        valid=g.status.eq("ok")
        e=g.loc[valid,"error"].to_numpy(dtype=float)
        succ=int(g.covers_target.astype(bool).sum())
        lo,hi=wilson(succ,len(g)); flo,fhi=wilson(int((~valid).sum()),len(g))
        mse=np.mean(e*e) if len(e) else np.nan
        rmse=np.sqrt(mse)
        rmse_mcse=(np.std(e*e,ddof=1)/np.sqrt(len(e))/(2*rmse)
                   if len(e)>1 and rmse>0 else np.nan)
        out.append(dict(auctions=n,method=method,repetitions=len(g),
                        valid_repetitions=int(valid.sum()),failures=int((~valid).sum()),
                        failure_rate=(~valid).mean(),failure_low=flo,failure_high=fhi,
                        bias=np.mean(e) if len(e) else np.nan,
                        bias_mcse=np.std(e,ddof=1)/np.sqrt(len(e)) if len(e)>1 else np.nan,
                        rmse=rmse,rmse_mcse=rmse_mcse,
                        coverage=succ/len(g),coverage_low=lo,coverage_high=hi,
                        conditional_coverage=succ/int(valid.sum()) if valid.any() else np.nan,
                        mean_interval_length=g.loc[valid,"interval_length"].mean(),
                        failure_reasons=json.dumps(g.loc[~valid,"failure_reason"].value_counts().to_dict()),
                        error_scope="conditional on valid fits; unconditional coverage counts failures as noncoverage"))
    summary=pd.DataFrame(out)
    paired=[]
    for n,g in raw.groupby("auctions"):
        pivot=g.pivot(index="repetition",columns="method",values="squared_error")
        for ref in METHODS[1:]:
            d=(pivot[METHODS[0]]-pivot[ref]).dropna()
            paired.append(dict(auctions=n,comparison=f"calibrated minus {ref}",
                               valid_pairs=len(d),mse_difference=d.mean(),
                               paired_mcse=d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else np.nan))
    raw.to_csv(output/"raw_results.csv",index=False)
    summary.to_csv(output/"summary.csv",index=False)
    pd.DataFrame(paired).to_csv(output/"paired_comparisons.csv",index=False)
    (output/"protocols.json").write_text(json.dumps(protocols,indent=2)+"\n")
    print(summary[["auctions","method","failures","bias","rmse","coverage",
                   "conditional_coverage"]].to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size",type=int)
    p.add_argument("--reps",type=int,default=10)
    p.add_argument("--seed",type=int,default=20261108)
    p.add_argument("--start-rep",type=int,default=0,
                   help="zero-based global repetition offset; seed formulas stay unchanged")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--aggregate",nargs="+",type=Path)
    a=p.parse_args()
    if a.aggregate: aggregate(a.aggregate,a.output_dir)
    elif a.size is None or a.size<1 or a.reps<1 or a.start_rep<0:
        p.error("positive --size and --reps, and nonnegative --start-rep are required")
    else: run_cell(a.size,a.reps,a.seed,a.output_dir,a.start_rep)
