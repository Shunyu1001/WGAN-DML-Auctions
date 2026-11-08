"""Fresh-seed validation of the frozen Wald versus score-inversion procedure."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

import calibrated_wgan_pilot
import orthogonal_wgan_dml
import wgan_gp_baseline
import crossfit_wgan_pilot
from calibrated_wgan_pilot import CalibrationConfig, calibration_weight, estimate, synthetic_for_fold, wilson
from baseline_direct_inversion import SimulationConfig, valuation_distribution
from crossfit_wgan_pilot import make_folds
from fixed_h_small_simulation import CachedMoments, METHODS
from orthogonal_wgan_dml import DMLConfig, true_regularized_target
from confidence_set_reporting import describe_confidence_set
from diagnose_fixed_h_coverage import score_statistics, invert_statistic, CRITICAL, BOUNDS

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026110800
SIZES = (500,2000)
REPS = 100
SHARD = 10
ALGORITHM_HASH = "fa64e7539a4ba41085fa6edca56b47ca23d6af5d6ad92ff19d6973ef91ef3692"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def csv_write(path, frame):
    temporary=path.with_suffix(path.suffix+".tmp")
    frame.to_csv(temporary,index=False)
    temporary.replace(path)


def plan(seed=SEED,reps=REPS):
    records=[]
    for n in SIZES:
        for rep in range(reps):
            base=seed+n*100+rep
            records.append(dict(auctions=n,repetition=rep+1,data_seed=base,
                fold_seed=base+31415926,gan_seeds=[base+1000000+(k+1)*10000 for k in range(3)],
                generation_seeds=[base+1000000+(k+1)*10000+102 for k in range(3)]))
    return records


def seed_values(records):
    return [v for r in records for v in [r["data_seed"],r["fold_seed"],*r["gan_seeds"],*r["generation_seeds"]]]


def cell(n,start,output):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    for module in (calibrated_wgan_pilot,orthogonal_wgan_dml,wgan_gp_baseline,crossfit_wgan_pilot):
        if Path(module.__file__).resolve().parent != Path(__file__).resolve().parent:
            raise RuntimeError("A worker imported an estimator outside its frozen snapshot")
    if sha(Path(__file__).with_name("diagnose_fixed_h_coverage.py"))!=ALGORITHM_HASH:
        raise RuntimeError("Frozen score-inversion algorithm changed")
    torch.set_num_threads(1)
    config=CalibrationConfig(sample_sizes=(n,),repetitions=SHARD,seed=SEED)
    dc=DMLConfig(bandwidth=config.bandwidth,bidders=config.bidders,
        theta_lower=config.theta_lower,theta_upper=config.theta_upper)
    target=true_regularized_target(dc)
    target_refined=true_regularized_target(DMLConfig(**{**asdict(dc),"target_integration_draws":200000}))
    protocol=dict(configuration=asdict(config),repetition_start=start+1,repetition_end=start+SHARD,
        target=target,refined_target=target_refined,critical_value=CRITICAL,
        score_grid_size=281,score_endpoint_tolerance=1e-11,
        completed=False,rows=0,algorithm_sha256=ALGORITHM_HASH)
    json_write(output/"protocol.json",protocol)
    cache=output/"synthetic_cache"; cache.mkdir()
    distribution=valuation_distribution(SimulationConfig(bidders=config.bidders))
    rows=[]; clock=time.perf_counter()
    for rep in range(start,start+SHARD):
        print(f"n={n} repetition={rep+1}/100",flush=True)
        seed=SEED+n*100+rep
        observed=distribution.rvs(size=(n,config.bidders),random_state=np.random.default_rng(seed)).max(axis=1)
        folds=make_folds(n,config.folds,seed+31415926)
        moments=[]; weights=[]; training_time=0.
        for k,holdout in enumerate(folds):
            mask=np.ones(n,dtype=bool); mask[holdout]=False
            training=observed[mask]
            t=time.perf_counter()
            synthetic=synthetic_for_fold(training,n,rep,k,config,cache)
            training_time+=time.perf_counter()-t
            moments.append(CachedMoments(observed[holdout],training,synthetic,config.bandwidth,config.bidders))
            weights.append(calibration_weight(len(training),config.pseudo_observations))
        full=CachedMoments(observed,observed,np.empty(0),config.bandwidth,config.bidders)
        tasks=[(METHODS[0],moments,weights,False),(METHODS[1],moments,[0.]*config.folds,False),
               (METHODS[2],[full],[0.],True)]
        for method,data,w,plugin in tasks:
            t=time.perf_counter()
            fit=estimate(data,w,plugin=plugin,theta_bounds=BOUNDS,grid_size=141,
                jacobian_floor=1e-6,cdf_floor=1e-8)
            valid=fit["status"]=="ok"
            error=fit["estimate"]-target if valid else np.nan
            row=dict(auctions=n,repetition=rep+1,method=method,status=fit["status"],
                failure_reason=fit["failure_reason"],estimate=fit["estimate"],standard_error=fit["standard_error"],
                target=target,error=error,jacobian=fit["jacobian"],score_residual=fit["score_residual"],
                roots=fit["roots"],wald_coverage=bool(valid and abs(error)<=CRITICAL*fit["standard_error"]),
                wald_length=2*CRITICAL*fit["standard_error"] if valid else np.nan,
                inversion_status="failed",inversion_failure_reason="point_estimation_failed" if not valid else "",
                score_coverage=False,score_length=np.nan,score_components="[]",component_count=0,
                boundary_truncated=False,score_t_at_target=np.nan,refinement_max_difference=np.nan)
            if valid:
                try:
                    statistic=lambda theta: score_statistics(theta,data,w)[2]
                    components=invert_statistic(statistic,fit["estimate"])
                    t0=float(statistic([target])[0])
                    cover=abs(t0)<=CRITICAL
                    membership=any(a-1e-9<=target<=b+1e-9 for a,b in components)
                    if cover!=membership: raise AssertionError("Test and set membership disagree")
                    refinement=np.nan
                    if rep in (0,49,99):
                        finer=invert_statistic(statistic,fit["estimate"],561)
                        if np.shape(finer)!=np.shape(components): raise AssertionError("Refinement topology mismatch")
                        refinement=float(np.max(abs(np.asarray(finer)-np.asarray(components))))
                        if refinement>1e-8: raise AssertionError("Refinement endpoint mismatch")
                    row.update(inversion_status="ok",inversion_failure_reason="",score_coverage=bool(cover),
                        score_length=sum(b-a for a,b in components),score_components=json.dumps(components),
                        component_count=len(components),score_t_at_target=t0,
                        boundary_truncated=any(abs(a-BOUNDS[0])<1e-9 or abs(b-BOUNDS[1])<1e-9 for a,b in components),
                        refinement_max_difference=refinement)
                except ValueError as exc:
                    row["inversion_failure_reason"]=str(exc)
            report=describe_confidence_set(json.loads(row["score_components"]),row["inversion_status"])
            row.update(report)
            row["conservative_hull_coverage"]=bool(row["inversion_status"]=="ok" and
                report["conservative_hull_lower"]<=target<=report["conservative_hull_upper"])
            row.update(wgan_weight=float(np.mean(weights)) if method==METHODS[0] else 0.,
                training_seconds=training_time if method==METHODS[0] else 0.,
                estimation_seconds=time.perf_counter()-t,data_seed=seed,fold_seed=seed+31415926)
            rows.append(row)
            print(f"  {method}: point={row['status']}, score={row['inversion_status']}",flush=True)
        csv_write(output/"raw_results.csv",pd.DataFrame(rows))
        print(f"  elapsed={time.perf_counter()-clock:.1f}s",flush=True)
    protocol.update(completed=True,rows=len(rows),elapsed_seconds=time.perf_counter()-clock)
    json_write(output/"protocol.json",protocol)


def aggregate(output,cells):
    summary_dir=output/"summary"; summary_dir.mkdir()
    raw=pd.concat([pd.read_csv(c/"raw_results.csv",keep_default_na=False,na_values=[""]) for c in cells],ignore_index=True)
    records=[]; paired=[]
    for (n,method),g in raw.groupby(["auctions","method"],sort=False):
        point_ok=g.status.eq("ok"); score_ok=point_ok & g.inversion_status.eq("ok")
        for interval,valid,coverage,length in [
            ("Wald",point_ok,g.wald_coverage,g.wald_length),
            ("Score inversion",score_ok,g.score_coverage,g.score_length),
            ("Conservative envelope",score_ok,g.conservative_hull_coverage,g.conservative_hull_length)]:
            success=int((coverage & valid).sum()); failures=int((~valid).sum())
            low,high=wilson(success,len(g)); flow,fhigh=wilson(failures,len(g))
            records.append(dict(auctions=n,method=method,interval=interval,repetitions=len(g),
                point_failures=int((~point_ok).sum()),interval_failures=failures,
                additional_interval_failures=int((point_ok & ~valid).sum()),failure_rate=failures/len(g),
                failure_low=flow,failure_high=fhigh,coverage=success/len(g),coverage_low=low,coverage_high=high,
                mean_length=length[valid].mean(),valid_intervals=int(valid.sum()),
                conditional_coverage=success/int(valid.sum()) if valid.any() else np.nan,
                disconnected_sets=int((g.component_count>1).sum()) if interval=="Score inversion" else 0,
                source_disconnected_sets=int((g.component_count>1).sum()) if interval!="Wald" else 0,
                boundary_truncated=int(g.boundary_truncated.sum()) if interval!="Wald" else 0,
                touches_lower_bound=int(g.set_touches_lower_bound.fillna(False).sum()) if interval!="Wald" else 0,
                touches_upper_bound=int(g.set_touches_upper_bound.fillna(False).sum()) if interval!="Wald" else 0))
        delta=g.score_coverage.astype(int)-g.wald_coverage.astype(int)
        pairvalid=point_ok & score_ok
        paired.append(dict(auctions=n,method=method,gained=int((delta==1).sum()),lost=int((delta==-1).sum()),
            coverage_change=delta.mean(),paired_mcse=delta.std(ddof=1)/np.sqrt(len(g)),
            valid_length_pairs=int(pairvalid.sum()),
            mean_length_change=(g.score_length[pairvalid]-g.wald_length[pairvalid]).mean()))
    csv_write(summary_dir/"raw_results.csv",raw)
    csv_write(summary_dir/"summary.csv",pd.DataFrame(records))
    csv_write(summary_dir/"paired_comparisons.csv",pd.DataFrame(paired))


def launch(output,workers):
    if output.exists(): raise FileExistsError(output)
    new_values=seed_values(plan()); old_values=set(seed_values(plan(20261108,10))+seed_values(plan(202611080,100)))
    if len(new_values)!=len(set(new_values)) or old_values.intersection(new_values):
        raise RuntimeError("Seed collision")
    output.mkdir(parents=True); (output/"logs").mkdir()
    source=output/"source_snapshot/code"; source.mkdir(parents=True)
    prior=ROOT/"output/experiments/fixed_h_100_2026-11-08/source_snapshot/code"
    for name in ("calibrated_wgan_pilot.py","orthogonal_wgan_dml.py","wgan_gp_baseline.py",
                 "crossfit_wgan_pilot.py","baseline_direct_inversion.py","fixed_h_small_simulation.py"):
        shutil.copy2(prior/name,source/name)
    frozen=ROOT/"output/experiments/fixed_h_coverage_diagnostic_2026-11-08/source_snapshot/code/diagnose_fixed_h_coverage.py"
    if sha(frozen)!=ALGORITHM_HASH: raise RuntimeError("Archived inversion changed")
    shutil.copy2(frozen,source/frozen.name)
    for name in ("fresh_score_validation.py","verify_fresh_score_validation.py","confidence_set_reporting.py"):
        shutil.copy2(ROOT/"code"/name,source/name)
    previous={}
    for folder in (ROOT/"output/experiments/fixed_h_small_2026-11-08",
                   ROOT/"output/experiments/fixed_h_100_2026-11-08",
                   ROOT/"output/experiments/fixed_h_coverage_diagnostic_2026-11-08",ROOT/"paper"):
        for f in folder.rglob("*"):
            if f.is_file(): previous[str(f.relative_to(ROOT))]=sha(f)
    json_write(output/"previous_results_sha256.json",previous)
    json_write(output/"seed_plan.json",plan())
    protocol=dict(base_seed=SEED,sample_sizes=SIZES,repetitions_per_size=REPS,methods=METHODS,
        estimators=asdict(CalibrationConfig(seed=SEED,sample_sizes=SIZES,repetitions=REPS)),
        interval=dict(critical_value=1.96,bounds=BOUNDS,grid_size=281,endpoint_tolerance=1e-11,
            refined_grid_size=561,refined_repetitions=[1,50,100],algorithm_sha256=ALGORITHM_HASH),
        reporting=dict(rule="full_set_with_conservative_hull_v1",boundary_tolerance=1e-9,
            preserve_all_components=True,envelope_is_superset=True),
        purpose="independent fresh-seed confirmation; no tuning from validation outcomes",
        failure_policy="all repetitions retained; interval failures count as noncoverage; lengths conditional on valid intervals",
        target="fixed-bandwidth theta_0,h; not exact r0",manuscripts_changed=False)
    json_write(output/"protocol.json",protocol)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),base_seed=SEED,sample_sizes=SIZES,
        repetitions_per_size=REPS,workers=workers,shard_size=SHARD,completed=False,
        prior_results_preserved=False,seed_independence_verified=True,runner_pid=os.getpid(),
        source_sha256={f.name:sha(f) for f in source.glob("*.py")})
    json_write(output/"run_manifest.json",manifest)
    (output/"runner.pid").write_text(str(os.getpid())+"\n")
    started=time.perf_counter()
    def run(job):
        n,start=job
        cellpath=output/f"n{n}"/f"part_{start+1:03d}_{start+SHARD:03d}"
        logpath=output/"logs"/f"n{n}_{start+1:03d}_{start+SHARD:03d}.log"
        cmd=[sys.executable,str(source/"fresh_score_validation.py"),"--cell","--size",str(n),
             "--start-rep",str(start),"--output-dir",str(cellpath)]
        env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
                 VECLIB_MAXIMUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
        with logpath.open("x") as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        if result.returncode: raise RuntimeError(f"Shard failed: n={n} rep={start+1}; {logpath}")
        print(f"Finished n={n} repetitions {start+1}-{start+SHARD}",flush=True)
        return cellpath
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures=[executor.submit(run,(n,start)) for start in range(0,REPS,SHARD) for n in SIZES]
            cells=[f.result() for f in as_completed(futures)]
        aggregate(output,sorted(cells))
        manifest["prior_results_preserved"]=all(sha(ROOT/k)==v for k,v in previous.items())
        json_write(output/"run_manifest.json",manifest)
        subprocess.run([sys.executable,str(source/"verify_fresh_score_validation.py"),
            "--run-dir",str(output),"--project-root",str(ROOT)],check=True)
        manifest.update(completed=True,elapsed_seconds=time.perf_counter()-started)
        json_write(output/"run_manifest.json",manifest)
    except BaseException as exc:
        manifest.update(error=str(exc),elapsed_seconds=time.perf_counter()-started)
        json_write(output/"run_manifest.json",manifest)
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--cell",action="store_true")
    p.add_argument("--size",type=int,choices=SIZES)
    p.add_argument("--start-rep",type=int,default=0)
    a=p.parse_args()
    if a.cell:
        if a.size is None or a.start_rep not in range(0,REPS,SHARD): p.error("Invalid cell")
        cell(a.size,a.start_rep,a.output_dir.resolve())
    else:
        if a.workers<1: p.error("--workers must be positive")
        launch(a.output_dir.resolve(),a.workers)
