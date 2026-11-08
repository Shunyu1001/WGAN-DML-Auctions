"""Exploratory fixed-variance score inversion on saved validation samples.

No GAN training and no point re-estimation. The nonlinear original score mean
is divided by the pooled score SD at the already saved root.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"output/experiments/fixed_h_score_validation_2026-11-08"
SNAPSHOT = SOURCE/"source_snapshot/code"
sys.path.insert(0,str(SNAPSHOT))
from baseline_direct_inversion import SimulationConfig, valuation_distribution
from calibrated_wgan_pilot import CalibrationConfig, calibration_weight, wilson
from crossfit_wgan_pilot import make_folds
from fixed_h_small_simulation import CachedMoments, METHODS
from wgan_gp_baseline import WGANConfig
from diagnose_fixed_h_coverage import invert_statistic, BOUNDS, CRITICAL


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sample(n, rep):
    config=CalibrationConfig(sample_sizes=(n,),repetitions=10,seed=2026110800)
    seed=config.seed+n*100+rep
    dist=valuation_distribution(SimulationConfig(bidders=5))
    obs=dist.rvs(size=(n,5),random_state=np.random.default_rng(seed)).max(axis=1)
    folds=make_folds(n,3,seed+31415926)
    cache=SOURCE/f"n{n}"/f"part_{rep//10*10+1:03d}_{rep//10*10+10:03d}"/"synthetic_cache"
    wc=WGANConfig(training_steps=1200,bidders=5,generated_values=50000,device="cpu")
    moments=[]; weights=[]; cache_checks={}
    for k, holdout in enumerate(folds):
        mask=np.ones(n,dtype=bool); mask[holdout]=False
        training=obs[mask]
        metadata=json.dumps(dict(wgan=asdict(wc),seed=seed+1000000+(k+1)*10000,
            generated_maxima=50000,source_hash=digest(SNAPSHOT/"wgan_gp_baseline.py"),
            data_hash=hashlib.sha256(training.tobytes()).hexdigest()),sort_keys=True)
        path=cache/(hashlib.sha256(metadata.encode()).hexdigest()[:20]+".npz")
        with np.load(path,allow_pickle=False) as stored:
            if str(stored["metadata"])!=metadata: raise RuntimeError("Cache mismatch")
            synthetic=stored["synthetic"].copy()
        cache_checks[str(path.relative_to(SOURCE))]=digest(path)
        moments.append(CachedMoments(obs[holdout],training,synthetic,.15,5))
        weights.append(calibration_weight(len(training),50))
    full=CachedMoments(obs,obs,np.empty(0),.15,5)
    return [(METHODS[0],moments,weights),(METHODS[1],moments,[0.]*3),(METHODS[2],[full],[0.])],cache_checks


def mean_score(theta,moments,weights):
    values=np.array([f.score_mean(theta,w) for f,w in zip(moments,weights)])
    return np.average(values,axis=0,weights=[len(f.observed) for f in moments])


def run_size(n,out):
    out.mkdir()
    baseline=pd.read_csv(SOURCE/"summary/raw_results.csv",keep_default_na=False,na_values=[""])
    baseline=baseline[baseline.auctions.eq(n)].set_index(["repetition","method"])
    results=[]; checks={}; started=time.perf_counter()
    for rep in range(100):
        tasks,newchecks=load_sample(n,rep);checks.update(newchecks)
        for method,moments,weights in tasks:
            old=baseline.loc[(rep+1,method)]
            row=dict(auctions=n,repetition=rep+1,method=method,target=old.target,
                estimate=old.estimate,standard_error=old.standard_error,jacobian=old.jacobian,
                status=old.status,wald_coverage=bool(old.wald_coverage),wald_length=old.wald_length,
                original_score_status=old.inversion_status,original_score_coverage=bool(old.score_coverage),
                original_score_length=old.score_length,original_components=old.score_components,
                original_component_count=old.component_count,
                original_boundary_truncated=bool(old.boundary_truncated),
                fixed_status="failed",fixed_failure_reason="point_estimation_failed",
                fixed_coverage=False,fixed_length=np.nan,fixed_components="[]",
                fixed_component_count=0,fixed_boundary_truncated=False,fixed_t_at_target=np.nan,
                root_sd_reconstruction_error=np.nan)
            if old.status=="ok":
                fixed_sd=float(old.standard_error*np.sqrt(n)*abs(old.jacobian))
                if not np.isfinite(fixed_sd) or fixed_sd<=0:
                    raise RuntimeError("Invalid saved-root score SD")
                if rep in (0,28,49,99):
                    direct=np.concatenate([f.individual_scores(old.estimate,w) for f,w in zip(moments,weights)])
                    delta=abs(direct.std(ddof=1)-fixed_sd)
                    row["root_sd_reconstruction_error"]=delta
                    if delta>1e-10: raise RuntimeError("Saved-root SD reconstruction mismatch")
                def statistic(t):
                    answer=np.sqrt(n)*mean_score(t,moments,weights)/fixed_sd
                    if not np.all(np.isfinite(answer)): raise ValueError("nonfinite_statistic")
                    return answer
                try:
                    components=invert_statistic(statistic,float(old.estimate),281)
                    t0=float(statistic([float(old.target)])[0])
                    cover=abs(t0)<=CRITICAL
                    membership=any(a-1e-9<=old.target<=b+1e-9 for a,b in components)
                    if cover!=membership: raise ValueError("test_set_membership_mismatch")
                    row.update(fixed_status="ok",fixed_failure_reason="",fixed_coverage=bool(cover),
                        fixed_length=sum(b-a for a,b in components),fixed_components=json.dumps(components),
                        fixed_component_count=len(components),
                        fixed_boundary_truncated=any(abs(a-BOUNDS[0])<1e-9 or abs(b-BOUNDS[1])<1e-9 for a,b in components),
                        fixed_t_at_target=t0)
                except (ValueError,RuntimeError,FloatingPointError) as exc:
                    row["fixed_failure_reason"]=str(exc)
            results.append(row)
        pd.DataFrame(results).to_csv(out/"raw_results.csv",index=False)
        print(f"n={n} fixed-variance {rep+1}/100 elapsed={time.perf_counter()-started:.1f}s",flush=True)
    (out/"cache_sha256.json").write_text(json.dumps(checks,indent=2)+"\n")
    (out/"completed.json").write_text(json.dumps(dict(completed=True,rows=len(results),elapsed_seconds=time.perf_counter()-started),indent=2)+"\n")


def summarize(out):
    raw=pd.concat([pd.read_csv(out/f"n{n}/raw_results.csv",keep_default_na=False,na_values=[""])
                   for n in (500,2000)],ignore_index=True)
    summaries=[];paired=[]
    for (n,method),g in raw.groupby(["auctions","method"],sort=False):
        point=g.status.eq("ok");old=point & g.original_score_status.eq("ok");new=point & g.fixed_status.eq("ok")
        for label,valid,cov,length,parts,boundary in [
            ("Wald",point,g.wald_coverage,g.wald_length,np.ones(len(g)),np.zeros(len(g))),
            ("Original score",old,g.original_score_coverage,g.original_score_length,g.original_component_count,g.original_boundary_truncated),
            ("Fixed variance score",new,g.fixed_coverage,g.fixed_length,g.fixed_component_count,g.fixed_boundary_truncated)]:
            success=int((valid & cov).sum());low,high=wilson(success,len(g))
            summaries.append(dict(auctions=n,method=method,interval=label,repetitions=len(g),
                failures=int((~valid).sum()),failure_rate=(~valid).mean(),coverage=success/len(g),
                coverage_low=low,coverage_high=high,mean_length=length[valid].mean(),
                disconnected_sets=int((np.asarray(parts)[valid]>1).sum()),
                boundary_truncated=int(np.asarray(boundary)[valid].sum())))
        for label,cov,length,valid in [("Wald",g.wald_coverage,g.wald_length,point),
                                     ("Original score",g.original_score_coverage,g.original_score_length,old)]:
            diff=g.fixed_coverage.astype(int)-cov.astype(int);both=valid & new
            paired.append(dict(auctions=n,method=method,comparison=f"Fixed variance minus {label}",
                gained=int((diff==1).sum()),lost=int((diff==-1).sum()),coverage_change=diff.mean(),
                paired_mcse=diff.std(ddof=1)/np.sqrt(len(g)),valid_length_pairs=int(both.sum()),
                mean_length_change=(g.fixed_length[both]-length[both]).mean()))
    raw.to_csv(out/"raw_results.csv",index=False)
    pd.DataFrame(summaries).to_csv(out/"summary.csv",index=False)
    pd.DataFrame(paired).to_csv(out/"paired_comparisons.csv",index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    verification=dict(rows=len(raw),unique_keys=not raw.duplicated(["auctions","repetition","method"]).any(),
        matched_100_per_cell=bool(raw.groupby(["auctions","method"]).size().eq(100).all()),
        max_root_sd_reconstruction_error=float(pd.to_numeric(raw.root_sd_reconstruction_error,errors="coerce").max()),
        original_raw_sha256_unchanged=digest(SOURCE/"summary/raw_results.csv")==json.loads((out/"protocol.json").read_text())["original_raw_sha256"])
    verification["passed"]=bool(verification["rows"]==600 and verification["unique_keys"] and
        verification["matched_100_per_cell"] and verification["max_root_sd_reconstruction_error"]<1e-10 and verification["original_raw_sha256_unchanged"])
    (out/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    protocol=dict(design="exploratory paired fixed-variance shape sensitivity, not independent validation",
        definition="sqrt(n)*original nonlinear pooled score mean(theta)/score_sd(saved root)",
        root_score_sd="saved_standard_error * sqrt(n) * abs(saved_jacobian)",
        critical_value=CRITICAL,bounds=BOUNDS,grid_size=281,endpoint_tolerance=1e-11,
        sample_sizes=[500,2000],repetitions=100,source=str(SOURCE),new_training=False,
        original_raw_sha256=digest(SOURCE/"summary/raw_results.csv"),
        frozen_inversion_sha256=digest(SNAPSHOT/"diagnose_fixed_h_coverage.py"))
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda n:run_size(n,out/f"n{n}"),(500,2000)))
    summarize(out)
