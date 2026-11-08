"""Numerical and influence diagnostics for the two disconnected-set samples."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"output/experiments/fixed_h_score_validation_2026-11-08"
sys.path.insert(0,str(SOURCE/"source_snapshot/code"))
from baseline_direct_inversion import SimulationConfig, valuation_distribution
from calibrated_wgan_pilot import CalibrationConfig, calibration_weight, estimate, scores
from crossfit_wgan_pilot import make_folds
from fixed_h_small_simulation import CachedMoments, METHODS
from diagnose_fixed_h_coverage import score_statistics, invert_statistic
from wgan_gp_baseline import WGANConfig


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(rep):
    n=500; seed=2026110800+n*100+rep-1
    config=CalibrationConfig(sample_sizes=(n,),repetitions=10,seed=2026110800)
    observed=valuation_distribution(SimulationConfig(bidders=5)).rvs(
        size=(n,5),random_state=np.random.default_rng(seed)).max(axis=1)
    folds=make_folds(n,3,seed+31415926)
    start=((rep-1)//10)*10
    cache=SOURCE/f"n{n}"/f"part_{start+1:03d}_{start+10:03d}"/"synthetic_cache"
    wc=WGANConfig(training_steps=1200,bidders=5,generated_values=50000,device="cpu")
    moments=[]; weights=[]; ids=[]; cache_hashes={}
    for k,holdout in enumerate(folds):
        mask=np.ones(n,dtype=bool); mask[holdout]=False; training=observed[mask]
        metadata=json.dumps(dict(wgan=asdict(wc),seed=seed+1000000+(k+1)*10000,
            generated_maxima=50000,source_hash=sha(SOURCE/"source_snapshot/code/wgan_gp_baseline.py"),
            data_hash=hashlib.sha256(training.tobytes()).hexdigest()),sort_keys=True)
        path=cache/(hashlib.sha256(metadata.encode()).hexdigest()[:20]+".npz")
        with np.load(path,allow_pickle=False) as stored:
            if str(stored["metadata"])!=metadata: raise ValueError("cache mismatch")
            synthetic=stored["synthetic"].copy()
        cache_hashes[str(path.relative_to(SOURCE))]=sha(path)
        moments.append(CachedMoments(observed[holdout],training,synthetic,.15,5))
        weights.append(calibration_weight(len(training),50))
        ids.extend((int(i),k+1) for i in holdout)
    full=CachedMoments(observed,observed,np.empty(0),.15,5)
    return observed,[(METHODS[0],moments,weights,ids,False),
        (METHODS[1],moments,[0.]*3,ids,False),
        (METHODS[2],[full],[0.],[(i,0) for i in range(n)],True)],cache_hashes


def oracle_moments(theta):
    def maximum_log_density(z):
        return 5*np.exp(-.5*z*z)/np.sqrt(2*np.pi)*ndtr(z)**4
    def cdf_integrand(z):
        return ndtr((theta-np.exp(.5*z))/.15)*maximum_log_density(z)
    def density_integrand(z):
        v=(theta-np.exp(.5*z))/.15
        return np.exp(-.5*v*v)/(np.sqrt(2*np.pi)*.15)*maximum_log_density(z)
    a,ae=quad(cdf_integrand,-12,12,epsabs=2e-12,epsrel=2e-10,limit=200)
    g,ge=quad(density_integrand,-12,12,epsabs=2e-12,epsrel=2e-10,limit=200)
    return a,g,ae,ge


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/"grid_stability.csv").exists(): raise FileExistsError("Already analyzed")
    old=pd.read_csv(SOURCE/"summary/raw_results.csv")
    cases=[int(x) for x in sorted(old.loc[old.component_count>1,"repetition"].unique())]
    if cases!=[29,50]: raise ValueError(f"Unexpected anomalous cases {cases}")
    probe_rows=[]; grids=[]; curve_frames=[]; metadata=[]; cache_hashes={}
    for rep in cases:
        observed,tasks,hashes=load_case(int(rep)); cache_hashes.update(hashes)
        for method,moments,weights,ids,plugin in tasks:
            row=old[(old.auctions==500)&(old.repetition==rep)&(old.method==method)].iloc[0]
            fit=estimate(moments,weights,plugin=plugin,theta_bounds=(.45,1.15),grid_size=141,
                jacobian_floor=1e-6,cdf_floor=1e-8)
            if fit["status"]!="ok" or abs(fit["estimate"]-row.estimate)>1e-12:
                raise AssertionError("Point estimate does not replay")
            original=np.asarray(json.loads(row.score_components))
            fn=lambda t: score_statistics(t,moments,weights)[2]
            for count in (281,561,1121,2241):
                components=np.asarray(invert_statistic(fn,float(row.estimate),count))
                if components.shape!=original.shape: raise AssertionError("Topology changed")
                difference=float(abs(components-original).max())
                residuals=[abs(float(fn([x])[0])**2-1.96**2)
                           for x in components.ravel() if .45+1e-9<x<1.15-1e-9]
                grids.append(dict(repetition=rep,method=method,grid_size=count,
                    component_count=len(components),maximum_endpoint_difference=difference,
                    maximum_boundary_residual=max(residuals,default=0.),components=json.dumps(components.tolist())))
            theta=np.linspace(.45,1.15,2241)
            mu,var,tstat=score_statistics(theta,moments,weights)
            curve_frames.append(pd.DataFrame(dict(repetition=rep,method=method,theta=theta,
                mean=mu,sd=np.sqrt(var),t=tstat)))
            probes=np.unique([.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,row.target,row.estimate,.90])
            for point in probes:
                try:
                    mu,var,tstat=score_statistics([point],moments,weights)
                except ValueError as exc:
                    probe_rows.append(dict(repetition=rep,method=method,theta=point,status=str(exc)))
                    continue
                psi=np.concatenate([f.individual_scores(point,w) for f,w in zip(moments,weights)])
                centered=psi-psi.mean(); squares=centered**2
                shares=squares/squares.sum(); idx=int(np.argmax(squares))
                direct=np.var(psi,ddof=1)
                extended=float(np.var(psi.astype(np.longdouble),ddof=1))
                a=[float(f.nuisance(point,w)[0][0]) for f,w in zip(moments,weights)]
                g=[float(f.nuisance(point,w)[1][0]) for f,w in zip(moments,weights)]
                oa,og,ae,ge=oracle_moments(float(point))
                oracle=scores(point,observed,oa,og,.15,5)
                probe_rows.append(dict(repetition=rep,method=method,theta=point,status="ok",
                    mean=mu[0],sd=np.sqrt(var[0]),t=tstat[0],cdf_min=min(a),cdf_max=max(a),
                    fold_cdf=json.dumps(a),fold_density=json.dumps(g),
                    direct_variance=direct,formula_variance=var[0],extended_variance=extended,
                    formula_variance_abs_error=abs(direct-var[0]),
                    extended_variance_abs_error=abs(direct-extended),
                    top1_variance_share=max(shares),top5_variance_share=sum(sorted(shares)[-5:]),
                    effective_contributors=1/sum(shares**2),dominant_observation_index=ids[idx][0],
                    dominant_holdout_fold=ids[idx][1],dominant_maximum=observed[ids[idx][0]],
                    dominant_score=psi[idx],oracle_cdf=oa,oracle_density=og,
                    oracle_integration_error=max(ae,ge),oracle_mean=oracle.mean(),
                    oracle_sd=np.std(oracle,ddof=1),oracle_t=np.sqrt(500)*oracle.mean()/np.std(oracle,ddof=1)))
            metadata.append(dict(repetition=int(rep),method=method,estimate=float(row.estimate),
                target=float(row.target),minimum_maximum=float(observed.min()),
                second_smallest_maximum=float(np.sort(observed)[1]),
                saved_components=json.loads(row.score_components)))
        print(f"Diagnosed repetition {rep}",flush=True)
    pd.DataFrame(grids).to_csv(output/"grid_stability.csv",index=False)
    pd.DataFrame(probe_rows).to_csv(output/"probe_statistics.csv",index=False)
    pd.concat(curve_frames,ignore_index=True).to_csv(output/"score_curves.csv",index=False)
    (output/"cases.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (output/"cache_sha256.json").write_text(json.dumps(cache_hashes,indent=2)+"\n")
    hull_rows=[]
    for _,row in old.iterrows():
        components=json.loads(row.score_components)
        if components:
            low=min(c[0] for c in components); high=max(c[1] for c in components)
            hull_cover=low<=row.target<=high
            if row.score_coverage and not hull_cover: raise AssertionError("Hull lost coverage")
            hull_rows.append(dict(auctions=row.auctions,repetition=row.repetition,method=row.method,
                original_coverage=row.score_coverage,hull_coverage=hull_cover,
                original_length=row.score_length,hull_length=high-low,hull_lower=low,hull_upper=high,
                component_count=len(components),boundary_truncated=row.boundary_truncated))
    hull=pd.DataFrame(hull_rows); hull.to_csv(output/"convex_hull_raw.csv",index=False)
    summary=hull.groupby(["auctions","method"],sort=False).agg(repetitions=("repetition","size"),
        original_coverage=("original_coverage","mean"),hull_coverage=("hull_coverage","mean"),
        original_length=("original_length","mean"),hull_length=("hull_length","mean"),
        boundary_cases=("boundary_truncated","sum")).reset_index()
    summary.to_csv(output/"convex_hull_summary.csv",index=False)
    (output/"completed.json").write_text(json.dumps(dict(completed=True,cases=cases,
        grid_checks=len(grids),probe_rows=len(probe_rows),original_inputs_read_only=True),indent=2)+"\n")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",type=Path,required=True)
    run(p.parse_args().output_dir)
