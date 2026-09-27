#!/usr/bin/env python3
"""Paired pilot of a small, fixed local calibration of the existing WGAN-DML.

No target values enter estimation. The WGAN and its loss are unchanged. Its
smoothed maximum distribution receives the weight of 50 pseudo-observations;
training-fold empirical moments receive the weight of the actual observations.
All variants share data, folds, and trained generators. Original artifacts are
never overwritten. Cached synthetic samples make interrupted runs resumable.
"""
from __future__ import annotations

import argparse
import hashlib
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
from scipy.optimize import brentq
from scipy.special import ndtr

from baseline_direct_inversion import (
    SimulationConfig, estimate_reserve_from_maxima, make_revenue_interpolator,
    true_reserve, valuation_distribution,
)
from crossfit_wgan_pilot import make_folds
from orthogonal_wgan_dml import (
    DMLConfig, gaussian_kernel_moments, reserve_moment_and_derivatives,
    true_regularized_target,
)
from wgan_gp_baseline import (
    WGANConfig, choose_device, sample_maxima, set_seed, train_wgan_gp,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["WGAN-DML", "Calibrated WGAN-DML", "Empirical-local DML", "Smooth plug-in"]
COLORS = ["#B34C47", "#176D9C", "#3F846C", "#7A6B8D"]


@dataclass(frozen=True)
class CalibrationConfig:
    sample_sizes: tuple[int, ...] = (500, 2000, 10000)
    repetitions: int = 10
    bidders: int = 5
    folds: int = 3
    bandwidth: float = 0.15
    pseudo_observations: float = 50.0
    training_steps: int = 1200
    generated_maxima: int = 50000
    seed: int = 20260928
    threads: int = 1


def calibration_weight(training_size: int, pseudo_observations: float) -> float:
    return pseudo_observations / (training_size + pseudo_observations)


def scores(theta, observed, cdf, density, h, bidders):
    moment, da, dg = reserve_moment_and_derivatives(theta, cdf, density, bidders)
    z = (theta - observed) / h
    return moment + da * (ndtr(z) - cdf) + dg * (
        np.exp(-0.5 * z*z) / (np.sqrt(2*np.pi)*h) - density
    )


class FoldMoments:
    def __init__(self, observed, training, synthetic, bandwidth, bidders):
        self.observed, self.training, self.synthetic = observed, training, synthetic
        self.h, self.bidders = bandwidth, bidders

    def nuisance(self, theta, weight):
        a, g = gaussian_kernel_moments(theta, self.training, self.h)
        if weight:
            aw, gw = gaussian_kernel_moments(theta, self.synthetic, self.h)
            a, g = (1-weight)*a + weight*aw, (1-weight)*g + weight*gw
        return a, g

    def score_mean(self, theta, weight):
        a, g = self.nuisance(theta, weight)
        ao, go = gaussian_kernel_moments(theta, self.observed, self.h)
        m, da, dg = reserve_moment_and_derivatives(theta, a, g, self.bidders)
        return m + da*(ao-a) + dg*(go-g)

    def individual_scores(self, theta, weight):
        a, g = self.nuisance(theta, weight)
        return scores(theta, self.observed, a[0], g[0], self.h, self.bidders)


def estimate(fold_moments, weights, reference, plugin=False):
    sizes = np.array([len(f.observed) for f in fold_moments])
    n = sizes.sum()

    def mean_score(theta):
        if plugin:
            f = fold_moments[0]
            a, g = f.nuisance(theta, 0.0)
            return reserve_moment_and_derivatives(theta, a, g, f.bidders)[0]
        return np.average(np.array([f.score_mean(theta, w) for f, w in
                                   zip(fold_moments, weights)]), axis=0, weights=sizes)

    grid = np.linspace(0.45, 1.15, 141)
    grid_score = mean_score(grid)
    brackets = np.flatnonzero(grid_score[:-1]*grid_score[1:] <= 0)
    roots = [brentq(lambda r: float(mean_score(r)[0]), grid[j], grid[j+1],
                    xtol=1e-10) for j in brackets]
    r = min(roots, key=lambda x: abs(x-reference)) if roots else float(grid[np.argmin(abs(grid_score))])
    step = 1e-4
    jacobian = float((mean_score(r+step)[0]-mean_score(r-step)[0])/(2*step))
    psi = np.concatenate([f.individual_scores(r, w) for f, w in zip(fold_moments, weights)])
    influence = -psi/jacobian
    se = float(np.std(influence, ddof=1)/np.sqrt(n))
    return dict(estimate=r, standard_error=se, roots=len(roots),
                score_residual=float(mean_score(r)[0]), jacobian=jacobian,
                influence=influence)


def synthetic_for_fold(training, n, repetition, fold, config, cache):
    wc = WGANConfig(training_steps=config.training_steps, bidders=config.bidders,
                    generated_values=config.generated_maxima, device="cpu")
    seed = config.seed + n*100 + repetition + 1000000 + (fold+1)*10000
    metadata = json.dumps(dict(wgan=asdict(wc), seed=seed,
        generated_maxima=config.generated_maxima,
        source_hash=hashlib.sha256((ROOT/"code/wgan_gp_baseline.py").read_bytes()).hexdigest(),
        data_hash=hashlib.sha256(training.tobytes()).hexdigest()), sort_keys=True)
    key = hashlib.sha256(metadata.encode()).hexdigest()[:20]
    path = cache/f"{key}.npz"
    if path.exists():
        with np.load(path) as stored:
            if str(stored["metadata"]) != metadata:
                raise RuntimeError("Cache metadata mismatch")
            return stored["synthetic"].copy()
    started = time.perf_counter()
    generator, diagnostics = train_wgan_gp(training, wc, seed)
    set_seed(seed+102)
    synthetic = sample_maxima(generator, config.generated_maxima, config.bidders, choose_device("cpu"))
    np.savez_compressed(path, synthetic=synthetic, metadata=metadata)
    print(f"  fold {fold+1}: {time.perf_counter()-started:.1f}s; "
          f"training W1={diagnostics['best_max_bid_w1']:.4f}", flush=True)
    return synthetic


def run(config, output):
    torch.set_num_threads(config.threads)
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT/"output/tables/calibration_cache"
    cache.mkdir(parents=True, exist_ok=True)
    baseline = SimulationConfig(bidders=config.bidders)
    distribution = valuation_distribution(baseline)
    r0 = true_reserve(baseline)
    revenue = make_revenue_interpolator(baseline)
    optimal_revenue = float(revenue(r0))
    rh = true_regularized_target(DMLConfig(bandwidth=config.bandwidth, bidders=config.bidders))
    records = []
    for n in config.sample_sizes:
        for rep in range(config.repetitions):
            print(f"n={n}, repetition={rep+1}/{config.repetitions}", flush=True)
            rng = np.random.default_rng(config.seed+n*100+rep)
            observed = distribution.rvs(size=(n,config.bidders), random_state=rng).max(axis=1)
            folds = make_folds(n, config.folds, config.seed+n*100+rep+31415926)
            reference = estimate_reserve_from_maxima(observed, config.bidders)
            fold_data = []
            for k, holdout in enumerate(folds):
                mask = np.ones(n,dtype=bool); mask[holdout] = False
                training = observed[mask]
                synthetic = synthetic_for_fold(training,n,rep,k,config,cache)
                fold_data.append((observed[holdout],training,synthetic))
            alpha = [calibration_weight(len(t),config.pseudo_observations) for _,t,_ in fold_data]
            h_small = config.bandwidth*(n/500)**(-0.2)
            h_large = np.sqrt(2)*h_small
            fits = {}
            for h in sorted(set([config.bandwidth,h_small,h_large])):
                moments = [FoldMoments(o,t,s,h,config.bidders) for o,t,s in fold_data]
                for method, weights in zip(METHODS[:3], [[1.0]*config.folds,alpha,[0.0]*config.folds]):
                    fits[(method,h)] = estimate(moments,weights,reference)
                # Full-sample smoothed plug-in, with its own delta-method influence.
                full = FoldMoments(observed,observed,np.empty(0),h,config.bidders)
                fits[(METHODS[3],h)] = estimate([full],[0.0],reference,plugin=True)
            for method in METHODS:
                fit = fits[(method,config.bandwidth)]
                records.append(record(n,rep,method,"Fixed bandwidth",fit,rh,r0,
                    revenue,optimal_revenue,config.bandwidth,np.mean(alpha) if method==METHODS[1] else float(method==METHODS[0])))
                small,large = fits[(method,h_small)],fits[(method,h_large)]
                influence = 2*small["influence"]-large["influence"]
                corrected = dict(estimate=2*small["estimate"]-large["estimate"],
                    standard_error=float(np.std(influence,ddof=1)/np.sqrt(n)),
                    roots=min(small["roots"],large["roots"]),
                    multiple_roots=(small["roots"]>1 or large["roots"]>1),
                    score_residual=np.nan,jacobian=np.nan)
                records.append(record(n,rep,method,"Richardson",corrected,r0,r0,
                    revenue,optimal_revenue,h_small,np.mean(alpha) if method==METHODS[1] else float(method==METHODS[0])))
            pd.DataFrame(records).to_csv(output/"calibrated_wgan_raw.csv",index=False)
    raw = pd.DataFrame(records)
    summarize(raw,output)
    write_config(config, output, r0, rh)


def write_config(config, output, r0, rh):
    environment = dict(python=__import__("sys").version, numpy=np.__version__,
                       torch=torch.__version__,scipy=__import__("scipy").__version__,
                       pandas=pd.__version__,matplotlib=matplotlib.__version__,
                       platform=__import__("platform").platform(),device="cpu")
    (output/"calibrated_wgan_config.json").write_text(json.dumps(
        dict(**asdict(config),environment=environment,exact_reserve=r0,regularized_target=rh,
             design="paired data/folds/generators; fixed pseudo-count; all seeds retained"),indent=2)+"\n")


def record(n,rep,method,procedure,fit,target,r0,revenue,optimal_revenue,h,alpha):
    r,se = fit["estimate"],fit["standard_error"]
    return dict(auctions=n,repetition=rep+1,method=method,procedure=procedure,
        bandwidth=h,wgan_weight=alpha,estimate=r,standard_error=se,
        target=target,exact_reserve=r0,error_target=r-target,error_exact=r-r0,
        covers_target=bool(abs(r-target)<=1.96*se),covers_exact=bool(abs(r-r0)<=1.96*se),
        interval_length=3.92*se,revenue_regret=max(optimal_revenue-float(revenue(r)),0.0),
        missing_root=fit["roots"]==0,multiple_roots=fit.get("multiple_roots",fit["roots"]>1),
        score_residual=fit["score_residual"],jacobian=fit["jacobian"])


def wilson(successes,count):
    p=successes/count; z=1.96; den=1+z*z/count
    center=(p+z*z/(2*count))/den
    width=z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return max(0.0,center-width),min(1.0,center+width)


def summarize(raw,output):
    rows=[]
    for (n,procedure,method),group in raw.groupby(["auctions","procedure","method"],sort=False):
        e=group.error_target.to_numpy(); coverage=group.covers_target.mean()
        lo,hi=wilson(group.covers_target.sum(),len(group))
        rows.append(dict(auctions=n,procedure=procedure,method=method,repetitions=len(group),
            bias=e.mean(),rmse=np.sqrt(np.mean(e*e)),rmse_exact=np.sqrt(np.mean(group.error_exact**2)),
            coverage=coverage,coverage_low=lo,coverage_high=hi,
            coverage_mcse=np.sqrt(coverage*(1-coverage)/len(group)),
            mean_interval_length=group.interval_length.mean(),mean_revenue_regret=group.revenue_regret.mean(),
            missing_root_rate=group.missing_root.mean(),multiple_root_rate=group.multiple_roots.mean(),
            nonpositive_estimate_rate=(group.estimate<=0).mean(),
            mean_wgan_weight=group.wgan_weight.mean()))
    summary=pd.DataFrame(rows)
    summary.to_csv(output/"calibrated_wgan.csv",index=False)
    paired=[]
    for (n,procedure),group in raw.groupby(["auctions","procedure"]):
        pivot=group.pivot(index="repetition",columns="method",values="error_target")**2
        for reference in [METHODS[0],METHODS[2],METHODS[3]]:
            delta=pivot[METHODS[1]]-pivot[reference]
            paired.append(dict(auctions=n,procedure=procedure,comparison=reference,
                mse_difference=delta.mean(),paired_mcse=delta.std(ddof=1)/np.sqrt(len(delta))))
    pd.DataFrame(paired).to_csv(output/"calibrated_wgan_paired.csv",index=False)
    draw(summary,output)
    lines=[r"\begin{tabular}{rlrrrr}",r"\toprule",
           r"$n$ & Method & Bias & RMSE & Coverage & CI length \\",r"\midrule"]
    short=dict(zip(METHODS,["WGAN-DML","Calibrated WGAN","Empirical-local","Smooth plug-in"]))
    for n in sorted(raw.auctions.unique()):
        for method in METHODS:
            row=summary[(summary.auctions==n)&(summary.procedure=="Fixed bandwidth")&(summary.method==method)].iloc[0]
            lines.append(f"{n:,} & {short[method]} & {row.bias:.4f} & {row.rmse:.4f} & "
                         f"{100*row.coverage:.1f}\\% & {row.mean_interval_length:.4f} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (output/"calibrated_wgan.tex").write_text("\n".join(lines)+"\n")
    print(summary[["auctions","procedure","method","rmse","coverage"]].to_string(index=False),flush=True)


def draw(summary,output):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
    fig,axes=plt.subplots(2,2,figsize=(10.2,7.4),layout="constrained")
    ns=sorted(summary.auctions.unique()); x=np.arange(len(ns))
    markers=["o","D","^","s"]
    styles=["-","-","--",":"]
    for idx,(method,color) in enumerate(zip(METHODS,COLORS)):
        fixed=summary[(summary.method==method)&(summary.procedure=="Fixed bandwidth")].sort_values("auctions")
        exact=summary[(summary.method==method)&(summary.procedure=="Richardson")].sort_values("auctions")
        style=dict(color=color,marker=markers[idx],linestyle=styles[idx],lw=1.6,
                   ms=5,markerfacecolor="white",zorder=5 if idx==1 else 3)
        axes[0,0].plot(x,fixed.rmse,label=method,**style)
        offset=(idx-1.5)*0.09
        axes[0,1].errorbar(x+offset,100*fixed.coverage,
            yerr=np.vstack([100*(fixed.coverage-fixed.coverage_low),100*(fixed.coverage_high-fixed.coverage)]),
            fmt=markers[idx],color=color,capsize=3,ms=4,elinewidth=1)
        axes[1,0].plot(x,exact.rmse_exact,**style)
        axes[1,1].plot(x,exact.mean_revenue_regret*1e4,**style)
    titles=["A  Fixed-bandwidth target: RMSE","B  Fixed-bandwidth target: coverage",
            "C  Exact reserve: corrected RMSE","D  Exact reserve: corrected revenue regret"]
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,loc="left",fontsize=10.5,pad=10)
        ax.set_xticks(x,[f"{n:,}" for n in ns]); ax.set_xlabel("Number of auctions")
        ax.grid(axis="y",alpha=.18); ax.set_axisbelow(True)
    for ax in [axes[0,0],axes[1,0]]: ax.set_ylim(bottom=0)
    axes[1,1].set_yscale("log")
    axes[0,1].set_ylim(-3,103); axes[0,1].axhline(95,color="#444444",ls="--",lw=1)
    axes[0,1].set_ylabel("Coverage (%) with Wilson intervals")
    axes[1,1].set_ylabel(r"Revenue regret $\times 10^4$ (log scale)")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    reps=int(summary.repetitions.iloc[0])
    fig.suptitle(f"Local calibration of WGAN-DML | {reps} paired repetitions per sample size",fontsize=13)
    fig.savefig(output/"calibrated_wgan.png",dpi=220,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps",type=int,default=10)
    parser.add_argument("--sizes",type=int,nargs="+",default=[500,2000,10000])
    parser.add_argument("--steps",type=int,default=1200)
    parser.add_argument("--seed",type=int,default=20260928)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"output/tables/calibration_pilot")
    parser.add_argument("--combine-dirs",type=Path,nargs="+")
    args=parser.parse_args()
    if args.combine_dirs:
        configs=[json.loads((p/"calibrated_wgan_config.json").read_text()) for p in args.combine_dirs]
        fields=set(CalibrationConfig.__dataclass_fields__)-{"sample_sizes"}
        if any(any(c[key]!=configs[0][key] for key in fields) for c in configs):
            raise ValueError("Cannot pool experiments with different configurations")
        raw=pd.concat([pd.read_csv(p/"calibrated_wgan_raw.csv") for p in args.combine_dirs],ignore_index=True)
        if raw.duplicated(["auctions","repetition","method","procedure"]).any():
            raise ValueError("Duplicate simulation cells")
        args.output_dir.mkdir(parents=True,exist_ok=True)
        raw.to_csv(args.output_dir/"calibrated_wgan_raw.csv",index=False)
        summarize(raw,args.output_dir)
        config=CalibrationConfig(sample_sizes=tuple(sorted(raw.auctions.unique().tolist())),
            **{key:configs[0][key] for key in fields})
        write_config(config,args.output_dir,configs[0]["exact_reserve"],configs[0]["regularized_target"])
    else:
        run(CalibrationConfig(sample_sizes=tuple(args.sizes),repetitions=args.reps,
            training_steps=args.steps,seed=args.seed),args.output_dir)
