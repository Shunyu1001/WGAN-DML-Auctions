#!/usr/bin/env python3
"""Nested-validation ablation of monotone WGAN generation and adaptive mixing.

All selection is inside the outer training fold. Evaluation truth enters only
reporting. Original experiments and their artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import copy
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
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq, minimize
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import lognorm, norm, wasserstein_distance, weibull_min
from torch import nn
from torch.nn import functional as F

from baseline_direct_inversion import estimate_reserve_from_maxima
from calibrated_wgan_pilot import FoldMoments, calibration_weight, estimate, record, wilson
from crossfit_wgan_pilot import make_folds
from orthogonal_wgan_dml import gaussian_kernel_moments, reserve_moment_and_derivatives
from wgan_gp_baseline import WGANConfig, MaxBidCritic, gradient_penalty, sample_maxima, set_seed, train_wgan_gp

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["Original fixed", "Monotone fixed", "Original adaptive", "Monotone adaptive",
           "Empirical DML", "Smooth plug-in", "Lognormal MLE"]
COLORS = ["#B34C47", "#CB9246", "#7975A5", "#1677A0", "#39836C", "#565656", "#A25891"]


@dataclass(frozen=True)
class AdaptiveConfig:
    sample_sizes: tuple[int, ...] = (500, 2000)
    distributions: tuple[str, ...] = ("lognormal", "weibull")
    repetitions: int = 10
    bidders: int = 5
    folds: int = 3
    bandwidth: float = .15
    training_steps: int = 1200
    generated_maxima: int = 50000
    quadrature_nodes: int = 32768
    monotone_learning_rate: float = .0005
    validation_fraction: float = .25
    weight_grid: tuple[float, ...] = (0., .25, .5, 1.)
    pseudo_observations: float = 50.
    seed: int = 20260930
    threads: int = 1


class MonotoneLogQuantile(nn.Module):
    """11-knot linear spline in normal-quantile coordinates; linear tails.

    Positive slopes imply a strictly increasing quantile. Exponentiation
    gives positive values with unbounded upper support, without a tanh cap.
    Initialization is a data-only linear fit, not knowledge of the DGP.
    """
    def __init__(self, location: float, scale: float):
        super().__init__()
        knots = torch.tensor([-3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3.])
        self.register_buffer("knots", knots)
        self.anchor = nn.Parameter(torch.tensor(float(location)))
        positive_scale = max(float(scale), .02)
        self.raw_slopes = nn.Parameter(torch.full((10,), float(np.log(np.expm1(positive_scale)))))

    def log_from_z(self, z):
        slopes = F.softplus(self.raw_slopes) + 1e-5
        heights = torch.cat([slopes.new_zeros(1), torch.cumsum(slopes*torch.diff(self.knots), 0)])
        values = self.anchor + heights - heights[5]
        index = (torch.bucketize(z.contiguous(), self.knots)-1).clamp(0, len(slopes)-1)
        return values[index] + slopes[index]*(z-self.knots[index])

    def forward(self, u):
        return self.log_from_z(torch.special.ndtri(u.clamp(1e-7, 1-1e-7)))


def initialize_monotone(observed, bidders):
    p = np.linspace(.05, .95, 39)
    z = ndtri(p**(1/bidders))
    slope, intercept = np.polyfit(z, np.quantile(np.log(observed), p), 1)
    return MonotoneLogQuantile(intercept, max(slope, .02))


def monotone_maxima(generator, count, bidders, deterministic=False):
    u = ((torch.arange(count, dtype=torch.float64)+.5)/count if deterministic
         else torch.rand(count).clamp(1e-7, 1-1e-7))
    # z is calculated in double precision to retain extreme quadrature nodes.
    return generator.log_from_z(torch.special.ndtri(u.pow(1/bidders)).float()).reshape(-1, 1)


def train_monotone(observed, config, seed):
    set_seed(seed)
    generator = initialize_monotone(observed, config.bidders)
    critic = MaxBidCritic(64)
    real_raw = torch.as_tensor(np.log(observed).reshape(-1, 1), dtype=torch.float32)
    location, scale = real_raw.mean(), real_raw.std().clamp_min(1e-6)
    real = (real_raw-location)/scale
    go = torch.optim.Adam(generator.parameters(), lr=config.monotone_learning_rate, betas=(0., .9))
    co = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0., .9))
    rng = np.random.default_rng(seed)
    batch_size = min(512, len(observed))

    def fit_distance():
        with torch.no_grad():
            maxima = monotone_maxima(generator, 4096, config.bidders, True).exp().numpy().ravel()
        return wasserstein_distance(observed, maxima)

    best_fit = initial_fit = fit_distance()
    best_state, best_step = copy.deepcopy(generator.state_dict()), 0
    for step in range(config.training_steps):
        for _ in range(5):
            real_batch = real[rng.integers(0, len(real), size=batch_size)]
            fake = (monotone_maxima(generator, batch_size, config.bidders)-location)/scale
            loss = critic(fake.detach()).mean()-critic(real_batch).mean()
            loss = loss + .1*gradient_penalty(critic, real_batch, fake.detach())
            co.zero_grad(set_to_none=True); loss.backward(); co.step()
        fake = (monotone_maxima(generator, batch_size, config.bidders)-location)/scale
        loss = -critic(fake).mean()
        go.zero_grad(set_to_none=True); loss.backward(); go.step()
        if step % max(config.training_steps//60, 10) == 0 or step == config.training_steps-1:
            distance = fit_distance()
            if distance < best_fit:
                best_fit, best_step = distance, step+1
                best_state = copy.deepcopy(generator.state_dict())
    generator.load_state_dict(best_state)
    with torch.no_grad():
        synthetic = monotone_maxima(generator, config.quadrature_nodes, config.bidders, True).exp().numpy().ravel()
    return synthetic, dict(best_step=best_step, initial_w1=initial_fit, final_w1=best_fit,
                          generator_parameters=sum(p.numel() for p in generator.parameters()))


def cached_fit(observed, architecture, config, seed, cache):
    source = hashlib.sha256(Path(__file__).read_bytes() + (ROOT/"code/wgan_gp_baseline.py").read_bytes()).hexdigest()
    settings = {k: getattr(config, k) for k in ["bidders", "training_steps", "generated_maxima",
                "quadrature_nodes", "monotone_learning_rate"]}
    metadata = json.dumps(dict(architecture=architecture, settings=settings, seed=seed,
        source=source, data=hashlib.sha256(observed.tobytes()).hexdigest()), sort_keys=True)
    path = cache/(hashlib.sha256(metadata.encode()).hexdigest()[:24]+".npz")
    if path.exists():
        with np.load(path) as stored:
            if str(stored["metadata"]) != metadata:
                raise ValueError("Cache metadata mismatch")
            return stored["synthetic"].copy(), json.loads(str(stored["diagnostics"]))
    started = time.perf_counter()
    if architecture == "monotone":
        synthetic, diagnostics = train_monotone(observed, config, seed)
    else:
        wc = WGANConfig(training_steps=config.training_steps, bidders=config.bidders, device="cpu")
        generator, detail = train_wgan_gp(observed, wc, seed)
        set_seed(seed+102)
        synthetic = sample_maxima(generator, config.generated_maxima, config.bidders, torch.device("cpu"))
        diagnostics = dict(best_step=detail["best_step"], final_w1=detail["best_max_bid_w1"],
            generator_parameters=sum(p.numel() for p in generator.parameters()),
            lower_support=float(torch.exp(generator.output_center-generator.output_half_width)),
            upper_support=float(torch.exp(generator.output_center+generator.output_half_width)))
    diagnostics["training_seconds"] = time.perf_counter()-started
    if not np.all(np.isfinite(synthetic)) or np.any(synthetic <= 0):
        raise ValueError("Non-finite or nonpositive generated values")
    np.savez_compressed(path, synthetic=synthetic, metadata=metadata, diagnostics=json.dumps(diagnostics))
    return synthetic, diagnostics


def inner_split(training, seed, fraction):
    indices = np.random.default_rng(seed).permutation(len(training))
    nval = max(1, int(len(training)*fraction))
    return training[indices[nval:]], training[indices[:nval]]


def validation_weight(fitting, validation, synthetic, h, bidders, candidates):
    reference = estimate_reserve_from_maxima(fitting, bidders)
    grid = np.linspace(.6*reference, 1.4*reference, 21)
    ae, ge = gaussian_kernel_moments(grid, fitting, h)
    aw, gw = gaussian_kernel_moments(grid, synthetic, h)
    zfit = (grid[:, None]-fitting[None, :])/h
    fit_signals = np.stack([ndtr(zfit), np.exp(-zfit*zfit/2)/(np.sqrt(2*np.pi)*h)])
    # Scale using fitting data only; the floor is fixed, not chosen on validation.
    scale = np.maximum(fit_signals.var(axis=2), .01)
    zval = (grid[:, None]-validation[None, :])/h
    signals = np.stack([ndtr(zval), np.exp(-zval*zval/2)/(np.sqrt(2*np.pi)*h)])
    empirical, generated = np.stack([ae, ge]), np.stack([aw, gw])
    risks = []
    for weight in candidates:
        predicted = (1-weight)*empirical + weight*generated
        risks.append(float(np.mean((signals-predicted[:, :, None])**2/scale[:, :, None])))
    best = int(np.argmin(risks))  # Ascending grid: exact ties favor empirical.
    return float(candidates[best]), risks


def distribution_for(name):
    if name == "lognormal":
        return lognorm(s=.5, scale=1.)
    if name == "weibull":
        return weibull_min(c=1.5, scale=1.)
    raise ValueError(name)


def targets_and_revenue(distribution, bidders, h):
    r0 = brentq(lambda r: distribution.sf(r)-r*distribution.pdf(r), .02, 5.)
    def moment(r):
        a = quad(lambda b: ndtr((r-b)/h)*bidders*distribution.cdf(b)**(bidders-1)*distribution.pdf(b),
                 0, np.inf, epsabs=1e-10)[0]
        g = quad(lambda b: norm.pdf((r-b)/h)/h*bidders*distribution.cdf(b)**(bidders-1)*distribution.pdf(b),
                 0, np.inf, epsabs=1e-10)[0]
        return float(reserve_moment_and_derivatives(r, a, g, bidders)[0])
    rh = brentq(moment, .2, 1.8)
    grid = np.linspace(0, distribution.ppf(1-1e-10), 200000)
    cdf = distribution.cdf(grid)
    survival = 1-cdf**bidders-bidders*(1-cdf)*cdf**(bidders-1)
    integral = -cumulative_trapezoid(survival[::-1], grid[::-1], initial=0)[::-1]
    rev = grid*(1-cdf**bidders)+integral
    def revenue(r):
        # A negative reserve is equivalent to zero for nonnegative values.
        return np.interp(np.maximum(r, 0), grid, rev)
    return r0, rh, revenue


def lognormal_reserve(parameters):
    mu, log_sigma = parameters
    sigma = np.exp(log_sigma)
    z = brentq(lambda z: log_ndtr(-z)-norm.logpdf(z)+log_sigma, -15, 20)
    return float(np.exp(mu+sigma*z))


def lognormal_mle(observed, bidders):
    logb = np.log(observed)
    def individual_gradient(p):
        sigma = np.exp(p[1]); z = (logb-p[0])/sigma
        dz = (bidders-1)*np.exp(norm.logpdf(z)-log_ndtr(z))-z
        return np.column_stack([-dz/sigma, -1-z*dz])
    def objective(p):
        z = (logb-p[0])/np.exp(p[1])
        return -np.sum(np.log(bidders)+(bidders-1)*log_ndtr(z)+norm.logpdf(z)-p[1]-logb)
    p = np.linspace(.05, .95, 39)
    sigma, mu = np.polyfit(ndtri(p**(1/bidders)), np.quantile(logb, p), 1)
    fitted = minimize(objective, [mu, np.log(max(sigma, .02))],
        jac=lambda p: -individual_gradient(p).sum(axis=0), method="BFGS", options={"gtol":1e-6})
    parameters = fitted.x; step = 1e-5
    eye = np.eye(2)*step
    hessian = np.column_stack([(-individual_gradient(parameters+e).sum(axis=0)+
                                individual_gradient(parameters-e).sum(axis=0))/(2*step) for e in eye])
    inverse = np.linalg.inv((hessian+hessian.T)/2)
    score = individual_gradient(parameters)
    covariance = inverse@(score.T@score)@inverse
    derivative = np.array([(lognormal_reserve(parameters+e)-lognormal_reserve(parameters-e))/(2*step) for e in eye])
    gradient_norm = np.linalg.norm(score.sum(axis=0), ord=np.inf)
    return dict(estimate=lognormal_reserve(parameters), standard_error=float(np.sqrt(derivative@covariance@derivative)),
                roots=1, score_residual=gradient_norm, jacobian=np.nan,
                optimizer_success=bool(fitted.success or gradient_norm < 1e-3))


def run(config, output):
    torch.set_num_threads(config.threads)
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT/"output/tables/adaptive_cache"; cache.mkdir(parents=True, exist_ok=True)
    records, selections, diagnostics = [], [], []
    for name in config.distributions:
        distribution = distribution_for(name)
        r0, rh, revenue = targets_and_revenue(distribution, config.bidders, config.bandwidth)
        optimal_revenue = float(revenue(r0))
        offset = {"lognormal":0, "weibull":9000000}[name]
        for n in config.sample_sizes:
            for rep in range(config.repetitions):
                sample_seed = config.seed+offset+n*100+rep
                rng = np.random.default_rng(sample_seed)
                observed = distribution.rvs(size=(n,config.bidders), random_state=rng).max(axis=1)
                folds = make_folds(n, config.folds, sample_seed+31415926)
                reference = estimate_reserve_from_maxima(observed, config.bidders)
                hs = config.bandwidth*(n/500)**(-.2); hl = np.sqrt(2)*hs
                bandwidths = sorted(set([config.bandwidth, hs, hl]))
                full_data = {a:[] for a in ["original", "monotone"]}
                selected = {(a,h):[] for a in full_data for h in bandwidths}
                print(f"{name}, n={n}, repetition={rep+1}/{config.repetitions}", flush=True)
                for k, holdout in enumerate(folds):
                    mask = np.ones(n, dtype=bool); mask[holdout] = False
                    training = observed[mask]
                    fitting, validation = inner_split(training, sample_seed+(k+1)*7001, config.validation_fraction)
                    for ai, architecture in enumerate(full_data):
                        for stage, data in [("inner", fitting), ("outer", training)]:
                            seed = sample_seed+1000000+(k+1)*10000+ai*1000+(stage=="inner")*100
                            synthetic, detail = cached_fit(data, architecture, config, seed, cache)
                            diagnostics.append(dict(distribution=name, auctions=n, repetition=rep+1,
                                fold=k+1, architecture=architecture, stage=stage, **detail))
                            if stage == "inner":
                                for h in bandwidths:
                                    weight, risks = validation_weight(fitting, validation, synthetic, h, config.bidders, config.weight_grid)
                                    selected[(architecture,h)].append(weight)
                                    selections.append(dict(distribution=name, auctions=n, repetition=rep+1,
                                        fold=k+1, architecture=architecture, bandwidth=h, weight=weight,
                                        **{f"risk_{w:g}":risk for w,risk in zip(config.weight_grid,risks)}))
                            else:
                                full_data[architecture].append((observed[holdout],training,synthetic))
                        print(f"  fold {k+1} {architecture}: complete", flush=True)
                fits = {}
                fixed = [calibration_weight(len(t),config.pseudo_observations) for _,t,_ in full_data["original"]]
                for h in bandwidths:
                    for architecture in full_data:
                        moments = [FoldMoments(o,t,s,h,config.bidders) for o,t,s in full_data[architecture]]
                        for adaptive in [False,True]:
                            method = architecture.title()+(" adaptive" if adaptive else " fixed")
                            fits[(method,h)] = estimate(moments, selected[(architecture,h)] if adaptive else fixed, reference)
                    empirical = [FoldMoments(o,t,np.empty(0),h,config.bidders) for o,t,_ in full_data["original"]]
                    fits[("Empirical DML",h)] = estimate(empirical,[0.]*config.folds,reference)
                    full = FoldMoments(observed,observed,np.empty(0),h,config.bidders)
                    fits[("Smooth plug-in",h)] = estimate([full],[0.],reference,plugin=True)
                for method in METHODS[:-1]:
                    weight = (np.mean(selected[(method.split()[0].lower(), config.bandwidth)])
                              if "adaptive" in method else np.mean(fixed) if "fixed" in method else 0.)
                    row = record(n,rep,method,"Fixed bandwidth",fits[(method,config.bandwidth)],rh,r0,
                                 revenue,optimal_revenue,config.bandwidth,weight)
                    records.append(dict(distribution=name, **row))
                    small, large = fits[(method,hs)], fits[(method,hl)]
                    influence = 2*small["influence"]-large["influence"]
                    corrected = dict(estimate=2*small["estimate"]-large["estimate"],
                        standard_error=float(np.std(influence,ddof=1)/np.sqrt(n)),
                        roots=min(small["roots"],large["roots"]), multiple_roots=(small["roots"]>1 or large["roots"]>1),
                        score_residual=np.nan, jacobian=np.nan)
                    row = record(n,rep,method,"Exact reserve",corrected,r0,r0,revenue,optimal_revenue,hs,
                        np.mean(selected[(method.split()[0].lower(),hs)]) if "adaptive" in method else weight)
                    records.append(dict(distribution=name, **row))
                mle = lognormal_mle(observed,config.bidders)
                row = record(n,rep,"Lognormal MLE","Exact reserve",mle,r0,r0,revenue,optimal_revenue,0.,np.nan)
                records.append(dict(distribution=name, optimizer_success=mle["optimizer_success"], **row))
                for rows, filename in [(records,"raw"),(selections,"weights"),(diagnostics,"training")]:
                    pd.DataFrame(rows).to_csv(output/f"monotone_adaptive_{filename}.csv",index=False)
    (output/"monotone_adaptive_config.json").write_text(json.dumps(dict(**asdict(config),
        design="paired; inner 75/25 selection, outer 3-fold inference; all repetitions retained",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=__import__("sys").version,numpy=np.__version__,torch=torch.__version__,scipy=__import__("scipy").__version__),indent=2)+"\n")
    summarize(pd.DataFrame(records),pd.DataFrame(selections),output)


def combine(directories,output):
    configs=[json.loads((p/"monotone_adaptive_config.json").read_text()) for p in directories]
    keys=set(AdaptiveConfig.__dataclass_fields__)-{"sample_sizes","distributions"}
    if any(any(c[k]!=configs[0][k] for k in keys) for c in configs):
        raise ValueError("Incompatible experiment configurations")
    if any(c['source_sha256']!=configs[0]['source_sha256'] for c in configs):
        raise ValueError("Cannot combine different estimator source versions")
    output.mkdir(parents=True,exist_ok=True)
    for suffix,ids in [("raw",["distribution","auctions","repetition","procedure","method"]),
                       ("weights",["distribution","auctions","repetition","fold","architecture","bandwidth"]),
                       ("training",["distribution","auctions","repetition","fold","architecture","stage"])]:
        frame=pd.concat([pd.read_csv(p/f"monotone_adaptive_{suffix}.csv") for p in directories],ignore_index=True)
        if frame.duplicated(ids).any():
            raise ValueError(f"Duplicate cells in {suffix}")
        frame.to_csv(output/f"monotone_adaptive_{suffix}.csv",index=False)
    raw=pd.read_csv(output/"monotone_adaptive_raw.csv")
    config=dict(configs[0],sample_sizes=sorted(raw.auctions.unique().tolist()),distributions=sorted(raw.distribution.unique().tolist()))
    (output/"monotone_adaptive_config.json").write_text(json.dumps(config,indent=2)+"\n")
    summarize(raw,pd.read_csv(output/"monotone_adaptive_weights.csv"),output)


def summarize(raw, weights, output):
    rows=[]
    for (d,n,p,m), group in raw.groupby(["distribution","auctions","procedure","method"]):
        errors=group.error_target.to_numpy(); lo,hi=wilson(group.covers_target.sum(),len(group))
        rows.append(dict(distribution=d,auctions=n,procedure=p,method=m,repetitions=len(group),
            bias=errors.mean(),rmse=np.sqrt(np.mean(errors**2)),coverage=group.covers_target.mean(),
            coverage_low=lo,coverage_high=hi,interval_length=group.interval_length.mean(),
            revenue_regret=group.revenue_regret.mean(),multiple_root_rate=group.multiple_roots.mean(),
            missing_root_rate=group.missing_root.mean(),nonpositive_rate=(group.estimate<=0).mean()))
    summary=pd.DataFrame(rows); summary.to_csv(output/"monotone_adaptive.csv",index=False)
    paired=[]
    for (d,n,p), group in raw.groupby(["distribution","auctions","procedure"]):
        pivot=group.pivot(index="repetition",columns="method",values="error_target")**2
        for method in METHODS[:4]:
            for baseline in ["Original fixed","Empirical DML","Smooth plug-in"]:
                delta=pivot[method]-pivot[baseline]
                paired.append(dict(distribution=d,auctions=n,procedure=p,method=method,reference=baseline,
                    mse_difference=delta.mean(),paired_mcse=delta.std(ddof=1)/np.sqrt(len(delta))))
    pd.DataFrame(paired).to_csv(output/"monotone_adaptive_paired.csv",index=False)
    draw(summary,weights,output)
    lines=[r"\begin{tabular}{llrrrr}",r"\toprule",r"Design & Method & RMSE & Bias & Coverage & CI length \\",r"\midrule"]
    for (d,n),group in summary[summary.procedure=="Exact reserve"].groupby(["distribution","auctions"]):
        for j,m in enumerate(METHODS):
            row=group[group.method==m].iloc[0]
            label=f"{d.title()}, $n={n}$" if j==0 else ""
            lines.append(f"{label} & {m} & {row.rmse:.4f} & {row.bias:.4f} & {100*row.coverage:.0f}\\% & {row.interval_length:.4f} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (output/"monotone_adaptive.tex").write_text("\n".join(lines)+"\n")
    print(summary[summary.procedure=="Exact reserve"][["distribution","auctions","method","rmse","coverage"]].to_string(index=False),flush=True)


def draw(summary,weights,output):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    designs=list(summary.distribution.unique())
    fig,axes=plt.subplots(2,len(designs),figsize=(5.4*len(designs),7.7),layout="constrained",squeeze=False)
    for col,d in enumerate(designs):
        data=summary[(summary.distribution==d)&(summary.procedure=="Exact reserve")]
        ns=sorted(data.auctions.unique()); x=np.arange(len(ns))
        for mi,(method,color) in enumerate(zip(METHODS,COLORS)):
            part=data[data.method==method].sort_values("auctions")
            style=dict(color=color,marker=["o","s","v","D","^","P","X"][mi],lw=1.5,ms=5,
                       linestyle="--" if mi>=4 else "-",label=method)
            axes[0,col].plot(x,part.rmse,**style)
            axes[1,col].plot(x,part.revenue_regret*1e4,**style)
        axes[0,col].set_title(d.title()+": exact-reserve RMSE",loc="left",fontweight="bold")
        axes[1,col].set_title(d.title()+": revenue regret",loc="left",fontweight="bold")
        axes[0,col].set_ylabel("RMSE"); axes[1,col].set_ylabel(r"Revenue regret $\times 10^4$")
        for ax in axes[:,col]:
            ax.set_xticks(x,[f"{n:,}" for n in ns]); ax.set_xlabel("Number of auctions")
            ax.set_ylim(bottom=0); ax.grid(axis="y",alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle(f"Monotone generation and adaptive mixing | {int(summary.repetitions.min())} paired repetitions per cell",fontsize=13)
    fig.savefig(output/"monotone_adaptive.png",dpi=220,bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps",type=int,default=10)
    parser.add_argument("--sizes",type=int,nargs="+",default=[500,2000])
    parser.add_argument("--distributions",nargs="+",choices=["lognormal","weibull"],default=["lognormal","weibull"])
    parser.add_argument("--steps",type=int,default=1200)
    parser.add_argument("--seed",type=int,default=20260930)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"output/tables/monotone_adaptive")
    parser.add_argument("--combine-dirs",type=Path,nargs="+")
    args=parser.parse_args()
    if args.combine_dirs:
        combine(args.combine_dirs,args.output_dir)
    else:
        run(AdaptiveConfig(sample_sizes=tuple(args.sizes),distributions=tuple(args.distributions),
            repetitions=args.reps,training_steps=args.steps,seed=args.seed),args.output_dir)
