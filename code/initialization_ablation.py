#!/usr/bin/env python3
"""Paired retrospective ablation; prior estimator code remains unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

from baseline_direct_inversion import estimate_reserve_from_maxima
from calibrated_wgan_pilot import FoldMoments, calibration_weight, estimate, record, wilson
from crossfit_wgan_pilot import make_folds
from monotone_adaptive_pilot import (AdaptiveConfig, ROOT, cached_fit, distribution_for,
    initialize_monotone, inner_split, monotone_maxima, targets_and_revenue, validation_weight)
from run_monotone_release import EVALUATION_SEEDS

PREFIX = 'initialization_ablation'
NEW_METHODS = ['Initialization fixed', 'Initialization adaptive', 'Initialization matched']
OLD_METHODS = ['Monotone fixed', 'Monotone adaptive', 'Empirical DML', 'Smooth plug-in']
COMPARISONS = [('Monotone fixed', 'Initialization fixed', 'Fixed weights'),
    ('Monotone adaptive', 'Initialization matched', 'Matched weights'),
    ('Monotone adaptive', 'Initialization adaptive', 'Reselected weights')]


def initialized_maxima(observed, config):
    with torch.no_grad():
        model = initialize_monotone(observed, config.bidders)
        return monotone_maxima(model, config.quadrature_nodes, config.bidders, True).exp().numpy().ravel()


def authenticated_fit(observed, config, seed, cache, train_missing=False):
    """Mirror the frozen cache contract and fail closed if any field differs."""
    source = hashlib.sha256((ROOT/'code/monotone_adaptive_pilot.py').read_bytes() +
                           (ROOT/'code/wgan_gp_baseline.py').read_bytes()).hexdigest()
    settings = {k:getattr(config,k) for k in ['bidders','training_steps','generated_maxima',
                                           'quadrature_nodes','monotone_learning_rate']}
    metadata = json.dumps(dict(architecture='monotone', settings=settings, seed=seed,
        source=source, data=hashlib.sha256(observed.tobytes()).hexdigest()), sort_keys=True)
    path = cache/(hashlib.sha256(metadata.encode()).hexdigest()[:24]+'.npz')
    if not path.exists():
        if not train_missing:
            raise FileNotFoundError(f'Missing authenticated cache {path}; use --train-missing explicitly')
        cache.mkdir(parents=True, exist_ok=True)
        cached_fit(observed, 'monotone', config, seed, cache)
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['metadata']) != metadata:
            raise ValueError('Cache metadata mismatch')
        return saved['synthetic'].copy(), json.loads(str(saved['diagnostics'])), path.name


def corrected_fit(small, large, n):
    influence = 2*small['influence']-large['influence']
    return dict(estimate=2*small['estimate']-large['estimate'],
        standard_error=float(np.std(influence,ddof=1)/np.sqrt(n)),
        roots=min(small['roots'],large['roots']),
        multiple_roots=small['roots']>1 or large['roots']>1,
        score_residual=np.nan, jacobian=np.nan)


def select_archived_weights(archive, name, n, rep, k, bandwidths):
    part = archive[(archive.distribution==name)&(archive.auctions==n)&
                   (archive.repetition==rep+1)&(archive.fold==k+1)&
                   (archive.architecture=='monotone')]
    result = {}
    for h in bandwidths:
        rows = part[np.isclose(part.bandwidth,h,rtol=0,atol=1e-14)]
        if len(rows)!=1:
            raise ValueError('Missing or duplicated archived weight')
        result[h] = float(rows.iloc[0].weight)
    return result


def run(output, distributions=('lognormal','weibull'), sizes=(500,2000), train_missing=False):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    tables = ROOT/'paper/tables'
    prior_config = json.loads((tables/'monotone_adaptive_config.json').read_text())
    prior_source = hashlib.sha256((ROOT/'code/monotone_adaptive_pilot.py').read_bytes()).hexdigest()
    if prior_source != prior_config['source_sha256']:
        raise ValueError('Prior estimator source differs from the archived study')
    prior_raw = pd.read_csv(tables/'monotone_adaptive_raw.csv')
    prior_weights = pd.read_csv(tables/'monotone_adaptive_weights.csv')
    old = prior_raw[prior_raw.method.isin(OLD_METHODS)&prior_raw.distribution.isin(distributions)&
                    prior_raw.auctions.isin(sizes)].copy()
    records, weights, diagnostics, checks = [], [], [], []
    for name in distributions:
        config = AdaptiveConfig(distributions=(name,), sample_sizes=tuple(sizes), seed=EVALUATION_SEEDS[name])
        distribution = distribution_for(name)
        r0, rh, revenue = targets_and_revenue(distribution, config.bidders, config.bandwidth)
        optimal_revenue = float(revenue(r0))
        offset = {'lognormal':0,'weibull':9000000}[name]
        for n in sizes:
            hs = config.bandwidth*(n/500)**(-.2); hl = np.sqrt(2)*hs
            bandwidths = sorted(set([config.bandwidth,hs,hl]))
            for rep in range(config.repetitions):
                seed = config.seed+offset+n*100+rep
                observed = distribution.rvs(size=(n,config.bidders),random_state=np.random.default_rng(seed)).max(axis=1)
                folds = make_folds(n,config.folds,seed+31415926)
                reference = estimate_reserve_from_maxima(observed,config.bidders)
                data, trained_data = [], []
                adaptive = {h:[] for h in bandwidths}; matched = {h:[] for h in bandwidths}
                for k, holdout in enumerate(folds):
                    mask = np.ones(n,dtype=bool); mask[holdout]=False
                    training = observed[mask]
                    fitting, validation = inner_split(training,seed+(k+1)*7001,config.validation_fraction)
                    archived = select_archived_weights(prior_weights,name,n,rep,k,bandwidths)
                    for stage, values in [('inner',fitting),('outer',training)]:
                        fit_seed = seed+1000000+(k+1)*10000+1000+(stage=='inner')*100
                        trained, detail, cache_name = authenticated_fit(values,config,fit_seed,
                            ROOT/'output/tables/adaptive_cache',train_missing)
                        initial = initialized_maxima(values,config)
                        max_delta = float(np.max(np.abs(initial-trained)))
                        if detail['best_step']==0:
                            np.testing.assert_array_equal(initial,trained)
                        diagnostics.append(dict(distribution=name,auctions=n,repetition=rep+1,fold=k+1,
                            stage=stage,evaluation_base_seed=config.seed,cache_name=cache_name,
                            best_step=detail['best_step'],max_sample_difference=max_delta,
                            initialized_train_w1=wasserstein_distance(values,initial),
                            trained_train_w1=wasserstein_distance(values,trained),
                            initialized_holdout_w1=wasserstein_distance(observed[holdout],initial),
                            trained_holdout_w1=wasserstein_distance(observed[holdout],trained)))
                        if stage=='inner':
                            for h in bandwidths:
                                weight, risks = validation_weight(fitting,validation,initial,h,
                                    config.bidders,config.weight_grid)
                                # Recompute the old selection to verify the cache/data alignment.
                                old_weight, _ = validation_weight(fitting,validation,trained,h,
                                    config.bidders,config.weight_grid)
                                if old_weight != archived[h]:
                                    raise ValueError('Recomputed trained weight differs from archive')
                                adaptive[h].append(weight); matched[h].append(archived[h])
                                weights.append(dict(distribution=name,auctions=n,repetition=rep+1,fold=k+1,
                                    bandwidth=h,weight=weight,trained_weight=archived[h],
                                    evaluation_base_seed=config.seed,
                                    **{f'risk_{w:g}':r for w,r in zip(config.weight_grid,risks)}))
                        else:
                            data.append((observed[holdout],training,initial))
                            trained_data.append((observed[holdout],training,trained))
                fixed = [calibration_weight(len(t),config.pseudo_observations) for _,t,_ in data]
                fits = {}; selected = {}
                for h in bandwidths:
                    moments = [FoldMoments(o,t,s,h,config.bidders) for o,t,s in data]
                    for method, ws in zip(NEW_METHODS,[fixed,adaptive[h],matched[h]]):
                        selected[(method,h)] = ws
                        fits[(method,h)] = estimate(moments,ws,reference)
                    if rep==0:
                        trained_moments = [FoldMoments(o,t,s,h,config.bidders) for o,t,s in trained_data]
                        for method, ws in [('Monotone fixed',fixed),('Monotone adaptive',matched[h])]:
                            fits[(method,h)] = estimate(trained_moments,ws,reference)
                for method in NEW_METHODS + (OLD_METHODS[:2] if rep==0 else []):
                    for procedure, fit, target, h in [
                        ('Fixed bandwidth',fits[(method,config.bandwidth)],rh,config.bandwidth),
                        ('Exact reserve',corrected_fit(fits[(method,hs)],fits[(method,hl)],n),r0,hs)]:
                        if method in NEW_METHODS:
                            row = record(n,rep,method,procedure,fit,target,r0,revenue,optimal_revenue,h,
                                         np.mean(selected[(method,h)]))
                            records.append(dict(distribution=name,evaluation_base_seed=config.seed,**row))
                        else:
                            archived = old[(old.distribution==name)&(old.auctions==n)&
                                (old.repetition==rep+1)&(old.method==method)&(old.procedure==procedure)]
                            if len(archived)!=1:
                                raise ValueError('Missing paired trained comparator')
                            for metric in ['estimate','standard_error']:
                                delta = float(fit[metric]-archived.iloc[0][metric])
                                np.testing.assert_allclose(fit[metric],archived.iloc[0][metric],atol=2e-9,rtol=0)
                                checks.append(dict(distribution=name,auctions=n,repetition=rep+1,
                                    method=method,procedure=procedure,metric=metric,difference=delta))
                print(f'{name} n={n} repetition={rep+1}/10 complete (no new optimizer steps)',flush=True)
                pd.DataFrame(records).to_csv(output/f'{PREFIX}_new_raw.csv',index=False)
    raw = pd.concat([old,pd.DataFrame(records)],ignore_index=True)
    for frame, suffix in [(raw,'raw'),(pd.DataFrame(weights),'weights'),
                          (pd.DataFrame(diagnostics),'diagnostics'),(pd.DataFrame(checks),'checks')]:
        frame.to_csv(output/f'{PREFIX}_{suffix}.csv',index=False)
    provenance = dict(design='retrospective paired mechanism ablation; no new independent evaluation',
        distributions=list(distributions),sample_sizes=list(sizes),repetitions=10,
        evaluation_seed_by_distribution=EVALUATION_SEEDS,prior_estimator_sha256=prior_source,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prior_raw_sha256=hashlib.sha256((tables/'monotone_adaptive_raw.csv').read_bytes()).hexdigest(),
        prior_weights_sha256=hashlib.sha256((tables/'monotone_adaptive_weights.csv').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((ROOT/'docs/initialization_ablation.md').read_bytes()).hexdigest(),
        initialization_optimizer_steps=0,quadrature_nodes=32768,
        python=__import__('sys').version,numpy=np.__version__,torch=torch.__version__)
    (output/f'{PREFIX}_config.json').write_text(json.dumps(provenance,indent=2)+'\n')
    summarize(raw,output)


def summarize(raw,output):
    summary=[]; paired=[]
    for (d,n,p,m),group in raw.groupby(['distribution','auctions','procedure','method']):
        low,high=wilson(group.covers_target.sum(),len(group))
        summary.append(dict(distribution=d,auctions=n,procedure=p,method=m,repetitions=len(group),
            rmse=np.sqrt(np.mean(group.error_target**2)),bias=group.error_target.mean(),
            coverage=group.covers_target.mean(),coverage_low=low,coverage_high=high,
            interval_length=group.interval_length.mean(),revenue_regret=group.revenue_regret.mean(),
            missing_root_rate=group.missing_root.mean(),multiple_root_rate=group.multiple_roots.mean()))
    for (d,n,p),group in raw.groupby(['distribution','auctions','procedure']):
        pivot=group.pivot(index='repetition',columns='method',values='error_target')**2
        for trained,initial,label in COMPARISONS:
            delta=pivot[trained]-pivot[initial]
            paired.append(dict(distribution=d,auctions=n,procedure=p,comparison=label,
                trained=trained,initial=initial,mse_difference=delta.mean(),
                paired_mcse=delta.std(ddof=1)/np.sqrt(len(delta))))
    summary=pd.DataFrame(summary); paired=pd.DataFrame(paired)
    summary.to_csv(output/f'{PREFIX}.csv',index=False)
    paired.to_csv(output/f'{PREFIX}_paired.csv',index=False)
    methods=['Initialization fixed','Monotone fixed','Initialization adaptive','Monotone adaptive','Initialization matched']
    lines=[r'\begin{tabular}{lrccccc}',r'\toprule',
        r' & & \multicolumn{2}{c}{Fixed weights} & \multicolumn{2}{c}{Reselected weights} & Matched \\',
        r'Design & $n$ & Init. & Trained & Init. & Trained & Init. \\',r'\midrule']
    for (d,n),group in summary[summary.procedure=='Exact reserve'].groupby(['distribution','auctions']):
        vals=[float(group[group.method==m].iloc[0].rmse) for m in methods]
        lines.append(f'{d.title()} & {n} & '+' & '.join(f'{v:.4f}' for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (output/f'{PREFIX}.tex').write_text('\n'.join(lines)+'\n')
    draw(paired,output)
    print(summary[summary.procedure=='Exact reserve'][['distribution','auctions','method','rmse','coverage']].to_string(index=False),flush=True)


def draw(paired,output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    part=paired[paired.procedure=='Exact reserve']
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained',sharey=True)
    cells=list(part[['distribution','auctions']].drop_duplicates().itertuples(index=False,name=None))
    labels=[f'{d.title()}, n={n:,}' for d,n in cells]
    for ax,(_,_,label),color in zip(axes,COMPARISONS,['#7975A5','#1677A0','#39836C']):
        data=part[part.comparison==label].set_index(['distribution','auctions']).loc[cells]
        ax.errorbar(data.mse_difference*1e3,np.arange(len(cells)),
                    xerr=1.96*data.paired_mcse*1e3,fmt='o',capsize=3,color=color)
        ax.axvline(0,color='#555555',lw=1,ls='--'); ax.set_title(label,loc='left',fontweight='bold')
        ax.set_xlabel(r'Trained minus initialized MSE ($\times 10^{-3}$)'); ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(np.arange(len(cells)),labels); axes[0].invert_yaxis()
    fig.suptitle('Adversarial-training contribution | paired retrospective diagnostic',fontsize=12)
    fig.savefig(output/f'{PREFIX}.png',dpi=220,bbox_inches='tight'); plt.close(fig)


def combine(directories,output):
    configs=[json.loads((p/f'{PREFIX}_config.json').read_text()) for p in directories]
    for key in ['source_sha256','prior_estimator_sha256','prior_raw_sha256','prior_weights_sha256','protocol_sha256']:
        if len({c[key] for c in configs})!=1:
            raise ValueError(f'Inconsistent {key}')
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ['raw','weights','diagnostics','checks']:
        frame=pd.concat([pd.read_csv(p/f'{PREFIX}_{suffix}.csv') for p in directories],ignore_index=True)
        if suffix=='raw' and frame.duplicated(['distribution','auctions','repetition','procedure','method']).any():
            raise ValueError('Overlapping design cells')
        frame.to_csv(output/f'{PREFIX}_{suffix}.csv',index=False)
    raw=pd.read_csv(output/f'{PREFIX}_raw.csv')
    config=dict(configs[0],distributions=sorted(raw.distribution.unique()),sample_sizes=sorted(raw.auctions.unique().tolist()))
    (output/f'{PREFIX}_config.json').write_text(json.dumps(config,indent=2)+'\n')
    summarize(raw,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/tables/initialization_ablation')
    parser.add_argument('--distributions',nargs='+',choices=['lognormal','weibull'],default=['lognormal','weibull'])
    parser.add_argument('--sizes',nargs='+',type=int,choices=[500,2000],default=[500,2000])
    parser.add_argument('--train-missing',action='store_true')
    parser.add_argument('--combine-dirs',nargs='+',type=Path)
    args=parser.parse_args()
    if args.combine_dirs:
        combine(args.combine_dirs,args.output_dir)
    else:
        run(args.output_dir,tuple(args.distributions),tuple(args.sizes),args.train_missing)
