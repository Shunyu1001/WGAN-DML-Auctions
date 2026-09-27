#!/usr/bin/env python3
"""Run or combine the final pilot with non-overlapping evaluation cohorts.

The original lognormal seed 20260930 overlaps eight of ten dataset seeds in
the earlier 20260928 calibration study. Its results are retained separately;
the final lognormal cohort uses 90260930. Weibull has its own DGP seed offset
and retains its prespecified 20260930 cohort. No estimator setting changes.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

from monotone_adaptive_pilot import ROOT, AdaptiveConfig, run, summarize

EVALUATION_SEEDS = {"lognormal":90260930,"weibull":20260930}


def combine_release(directories,output):
    configs=[json.loads((p/'monotone_adaptive_config.json').read_text()) for p in directories]
    keys=set(AdaptiveConfig.__dataclass_fields__)-{'sample_sizes','distributions','seed'}
    for c in configs:
        if any(c[k]!=configs[0][k] for k in keys):
            raise ValueError('Estimator settings differ across cohorts')
        if c['source_sha256']!=configs[0]['source_sha256']:
            raise ValueError('Estimator source versions differ across cohorts')
        if any(c['seed']!=EVALUATION_SEEDS[d] for d in c['distributions']):
            raise ValueError('Unapproved evaluation seed for this distribution')
    output.mkdir(parents=True,exist_ok=True)
    for suffix,ids in [('raw',['distribution','auctions','repetition','procedure','method']),
                      ('weights',['distribution','auctions','repetition','fold','architecture','bandwidth']),
                      ('training',['distribution','auctions','repetition','fold','architecture','stage'])]:
        parts=[]
        for directory,c in zip(directories,configs):
            frame=pd.read_csv(directory/f'monotone_adaptive_{suffix}.csv')
            frame['evaluation_base_seed']=c['seed']
            if not set(frame.distribution).issubset(c['distributions']):
                raise ValueError('Data/config distribution mismatch')
            parts.append(frame)
        frame=pd.concat(parts,ignore_index=True)
        if frame.duplicated(ids).any():
            raise ValueError(f'Duplicate records: {suffix}')
        frame.to_csv(output/f'monotone_adaptive_{suffix}.csv',index=False)
    raw=pd.read_csv(output/'monotone_adaptive_raw.csv')
    config=dict(configs[0],seed=AdaptiveConfig().seed,
        evaluation_seed_by_distribution=EVALUATION_SEEDS,
        sample_sizes=sorted(raw.auctions.unique().tolist()),
        distributions=sorted(raw.distribution.unique().tolist()),
        seed_note='seed is the nominal code default; evaluation_base_seed in every row records the actual cohort seed. The overlapping lognormal cohort is retained as a separate audit, not pooled.')
    (output/'monotone_adaptive_config.json').write_text(json.dumps(config,indent=2)+'\n')
    summarize(raw,pd.read_csv(output/'monotone_adaptive_weights.csv'),output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/tables/monotone_release')
    parser.add_argument('--combine-dirs',type=Path,nargs='+')
    args=parser.parse_args()
    if args.combine_dirs:
        combine_release(args.combine_dirs,args.output_dir)
    else:
        directories=[]
        for name,seed in EVALUATION_SEEDS.items():
            destination=args.output_dir/name
            run(AdaptiveConfig(distributions=(name,),seed=seed),destination)
            directories.append(destination)
        combine_release(directories,args.output_dir)
