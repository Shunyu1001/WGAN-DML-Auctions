#!/usr/bin/env python3
"""Audit the complete initialization mechanism study without training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from initialization_ablation import PREFIX, ROOT, NEW_METHODS, OLD_METHODS


def audit(directory):
    config=json.loads((directory/f'{PREFIX}_config.json').read_text())
    assert sorted(config['distributions'])==['lognormal','weibull']
    assert sorted(config['sample_sizes'])==[500,2000]
    assert config['repetitions']==10 and config['initialization_optimizer_steps']==0
    assert config['evaluation_seed_by_distribution']=={'lognormal':90260930,'weibull':20260930}
    for key,path in [('source_sha256',ROOT/'code/initialization_ablation.py'),
                     ('prior_estimator_sha256',ROOT/'code/monotone_adaptive_pilot.py'),
                     ('prior_raw_sha256',ROOT/'paper/tables/monotone_adaptive_raw.csv'),
                     ('prior_weights_sha256',ROOT/'paper/tables/monotone_adaptive_weights.csv'),
                     ('protocol_sha256',ROOT/'docs/initialization_ablation.md')]:
        assert config[key]==hashlib.sha256(path.read_bytes()).hexdigest(),f'Source changed: {path}'
    raw=pd.read_csv(directory/f'{PREFIX}_raw.csv')
    weights=pd.read_csv(directory/f'{PREFIX}_weights.csv')
    diagnostics=pd.read_csv(directory/f'{PREFIX}_diagnostics.csv')
    checks=pd.read_csv(directory/f'{PREFIX}_checks.csv')
    summary=pd.read_csv(directory/f'{PREFIX}.csv')
    paired=pd.read_csv(directory/f'{PREFIX}_paired.csv')
    ids=['distribution','auctions','repetition','procedure','method']
    assert len(raw)==560 and not raw.duplicated(ids).any()
    assert set(raw.method)==set(NEW_METHODS+OLD_METHODS)
    assert np.isfinite(raw[['estimate','standard_error','error_target','interval_length']]).all().all()
    assert (raw.standard_error>0).all()
    np.testing.assert_array_equal(raw.evaluation_base_seed,raw.distribution.map(config['evaluation_seed_by_distribution']))
    np.testing.assert_allclose(raw.error_target,raw.estimate-raw.target,atol=1e-12)
    np.testing.assert_allclose(raw.interval_length,3.92*raw.standard_error,atol=1e-12)
    np.testing.assert_array_equal(raw.covers_target,abs(raw.error_target)<=1.96*raw.standard_error)
    old=pd.read_csv(ROOT/'paper/tables/monotone_adaptive_raw.csv')
    baseline=old[old.method.isin(OLD_METHODS)].set_index(ids).sort_index()
    carried=raw[raw.method.isin(OLD_METHODS)].set_index(ids).sort_index()
    pd.testing.assert_index_equal(baseline.index,carried.index)
    np.testing.assert_allclose(baseline[['estimate','standard_error']],carried[['estimate','standard_error']],atol=1e-13,rtol=0)
    assert len(weights)==300 and not weights.duplicated(['distribution','auctions','repetition','fold','bandwidth']).any()
    candidates=np.array([0.,.25,.5,1.])
    risks=weights[[f'risk_{w:g}' for w in candidates]].to_numpy()
    np.testing.assert_array_equal(weights.weight,candidates[risks.argmin(axis=1)])
    assert len(diagnostics)==240 and not diagnostics.duplicated(['distribution','auctions','repetition','fold','stage']).any()
    assert diagnostics.cache_name.nunique()==240
    assert (diagnostics[diagnostics.best_step==0].max_sample_difference==0).all()
    assert len(checks)==32 and checks.difference.abs().max()<2e-9
    assert len(summary)==56 and len(paired)==24
    for _,row in summary.iterrows():
        group=raw[(raw.distribution==row.distribution)&(raw.auctions==row.auctions)&
                  (raw.procedure==row.procedure)&(raw.method==row.method)]
        assert len(group)==10
        np.testing.assert_allclose(row.rmse,np.sqrt(np.mean(group.error_target**2)),atol=1e-12)
        np.testing.assert_allclose(row.coverage,group.covers_target.mean(),atol=1e-12)
    for _,row in paired.iterrows():
        part=raw[(raw.distribution==row.distribution)&(raw.auctions==row.auctions)&(raw.procedure==row.procedure)]
        pivot=part.pivot(index='repetition',columns='method',values='error_target')**2
        delta=pivot[row.trained]-pivot[row.initial]
        np.testing.assert_allclose(row.mse_difference,delta.mean(),atol=1e-12)
        np.testing.assert_allclose(row.paired_mcse,delta.std(ddof=1)/np.sqrt(10),atol=1e-12)
    print('Initialization audit passed: 560 estimates, 240 authenticated fits, 300 selections, 32 comparator checks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,nargs='?',default=ROOT/'paper/tables')
    audit(parser.parse_args().directory)
