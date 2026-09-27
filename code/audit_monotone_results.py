#!/usr/bin/env python3
"""Check the archived nested-validation study without training any networks."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def audit(directory,require_full=False):
    config=json.loads((directory/'monotone_adaptive_config.json').read_text())
    if require_full:
        assert sorted(config['distributions'])==['lognormal','weibull']
        assert sorted(config['sample_sizes'])==[500,2000]
        assert config['repetitions']==10
        assert config['evaluation_seed_by_distribution']=={'lognormal':90260930,'weibull':20260930}
    raw=pd.read_csv(directory/'monotone_adaptive_raw.csv')
    summary=pd.read_csv(directory/'monotone_adaptive.csv')
    weights=pd.read_csv(directory/'monotone_adaptive_weights.csv')
    training=pd.read_csv(directory/'monotone_adaptive_training.csv')
    cells=len(config['sample_sizes'])*len(config['distributions'])
    repetitions=config['repetitions']
    assert len(raw)==cells*repetitions*13, 'Missing or extra estimator rows'
    identifiers=['distribution','auctions','repetition','procedure','method']
    assert not raw.duplicated(identifiers).any(), 'Duplicate estimator rows'
    if 'evaluation_seed_by_distribution' in config:
        expected_seeds=raw.distribution.map(config['evaluation_seed_by_distribution'])
        np.testing.assert_array_equal(raw.evaluation_base_seed,expected_seeds)
        primary=raw[raw.distribution=='lognormal']
        primary_seeds=set(primary.evaluation_base_seed+primary.auctions*100+primary.repetition-1)
        prior={20260928+n*100+r for n in [500,2000,10000] for r in range(10)}
        assert not primary_seeds.intersection(prior), 'Primary evaluation reuses prior calibration samples'
        overlap_path=directory/'monotone_overlap_raw.csv'
        if overlap_path.exists():
            overlap=pd.read_csv(overlap_path)
            assert len(overlap)==260, 'Incomplete retained overlap cohort'
            assert not overlap.duplicated(identifiers).any()
            old_seeds=set(overlap.evaluation_base_seed+overlap.auctions*100+overlap.repetition-1)
            assert not primary_seeds.intersection(old_seeds)
    assert np.isfinite(raw[['estimate','standard_error','error_target','interval_length']]).all().all()
    assert (raw.standard_error>0).all()
    np.testing.assert_allclose(raw.error_target,raw.estimate-raw.target,atol=1e-12)
    np.testing.assert_allclose(raw.interval_length,3.92*raw.standard_error,atol=1e-12)
    expected=(abs(raw.error_target)<=1.96*raw.standard_error).to_numpy()
    np.testing.assert_array_equal(raw.covers_target.to_numpy(),expected)
    assert set(weights.weight.unique()).issubset(config['weight_grid'])
    candidates=np.array(config['weight_grid'])
    risks=weights[[f'risk_{w:g}' for w in candidates]].to_numpy()
    assert np.isfinite(risks).all()
    np.testing.assert_array_equal(weights.weight.to_numpy(),candidates[np.argmin(risks,axis=1)])
    expected_weights=0
    for n in config['sample_sizes']:
        hs=config['bandwidth']*(n/500)**(-.2)
        count=len(set([config['bandwidth'],hs,np.sqrt(2)*hs]))
        expected_weights += len(config['distributions'])*repetitions*config['folds']*2*count
    assert len(weights)==expected_weights, 'Missing weight-selection records'
    assert len(training)==cells*repetitions*config['folds']*4
    assert not training.duplicated(['distribution','auctions','repetition','fold','architecture','stage']).any()
    assert (training.training_seconds>0).all()
    mono=training[training.architecture=='monotone']
    assert (mono.generator_parameters==11).all()
    assert (mono.final_w1<=mono.initial_w1+1e-12).all()
    assert raw[raw.method=='Lognormal MLE'].optimizer_success.all(), 'MLE did not converge'
    for _,row in summary.iterrows():
        group=raw[(raw.distribution==row.distribution)&(raw.auctions==row.auctions)&
                  (raw.method==row.method)&(raw.procedure==row.procedure)]
        assert len(group)==repetitions
        np.testing.assert_allclose(row.rmse,np.sqrt(np.mean(group.error_target**2)),rtol=1e-12)
        np.testing.assert_allclose(row.coverage,group.covers_target.mean(),atol=1e-12)
    print(f'Nested-study audit passed: {len(raw)} estimates, {len(training)} fits, {len(weights)} weight selections.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,nargs='?',default=Path(__file__).resolve().parents[1]/'paper/tables')
    parser.add_argument('--require-full',action='store_true')
    args=parser.parse_args()
    audit(args.directory,args.require_full)
