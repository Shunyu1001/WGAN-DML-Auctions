"""Deterministic invariants of the initialization-only comparison."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from initialization_ablation import (AdaptiveConfig, authenticated_fit, corrected_fit,
    initialized_maxima, select_archived_weights)
from monotone_adaptive_pilot import cached_fit, train_monotone


class InitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data=np.exp(np.linspace(-.1,1,80))
        self.config=replace(AdaptiveConfig(),training_steps=0,quadrature_nodes=256)

    def test_initialized_distribution_equals_step_zero(self):
        initial=initialized_maxima(self.data,self.config)
        trained,detail=train_monotone(self.data,self.config,123)
        np.testing.assert_array_equal(initial,trained)
        self.assertEqual(detail['best_step'],0)

    def test_initialization_is_deterministic_without_rng(self):
        torch.manual_seed(12)
        first=initialized_maxima(self.data,self.config)
        torch.manual_seed(888)
        np.testing.assert_array_equal(first,initialized_maxima(self.data,self.config))
        self.assertTrue(np.all(np.diff(first)>0))

    def test_authenticated_cache_requires_exact_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            with self.assertRaises(FileNotFoundError):
                authenticated_fit(self.data,self.config,123,path)
            expected,_=cached_fit(self.data,'monotone',self.config,123,path)
            actual,detail,_=authenticated_fit(self.data,self.config,123,path)
            np.testing.assert_array_equal(expected,actual)
            self.assertEqual(detail['best_step'],0)
            with self.assertRaises(FileNotFoundError):
                authenticated_fit(self.data*1.01,self.config,123,path)

    def test_correction_retains_influence_covariance(self):
        small=dict(estimate=.8,influence=np.array([-2.,-1.,1.,2.]),roots=1)
        large=dict(estimate=.9,influence=2*small['influence'],roots=2)
        result=corrected_fit(small,large,4)
        self.assertAlmostEqual(result['estimate'],.7)
        self.assertEqual(result['standard_error'],0)
        self.assertTrue(result['multiple_roots'])

    def test_archive_lookup_rejects_ambiguous_matching(self):
        frame=pd.DataFrame([dict(distribution='lognormal',auctions=500,repetition=1,
            fold=1,architecture='monotone',bandwidth=.15,weight=.5)])
        self.assertEqual(select_archived_weights(frame,'lognormal',500,0,0,[.15]),{.15:.5})
        with self.assertRaises(ValueError):
            select_archived_weights(pd.concat([frame,frame]),'lognormal',500,0,0,[.15])


if __name__=='__main__':
    unittest.main()
