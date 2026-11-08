"""Numerical checks for the calibration, score, and interval implementation."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pandas as pd

from calibrated_wgan_pilot import (
    METHODS, CalibrationConfig, FoldMoments, calibration_weight, estimate,
    scores, wilson, richardson_fit, record, summarize, run,
)
from orthogonal_wgan_dml import orthogonal_scores, score_grid_for_fold


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(671)
        self.training = np.exp(rng.normal(.45, .33, 180))
        self.observed = np.exp(rng.normal(.45, .33, 120))
        self.synthetic = np.exp(rng.normal(.35, .38, 500))
        self.fold = FoldMoments(self.observed, self.training, self.synthetic, .15, 5)

    def test_endpoints_match_existing_scores(self):
        for weight, nuisance in [(0, self.training), (1, self.synthetic)]:
            expected = orthogonal_scores(.83, self.observed, nuisance, .15, 5)
            np.testing.assert_allclose(self.fold.individual_scores(.83, weight), expected)
            grid = np.linspace(.65, 1.05, 9)
            np.testing.assert_allclose(self.fold.score_mean(grid, weight),
                score_grid_for_fold(grid, self.observed, nuisance, .15, 5))

    def test_distribution_and_density_are_calibrated_together(self):
        grid = np.linspace(.6, 1.1, 9)
        a, g = self.fold.nuisance(grid, .2)
        step = 1e-5
        ap, _ = self.fold.nuisance(grid+step, .2)
        am, _ = self.fold.nuisance(grid-step, .2)
        np.testing.assert_allclose((ap-am)/(2*step),g,rtol=1e-7,atol=1e-8)
        self.assertTrue(np.all(np.diff(a)>0))

    def test_nuisance_uses_training_not_heldout(self):
        changed = FoldMoments(self.observed*3, self.training, self.synthetic, .15, 5)
        np.testing.assert_array_equal(self.fold.nuisance(.8,.2),changed.nuisance(.8,.2))

    def test_fixed_weight_vanishes_and_root_is_refined(self):
        self.assertAlmostEqual(calibration_weight(950,50),.05)
        self.assertLess(calibration_weight(9950,50),.006)
        fit = estimate([self.fold],[.1],.8)
        self.assertGreater(fit['roots'],0)
        self.assertLess(abs(fit['score_residual']),1e-8)
        self.assertGreater(fit['standard_error'],0)
        self.assertEqual(fit['status'],'ok')

    def test_analytic_score_derivative_matches_total_finite_difference(self):
        grid=np.array([.72,.83,.94])
        step=1e-6
        for weight in [0,.2,1]:
            numeric=(self.fold.score_mean(grid+step,weight)-
                     self.fold.score_mean(grid-step,weight))/(2*step)
            np.testing.assert_allclose(self.fold.score_derivative_mean(grid,weight),
                                       numeric,rtol=2e-7,atol=2e-8)

    def test_plugin_analytic_derivative_matches_total_finite_difference(self):
        full=FoldMoments(self.training,self.training,np.empty(0),.15,5)
        grid=np.array([.72,.83,.94]); step=1e-6
        def moment(theta):
            a,g=full.nuisance(theta,0)
            from orthogonal_wgan_dml import reserve_moment_and_derivatives
            return reserve_moment_and_derivatives(theta,a,g,5)[0]
        numeric=(moment(grid+step)-moment(grid-step))/(2*step)
        np.testing.assert_allclose(full.score_derivative_mean(grid,0,plugin=True),
                                   numeric,rtol=2e-7,atol=2e-8)

    def test_plugin_requires_one_full_sample_fold(self):
        full=FoldMoments(self.training,self.training,np.empty(0),.15,5)
        with self.assertRaises(ValueError):
            estimate([full,full],[0,0],plugin=True)
        with self.assertRaises(ValueError):
            estimate([self.fold],[0],plugin=True)

    def test_plugin_requires_zero_wgan_weight(self):
        full=FoldMoments(self.training,self.training,self.synthetic,.15,5)
        with self.assertRaises(ValueError):
            estimate([full],[.1],plugin=True)
        fit=estimate([full],[0],plugin=True)
        self.assertIn(fit['status'],['ok','failed'])

    def test_orthogonality_at_truth(self):
        # Use the evaluation distribution itself as the population quadrature.
        a,g = self.fold.nuisance(.83,0)
        for component in [0,1]:
            step=1e-6
            plus=np.array([a[0],g[0]]); minus=plus.copy()
            plus[component]+=step; minus[component]-=step
            derivative=(scores(.83,self.training,*plus,.15,5).mean()-
                        scores(.83,self.training,*minus,.15,5).mean())/(2*step)
            self.assertLess(abs(derivative),1e-5)

    def test_wilson_includes_boundary(self):
        lo,hi=wilson(0,10)
        self.assertAlmostEqual(lo,0)
        self.assertGreater(hi,.2)


class ControlledScoreFold:
    """A controlled smooth score for numerical failure-policy tests."""
    def __init__(self, function, derivative, cdf=.5, variance=True):
        self.function=function; self.derivative=derivative
        self.cdf=cdf; self.variance=variance
        self.observed=np.array([0.,1.,2.,3.])

    def nuisance(self,theta,weight):
        theta=np.atleast_1d(theta)
        return np.full(theta.shape,self.cdf),np.full(theta.shape,.1)

    def score_mean(self,theta,weight):
        return self.function(np.atleast_1d(theta))

    def score_derivative_mean(self,theta,weight,plugin=False):
        return self.derivative(np.atleast_1d(theta))

    def individual_scores(self,theta,weight):
        noise=np.array([-.2,-.1,.1,.2]) if self.variance else np.zeros(4)
        return float(self.function(np.array([theta]))[0])+noise


class FailurePolicyTests(unittest.TestCase):
    def assert_failed(self,fit,reason):
        self.assertEqual(fit['status'],'failed')
        self.assertEqual(fit['failure_reason'],reason)
        self.assertTrue(np.isnan(fit['estimate']))
        self.assertTrue(np.isnan(fit['standard_error']))
        self.assertEqual(len(fit['influence']),4)
        self.assertTrue(np.isnan(fit['influence']).all())

    def test_reference_never_selects_a_root(self):
        fold=ControlledScoreFold(lambda x:.8-x,lambda x:-np.ones_like(x))
        a=estimate([fold],[0],reference=.1)
        b=estimate([fold],[0],reference=100.)
        self.assertEqual(a['status'],'ok')
        self.assertAlmostEqual(a['estimate'],.8)
        self.assertEqual(a['estimate'],b['estimate'])
        self.assertAlmostEqual(a['jacobian'],-1.)

    def test_no_root_never_gets_a_wald_interval(self):
        fold=ControlledScoreFold(lambda x:np.ones_like(x),lambda x:np.zeros_like(x))
        self.assert_failed(estimate([fold],[0]),'no_root')

    def test_multiple_roots_are_not_selected_by_reference(self):
        fold=ControlledScoreFold(lambda x:(x-.6)*(x-.8)*(x-1.),
                                 lambda x:3*x*x-4.8*x+1.88)
        self.assert_failed(estimate([fold],[0],reference=.6),'multiple_roots')

    def test_boundary_root_is_failure(self):
        fold=ControlledScoreFold(lambda x:.45-x,lambda x:-np.ones_like(x))
        self.assert_failed(estimate([fold],[0]),'boundary_root')

    def test_upward_root_is_failure(self):
        fold=ControlledScoreFold(lambda x:x-.8,lambda x:np.ones_like(x))
        self.assert_failed(estimate([fold],[0]),'boundary_sign_failure')

    def test_weak_jacobian_is_failure(self):
        fold=ControlledScoreFold(lambda x:1e-8*(.8-x),
                                 lambda x:np.full_like(x,-1e-8))
        self.assert_failed(estimate([fold],[0]),'weak_or_nonnegative_jacobian')

    def test_cdf_clipping_is_failure(self):
        fold=ControlledScoreFold(lambda x:.8-x,lambda x:-np.ones_like(x),cdf=1e-9)
        self.assert_failed(estimate([fold],[0]),'cdf_clip_active')

    def test_nonfinite_score_is_failure(self):
        fold=ControlledScoreFold(lambda x:np.full_like(x,np.nan),
                                 lambda x:-np.ones_like(x))
        self.assert_failed(estimate([fold],[0]),'nonfinite_score')

    def test_degenerate_variance_is_failure(self):
        fold=ControlledScoreFold(lambda x:.8-x,lambda x:-np.ones_like(x),variance=False)
        self.assert_failed(estimate([fold],[0]),'nonpositive_or_nonfinite_variance')

    def test_tolerances_can_be_prespecified(self):
        fold=ControlledScoreFold(lambda x:.8-x,lambda x:-np.ones_like(x))
        fit=estimate([fold],[0],root_tolerance=1e-12,score_tolerance=1e-10)
        self.assertEqual(fit['root_tolerance'],1e-12)
        self.assertEqual(fit['score_tolerance'],1e-10)

    def test_existing_experiment_is_never_overwritten(self):
        with TemporaryDirectory() as folder:
            output=Path(folder)
            (output/'calibrated_wgan_raw.csv').write_text('existing results\n')
            with self.assertRaises(FileExistsError), patch('calibrated_wgan_pilot.train_wgan_gp') as train:
                run(CalibrationConfig(),output)
            train.assert_not_called()
            self.assertEqual((output/'calibrated_wgan_raw.csv').read_text(),'existing results\n')

    def test_richardson_propagates_failed_bandwidth(self):
        good=estimate([ControlledScoreFold(lambda x:.8-x,lambda x:-np.ones_like(x))],[0])
        bad=estimate([ControlledScoreFold(lambda x:np.ones_like(x),lambda x:np.zeros_like(x))],[0])
        combined=richardson_fit(good,bad)
        self.assert_failed(combined,'bandwidth_fit_failed:large:no_root')

    def test_summary_keeps_failures_and_labels_conditional_errors(self):
        good=estimate([ControlledScoreFold(lambda x:.8-x,lambda x:-np.ones_like(x))],[0])
        bad=estimate([ControlledScoreFold(lambda x:np.ones_like(x),lambda x:np.zeros_like(x))],[0])
        rows=[]
        for method in METHODS:
            for repetition in [0,1]:
                fit=bad if method==METHODS[1] and repetition==1 else good
                rows.append(record(500,repetition,method,'Fixed bandwidth',fit,.8,.8,
                                   lambda r:1.,1.,.15,.1))
        raw=pd.DataFrame(rows)
        with TemporaryDirectory() as folder, patch('calibrated_wgan_pilot.draw'):
            summarize(raw,Path(folder))
            summary=pd.read_csv(Path(folder)/'calibrated_wgan.csv')
            paired=pd.read_csv(Path(folder)/'calibrated_wgan_paired.csv')
        calibrated=summary.loc[summary.method==METHODS[1]].iloc[0]
        self.assertEqual(len(raw),8)
        self.assertEqual(calibrated.repetitions,2)
        self.assertEqual(calibrated.valid_repetitions,1)
        self.assertEqual(calibrated.failure_rate,.5)
        self.assertEqual(calibrated.coverage,.5)
        self.assertEqual(calibrated.conditional_coverage,1.)
        self.assertIn('conditional on valid',calibrated.error_summary_scope)
        self.assertTrue((paired.paired_repetitions==1).all())

    def test_all_failed_summary_has_no_artificial_rmse_or_conditional_coverage(self):
        bad=estimate([ControlledScoreFold(lambda x:np.ones_like(x),lambda x:np.zeros_like(x))],[0])
        rows=[record(500,0,method,'Fixed bandwidth',bad,.8,.8,
                     lambda r:1.,1.,.15,.1) for method in METHODS]
        with TemporaryDirectory() as folder, patch('calibrated_wgan_pilot.draw'):
            summarize(pd.DataFrame(rows),Path(folder))
            summary=pd.read_csv(Path(folder)/'calibrated_wgan.csv')
            paired=pd.read_csv(Path(folder)/'calibrated_wgan_paired.csv')
        self.assertTrue((summary.failure_rate==1).all())
        self.assertTrue((summary.coverage==0).all())
        self.assertTrue(summary.conditional_coverage.isna().all())
        self.assertTrue(summary.rmse.isna().all())
        self.assertTrue((paired.paired_repetitions==0).all())


if __name__ == '__main__':
    unittest.main()
