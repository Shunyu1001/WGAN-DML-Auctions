"""Numerical checks for the calibration, score, and interval implementation."""
import unittest
import numpy as np

from calibrated_wgan_pilot import FoldMoments, calibration_weight, estimate, scores, wilson
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


if __name__ == '__main__':
    unittest.main()
