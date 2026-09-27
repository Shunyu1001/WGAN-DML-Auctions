"""Deterministic invariants for the monotone/adaptive experiment."""
import unittest
import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import norm

from monotone_adaptive_pilot import (
    MonotoneLogQuantile, inner_split, initialize_monotone, lognormal_mle,
    lognormal_reserve, monotone_maxima, validation_weight, distribution_for,
    targets_and_revenue,
)
from orthogonal_wgan_dml import gaussian_kernel_moments


class MonotoneAdaptiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_positive_slopes_and_unbounded_log_tails(self):
        model=MonotoneLogQuantile(.1,.5)
        with torch.no_grad():
            model.raw_slopes.copy_(torch.linspace(-3,1,10))
            z=torch.linspace(-8,8,1001)
            values=model.log_from_z(z).numpy()
        self.assertTrue(np.all(np.diff(values)>0))
        self.assertLess(values[0],float(model.log_from_z(torch.tensor(-3.)).detach()))
        self.assertGreater(values[-1],float(model.log_from_z(torch.tensor(3.)).detach()))

    def test_quantile_spline_starts_linear_and_has_gradients(self):
        model=MonotoneLogQuantile(.1,.5)
        z=torch.tensor([-5.,-.75,.25,5.],requires_grad=True)
        values=model.log_from_z(z)
        np.testing.assert_allclose(values.detach().numpy(),.1+.50001*z.detach().numpy(),atol=1e-6)
        values.sum().backward()
        self.assertTrue(torch.all(z.grad>0))
        self.assertTrue(torch.isfinite(model.raw_slopes.grad).all())

    def test_maximum_map_and_quadrature_convergence(self):
        model=MonotoneLogQuantile(0.,.5)
        count=32768
        with torch.no_grad():
            first=monotone_maxima(model,count,5,True).exp().numpy().ravel()
            second=monotone_maxima(model,count*2,5,True).exp().numpy().ravel()
        u=(np.arange(count)+.5)/count
        expected=np.exp(.50001*ndtri(u**.2))
        np.testing.assert_allclose(first,expected,rtol=2e-6)
        a,g=gaussian_kernel_moments(np.array([.65,.8,1.]),first,.1)
        aa,gg=gaussian_kernel_moments(np.array([.65,.8,1.]),second,.1)
        np.testing.assert_allclose(a,aa,atol=2e-5)
        np.testing.assert_allclose(g,gg,atol=5e-5)

    def test_inner_split_is_disjoint_complete_reproducible(self):
        data=np.arange(101.)
        fitting,validation=inner_split(data,85,.25)
        self.assertEqual(len(validation),25)
        self.assertEqual(len(np.intersect1d(fitting,validation)),0)
        np.testing.assert_array_equal(np.sort(np.r_[fitting,validation]),data)
        np.testing.assert_array_equal(inner_split(data,85,.25)[0],fitting)

    def test_validation_can_choose_both_endpoints(self):
        data=np.exp(np.linspace(-.3,1,120))
        synthetic=data*1.4
        candidates=(0.,.25,.5,1.)
        weight,risks=validation_weight(data,data,synthetic,.15,5,candidates)
        self.assertEqual(weight,0.)
        self.assertEqual(risks[0],min(risks))
        weight,risks=validation_weight(data,synthetic,synthetic,.15,5,candidates)
        self.assertEqual(weight,1.)
        self.assertEqual(risks[-1],min(risks))

    def test_initialization_depends_on_training_data(self):
        data=np.exp(.2+.6*ndtri(((np.arange(2000)+.5)/2000)**.2))
        model=initialize_monotone(data,5)
        self.assertAlmostEqual(float(model.anchor.detach()),.2,places=2)
        self.assertEqual(sum(p.numel() for p in model.parameters()),11)

    def test_order_statistic_mle_recovers_deterministic_sample(self):
        u=(np.arange(5000)+.5)/5000
        data=np.exp(.5*ndtri(u**.2))
        result=lognormal_mle(data,5)
        self.assertTrue(result['optimizer_success'])
        self.assertAlmostEqual(result['estimate'],lognormal_reserve([0.,np.log(.5)]),places=3)
        self.assertGreater(result['standard_error'],0)

    def test_population_targets_and_negative_reserve_revenue(self):
        for name in ['lognormal','weibull']:
            d=distribution_for(name)
            r0,rh,revenue=targets_and_revenue(d,5,.15)
            self.assertLess(abs(d.sf(r0)-r0*d.pdf(r0)),1e-9)
            self.assertGreater(rh,0)
            self.assertEqual(revenue(-.2),revenue(0.))
            self.assertGreaterEqual(revenue(r0),revenue(r0*.8))


if __name__=='__main__':
    unittest.main()
