"""Checks for the observed-price, tie-accepting direct-inversion benchmark."""
import unittest

import numpy as np

from baseline_direct_inversion import estimate_reserve_from_maxima


class BaselineReserveTests(unittest.TestCase):
    def test_uses_left_limit_at_observed_price(self):
        # Right-continuous evaluation incorrectly selects 1 for these data.
        self.assertEqual(estimate_reserve_from_maxima(np.array([1., 2., 10.]), 2), 10.)

    def test_tied_prices_share_no_sale_probability(self):
        self.assertEqual(estimate_reserve_from_maxima(np.array([1., 1., 10., 10.]), 2), 10.)

    def test_single_and_constant_samples(self):
        self.assertEqual(estimate_reserve_from_maxima(np.array([3.]), 5), 3.)
        self.assertEqual(estimate_reserve_from_maxima(np.repeat(3., 10), 5), 3.)

    def test_permutation_invariance(self):
        x = np.array([1., 2., 10., 3., 10.])
        self.assertEqual(estimate_reserve_from_maxima(x, 3),
                         estimate_reserve_from_maxima(x[::-1], 3))

    def test_invalid_inputs(self):
        for x in [np.array([]), np.array([[1., 2.]]), np.array([np.nan]),
                  np.array([0.]), np.array([-1.])]:
            with self.assertRaises(ValueError):
                estimate_reserve_from_maxima(x, 2)
        for n in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                estimate_reserve_from_maxima(np.array([1.]), n)


if __name__ == "__main__":
    unittest.main()
