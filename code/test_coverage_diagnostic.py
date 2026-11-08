"""Small deterministic tests for coverage diagnostics; no GAN training.

Run from the project root with:
    /opt/anaconda3/bin/python -B code/test_coverage_diagnostic.py
"""
from __future__ import annotations

import unittest

import numpy as np

import diagnose_fixed_h_coverage as diagnostic


class InvertStatisticTests(unittest.TestCase):
    def test_linear_score_has_analytic_acceptance_interval(self):
        root, scale = 0.8, 0.06

        def statistic(points):
            return (np.asarray(points, dtype=float) - root) / scale

        actual = diagnostic.invert_statistic(statistic, root)
        expected = [[root - diagnostic.CRITICAL * scale,
                     root + diagnostic.CRITICAL * scale]]
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-11)

    def test_two_disconnected_components_are_not_replaced_by_hull(self):
        center, distance, coefficient = 0.8, 0.15, 100.0

        def statistic(points):
            x = np.asarray(points, dtype=float)
            return coefficient * ((x - center)**2 - distance**2)

        outer = np.sqrt(distance**2 + diagnostic.CRITICAL / coefficient)
        inner = np.sqrt(distance**2 - diagnostic.CRITICAL / coefficient)
        expected = [[center - outer, center - inner],
                    [center + inner, center + outer]]
        actual = diagnostic.invert_statistic(statistic, center-distance)
        self.assertEqual(len(actual), 2)
        self.assertTrue(actual[0][1] < center < actual[1][0])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-11)
        finer = diagnostic.invert_statistic(statistic, center-distance, grid_size=561)
        np.testing.assert_allclose(finer, expected, rtol=0.0, atol=2e-11)

    def test_acceptance_interval_is_clipped_at_either_search_boundary(self):
        scale = 0.04
        for root in (0.46, 1.14):
            with self.subTest(root=root):
                def statistic(points):
                    return (np.asarray(points, dtype=float) - root) / scale

                actual = diagnostic.invert_statistic(statistic, root)
                expected = [[max(diagnostic.BOUNDS[0], root-diagnostic.CRITICAL*scale),
                             min(diagnostic.BOUNDS[1], root+diagnostic.CRITICAL*scale)]]
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-11)


class ScoreStatisticsTests(unittest.TestCase):
    @staticmethod
    def folds():
        observed_groups = [np.array([0.75, 1.05]),
                           np.array([0.8, 1.2, 1.5]),
                           np.array([0.9, 1.1, 1.4, 1.8])]
        synthetic = np.array([0.65, 0.85, 1.0, 1.2, 1.4, 1.7])
        return [diagnostic.CachedMoments(
            group, np.concatenate(observed_groups[:index] + observed_groups[index+1:]),
            synthetic, 0.3, 5) for index, group in enumerate(observed_groups)]

    def test_pooled_variance_and_t_equal_direct_observation_level_scores(self):
        moments = self.folds()
        weights = [0.0, 0.25, 0.7]
        points = np.array([0.65, 0.85, 1.15])
        direct = np.stack([
            np.concatenate([fold.individual_scores(point, weight)
                            for fold, weight in zip(moments, weights)])
            for point in points])
        expected_mean = direct.mean(axis=1)
        expected_variance = direct.var(axis=1, ddof=1)
        expected_t = np.sqrt(direct.shape[1]) * expected_mean / np.sqrt(expected_variance)
        actual_mean, actual_variance, actual_t = diagnostic.score_statistics(points, moments, weights)
        np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(actual_variance, expected_variance, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(actual_t, expected_t, rtol=1e-12, atol=1e-14)

    def test_plugin_score_mean_equals_full_sample_moment(self):
        observed = np.array([0.75, 0.8, 0.9, 1.05, 1.1, 1.2, 1.4, 1.5, 1.8])
        full = diagnostic.CachedMoments(observed, observed, np.empty(0), 0.3, 5)
        points = np.array([0.65, 0.85, 1.15])
        mean, variance, _ = diagnostic.score_statistics(points, [full], [0.0])
        cdf, density = full.nuisance(points, 0.0)
        moment, _, _ = diagnostic.reserve_moment_and_derivatives(points, cdf, density, 5)
        np.testing.assert_allclose(mean, moment, rtol=1e-12, atol=1e-14)
        self.assertTrue(np.all(variance > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
