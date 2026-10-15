"""Deterministic checks for the two-threshold orthogonal-score pilot.

These tests exercise the estimator algebra and fold boundaries, not its
finite-sample coverage. In particular, clipping probabilities for numerical
safety is not evidence that the population overlap condition holds.
"""

import contextlib
import hashlib
import io
import unittest
from pathlib import Path

import numpy as np

import audit_two_threshold_results as archive_audit

from baseline_direct_inversion import (
    SimulationConfig,
    true_reserve,
    valuation_distribution,
)
from two_threshold_pilot import (
    PROBABILITY_FLOOR,
    ThresholdFold,
    confidence_segments,
    ecdf,
    pooled_stats,
    raw_threshold_moment,
    score_covers,
    threshold_moment_and_derivatives,
    threshold_scores,
    true_threshold_target,
)


def scalar(value):
    """Accept either a scalar or a one-element NumPy result."""
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise AssertionError(f"Expected one value, received shape {array.shape}")
    return float(array.reshape(-1)[0])


class TwoThresholdTests(unittest.TestCase):
    def setUp(self):
        self.observed = np.array([0.4, 0.65, 0.85, 1.1, 1.5, 2.2])
        self.training = np.array([0.45, 0.6, 0.8, 1.0, 1.25, 1.7, 2.1])
        self.synthetic = np.array([0.25, 0.55, 0.65, 0.95, 1.3, 1.9, 2.4])
        self.delta = 0.15
        self.bidders = 5
        self.fold = ThresholdFold(
            self.observed,
            self.training,
            self.synthetic,
            self.delta,
            self.bidders,
        )

    def test_ecdf_is_raw_right_continuous_and_handles_ties(self):
        sample = np.array([2.0, 1.0, 2.0, 4.0])
        points = np.array([-1.0, 1.0, 1.5, 2.0, 4.0, 10.0])
        expected = np.array([0.0, 0.25, 0.25, 0.75, 1.0, 1.0])
        np.testing.assert_array_equal(
            np.asarray(ecdf(points, sample)).reshape(-1), expected
        )
        self.assertEqual(scalar(ecdf(0.5, sample)), 0.0)
        np.testing.assert_array_equal(sample, [2.0, 1.0, 2.0, 4.0])

    def test_empty_lower_tail_does_not_receive_a_pseudocount(self):
        sample = np.linspace(1.0, 2.0, 333)
        np.testing.assert_array_equal(
            np.asarray(ecdf(np.array([0.60, 0.75, 0.90]), sample)),
            np.zeros(3),
        )

    def test_probabilities_are_ordered_for_all_mixing_weights(self):
        grid = np.linspace(0.35, 2.4, 21)
        for weight in (0.0, 0.2, 1.0):
            with self.subTest(weight=weight):
                p_minus, p_plus = self.fold.nuisance(grid, weight)
                p_minus = np.asarray(p_minus).reshape(-1)
                p_plus = np.asarray(p_plus).reshape(-1)
                self.assertTrue(np.all((0.0 <= p_minus) & (p_minus <= 1.0)))
                self.assertTrue(np.all((0.0 <= p_plus) & (p_plus <= 1.0)))
                self.assertTrue(np.all(p_minus <= p_plus))
                self.assertTrue(np.all(np.diff(p_minus) >= 0.0))
                self.assertTrue(np.all(np.diff(p_plus) >= 0.0))

    def test_both_probabilities_use_the_same_calibration_weight(self):
        grid = np.array([0.7, 1.0, 1.4])
        for weight in (0.0, 0.3, 1.0):
            actual_minus, actual_plus = self.fold.nuisance(grid, weight)
            for offset, actual in (
                (-self.delta, actual_minus),
                (self.delta, actual_plus),
            ):
                expected = (
                    (1.0 - weight) * np.asarray(ecdf(grid + offset, self.training))
                    + weight * np.asarray(ecdf(grid + offset, self.synthetic))
                )
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_nuisance_does_not_use_heldout_observations(self):
        changed = ThresholdFold(
            self.observed + 10.0,
            self.training,
            self.synthetic,
            self.delta,
            self.bidders,
        )
        grid = np.array([0.7, 1.0, 1.4])
        np.testing.assert_array_equal(
            self.fold.nuisance(grid, 0.3), changed.nuisance(grid, 0.3)
        )
        self.assertFalse(
            np.allclose(self.fold.score_mean(grid, 0.3), changed.score_mean(grid, 0.3))
        )

    def test_empirical_nuisance_changes_when_training_data_change(self):
        changed = ThresholdFold(
            self.observed,
            self.training + 10.0,
            self.synthetic,
            self.delta,
            self.bidders,
        )
        grid = np.array([0.7, 1.0, 1.4])
        self.assertFalse(
            np.allclose(self.fold.nuisance(grid, 0.0), changed.nuisance(grid, 0.0))
        )

    def test_score_mean_matches_all_individual_heldout_scores(self):
        grid = np.array([0.7, 1.0, 1.4])
        for weight in (0.0, 0.3, 1.0):
            expected_means = []
            for theta in grid:
                p_minus, p_plus = self.fold.nuisance(theta, weight)
                expected = threshold_scores(
                    theta,
                    self.observed,
                    scalar(p_minus),
                    scalar(p_plus),
                    self.delta,
                    self.bidders,
                )
                actual = self.fold.individual_scores(theta, weight)
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
                self.assertEqual(np.asarray(actual).size, self.observed.size)
                expected_means.append(np.asarray(actual).mean())
            np.testing.assert_allclose(
                np.asarray(self.fold.score_mean(grid, weight)).reshape(-1),
                expected_means,
                rtol=1e-13,
                atol=1e-13,
            )

    def test_moment_and_derivative_signs_match_analytic_example(self):
        # q(1.2) = 1.2*(1-sqrt(.36)) = .48;
        # q(.8)  =  .8*(1-sqrt(.25)) = .40.
        moment, d_minus, d_plus = threshold_moment_and_derivatives(
            1.0, 0.25, 0.36, 0.2, 2
        )
        self.assertAlmostEqual(scalar(moment), 0.08, places=13)
        self.assertAlmostEqual(scalar(d_minus), 0.8, places=13)
        self.assertAlmostEqual(scalar(d_plus), -1.0, places=13)

    def test_raw_plugin_moment_preserves_legal_zero_cdf_values(self):
        theta, delta, bidders = 1.0, 0.2, 5
        p_minus = np.array([0.0, 0.0, 0.10, 0.25])
        p_plus = np.array([0.0, 0.02, 0.20, 0.36])
        expected = (
            (theta + delta) * (1.0 - p_plus ** (1.0 / bidders))
            - (theta - delta) * (1.0 - p_minus ** (1.0 / bidders))
        )
        actual = raw_threshold_moment(theta, p_minus, p_plus, delta, bidders)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
        self.assertAlmostEqual(float(actual[0]), 2.0 * delta, places=13)

        # Flooring guards the negative powers of CDF probabilities in the
        # orthogonal score; the plug-in objective itself is defined at zero.
        guarded = threshold_moment_and_derivatives(
            theta, p_minus, p_plus, delta, bidders
        )[0]
        self.assertGreater(abs(float(actual[0] - guarded[0])), 1e-3)
        self.assertGreater(abs(float(actual[1] - guarded[1])), 1e-3)
        np.testing.assert_allclose(actual[2:], guarded[2:], rtol=0.0, atol=1e-14)

    def test_moment_derivatives_match_finite_differences(self):
        theta, p_minus, p_plus, delta, bidders = 1.0, 0.25, 0.36, 0.2, 2
        _, d_minus, d_plus = threshold_moment_and_derivatives(
            theta, p_minus, p_plus, delta, bidders
        )
        step = 1e-6
        for component, derivative in enumerate((d_minus, d_plus)):
            plus = np.array([p_minus, p_plus])
            minus = plus.copy()
            plus[component] += step
            minus[component] -= step
            upper = threshold_moment_and_derivatives(theta, *plus, delta, bidders)[0]
            lower = threshold_moment_and_derivatives(theta, *minus, delta, bidders)[0]
            numerical = (scalar(upper) - scalar(lower)) / (2.0 * step)
            self.assertAlmostEqual(numerical, scalar(derivative), places=8)

    def test_uniform_values_give_correct_population_root_and_signs(self):
        # With F(v)=v on [0,1], q(r)=r*(1-r). Its symmetric revenue
        # difference is 2*delta*(1-2*r), for every known bidder count.
        theta = np.array([0.4, 0.5, 0.6])
        delta, bidders = 0.1, 5
        p_minus = (theta - delta) ** bidders
        p_plus = (theta + delta) ** bidders
        moment, _, _ = threshold_moment_and_derivatives(
            theta, p_minus, p_plus, delta, bidders
        )
        np.testing.assert_allclose(
            moment, 2.0 * delta * (1.0 - 2.0 * theta), rtol=0.0, atol=1e-14
        )

    def test_score_keeps_both_thresholds_on_the_same_observation(self):
        observed = np.array([0.5, 0.9, 1.1, 1.5])
        theta, delta, bidders = 1.0, 0.2, 2
        p_minus, p_plus = 0.25, 0.75
        alpha = 1.0 / bidders
        moment = (
            (theta + delta) * (1.0 - p_plus**alpha)
            - (theta - delta) * (1.0 - p_minus**alpha)
        )
        expected = (
            moment
            + alpha * (theta - delta) * p_minus ** (alpha - 1.0)
            * ((observed <= theta - delta).astype(float) - p_minus)
            - alpha * (theta + delta) * p_plus ** (alpha - 1.0)
            * ((observed <= theta + delta).astype(float) - p_plus)
        )
        actual = threshold_scores(theta, observed, p_minus, p_plus, delta, bidders)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
        self.assertAlmostEqual(np.asarray(actual).mean(), moment, places=13)
        self.assertAlmostEqual(np.asarray(actual).var(), expected.var(), places=13)

    def test_expected_score_is_orthogonal_in_both_nuisance_directions(self):
        # The four equally weighted atoms are the population, not an
        # approximation to a continuous population with Monte Carlo noise.
        population = np.array([0.5, 0.9, 1.1, 1.5])
        theta, delta, bidders = 1.0, 0.2, 2
        truth = np.array([0.25, 0.75])
        step = 1e-5
        for component in (0, 1):
            plus = truth.copy()
            minus = truth.copy()
            plus[component] += step
            minus[component] -= step
            upper = np.mean(threshold_scores(theta, population, *plus, delta, bidders))
            lower = np.mean(threshold_scores(theta, population, *minus, delta, bidders))
            self.assertLess(abs((upper - lower) / (2.0 * step)), 2e-8)

    def test_nuisance_bias_is_second_order_not_exact_double_robustness(self):
        population = np.array([0.5, 0.9, 1.1, 1.5])
        theta, delta, bidders = 1.0, 0.2, 2
        p_minus, p_plus = 0.25, 0.75
        target_moment = scalar(
            threshold_moment_and_derivatives(theta, p_minus, p_plus, delta, bidders)[0]
        )
        biases = []
        for shift in (0.001, 0.0005):
            average_score = np.mean(
                threshold_scores(
                    theta, population, p_minus + shift, p_plus, delta, bidders
                )
            )
            biases.append(abs(average_score - target_moment))
        self.assertGreater(biases[0], 1e-9)
        self.assertAlmostEqual(biases[1] / biases[0], 0.25, delta=0.003)

    def test_probability_floor_prevents_nonfinite_scores_without_pseudodata(self):
        observed = np.array([0.5, 0.9, 1.1, 1.5])
        for p_minus, p_plus in ((0.0, 0.0), (0.0, 0.02)):
            moment_and_derivatives = threshold_moment_and_derivatives(
                1.0, p_minus, p_plus, 0.2, 5
            )
            self.assertTrue(np.isfinite(np.asarray(moment_and_derivatives)).all())
            score = threshold_scores(1.0, observed, p_minus, p_plus, 0.2, 5)
            self.assertTrue(np.isfinite(score).all())

    def test_floor_is_used_consistently_in_moment_and_score_residuals(self):
        observed = np.array([0.5, 0.9, 1.1, 1.5])
        theta, delta, bidders = 1.0, 0.2, 5
        pm, pp = PROBABILITY_FLOOR, 0.02
        alpha = 1.0 / bidders
        moment = (
            (theta + delta) * (1.0 - pp**alpha)
            - (theta - delta) * (1.0 - pm**alpha)
        )
        expected = (
            moment
            + alpha * (theta - delta) * pm ** (alpha - 1.0)
            * ((observed <= theta - delta).astype(float) - pm)
            - alpha * (theta + delta) * pp ** (alpha - 1.0)
            * ((observed <= theta + delta).astype(float) - pp)
        )
        actual = threshold_scores(theta, observed, 0.0, pp, delta, bidders)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)

    def test_score_stats_preserve_nested_indicator_covariance(self):
        grid = np.array([0.7, 1.0, 1.4])
        for weight in (0.0, 0.3, 1.0):
            mean, second = self.fold.score_stats(grid, weight)
            individual = np.array(
                [self.fold.individual_scores(theta, weight) for theta in grid]
            )
            np.testing.assert_allclose(mean, individual.mean(axis=1), atol=1e-13)
            np.testing.assert_allclose(second, (individual**2).mean(axis=1), atol=1e-13)

        # Explicitly check that treating I- and I+ as independent would fail.
        theta, weight = 1.0, 0.3
        pm, pp = self.fold.nuisance(theta, weight)
        moment, dm, dp = threshold_moment_and_derivatives(
            theta, scalar(pm), scalar(pp), self.delta, self.bidders
        )
        constant = scalar(moment) - scalar(dm) * scalar(pm) - scalar(dp) * scalar(pp)
        hm = np.mean(self.observed <= theta - self.delta)
        hp = np.mean(self.observed <= theta + self.delta)
        wrong_independent_second = (
            constant**2 + scalar(dm)**2 * hm + scalar(dp)**2 * hp
            + 2.0 * constant * (scalar(dm) * hm + scalar(dp) * hp)
            + 2.0 * scalar(dm) * scalar(dp) * hm * hp
        )
        actual_second = scalar(self.fold.score_stats(theta, weight)[1])
        self.assertGreater(abs(actual_second - wrong_independent_second), 1e-3)

    def test_score_stats_match_individual_scores_when_both_cdfs_are_zero(self):
        fold = ThresholdFold(
            np.array([0.5, 0.9, 1.1, 1.5]),
            np.array([2.0, 3.0, 4.0]),
            np.array([2.5, 3.5, 4.5]),
            0.2,
            5,
        )
        np.testing.assert_array_equal(fold.nuisance(1.0, 0.3), [[0.0], [0.0]])
        individual = fold.individual_scores(1.0, 0.3)
        mean, second = fold.score_stats(1.0, 0.3)
        np.testing.assert_allclose(mean, np.mean(individual), rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(second, np.mean(individual**2), rtol=1e-13, atol=1e-12)

    def test_pooled_stats_use_observation_weights_and_sample_score_variance(self):
        second_fold = ThresholdFold(
            np.array([0.38, 1.05, 1.65]),
            self.training + 0.05,
            self.synthetic + 0.05,
            self.delta,
            self.bidders,
        )
        folds, weights = [self.fold, second_fold], [0.2, 0.6]
        grid = np.array([0.7, 1.0, 1.4])
        mean, se = pooled_stats(folds, weights, grid)
        expected_mean, expected_se = [], []
        for theta in grid:
            individual = np.concatenate(
                [fold.individual_scores(theta, weight) for fold, weight in zip(folds, weights)]
            )
            expected_mean.append(individual.mean())
            expected_se.append(individual.std(ddof=1) / np.sqrt(individual.size))
        np.testing.assert_allclose(mean, expected_mean, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(se, expected_se, rtol=1e-13, atol=1e-13)

    def test_confidence_segments_keep_disconnected_and_boundary_components(self):
        grid = np.arange(6.0)
        accepted = np.array([True, False, True, True, False, True])
        np.testing.assert_allclose(
            confidence_segments(grid, accepted),
            [[0.0, 0.5], [1.5, 3.5], [4.5, 5.0]],
        )
        self.assertEqual(confidence_segments(grid, np.zeros(6, dtype=bool)), [])
        np.testing.assert_allclose(
            confidence_segments(grid, np.ones(6, dtype=bool)), [[0.0, 5.0]]
        )

    def test_target_coverage_is_a_score_test_at_the_supplied_target(self):
        weight = 0.3
        for target in (0.7, 1.0, 1.4):
            individual = self.fold.individual_scores(target, weight)
            expected = bool(
                abs(individual.mean())
                <= 1.96 * individual.std(ddof=1) / np.sqrt(individual.size)
            )
            self.assertEqual(score_covers([self.fold], [weight], target), expected)

    def test_population_target_solves_revenue_difference_not_exact_foc(self):
        baseline = SimulationConfig(bidders=5)
        distribution = valuation_distribution(baseline)
        delta = 0.15
        target = true_threshold_target(delta, baseline)

        def revenue_difference(theta):
            return (
                (theta + delta) * distribution.sf(theta + delta)
                - (theta - delta) * distribution.sf(theta - delta)
            )

        self.assertLess(abs(revenue_difference(target)), 1e-11)
        self.assertGreater(revenue_difference(target - 0.001), 0.0)
        self.assertLess(revenue_difference(target + 0.001), 0.0)
        self.assertGreater(abs(target - true_reserve(baseline)), 1e-4)
        # Bidder count changes observed H, not the equivalent posted-price
        # revenue objective under the maintained symmetric IPV model.
        self.assertAlmostEqual(
            target, true_threshold_target(delta, SimulationConfig(bidders=2)), places=12
        )

    def test_population_target_bias_has_the_expected_quadratic_delta_order(self):
        baseline = SimulationConfig(bidders=5)
        exact = true_reserve(baseline)
        bias = abs(true_threshold_target(0.04, baseline) - exact)
        half_width_bias = abs(true_threshold_target(0.02, baseline) - exact)
        self.assertGreater(bias, 1e-6)
        self.assertAlmostEqual(half_width_bias / bias, 0.25, delta=0.01)

    def test_price_unit_changes_scale_scores_but_not_probabilities(self):
        scale = 3.0
        scaled = ThresholdFold(
            scale * self.observed,
            scale * self.training,
            scale * self.synthetic,
            scale * self.delta,
            self.bidders,
        )
        # Stay away from atoms so that binary floating-point tie handling
        # does not confound the change-of-units check.
        theta, weight = 1.13, 0.3
        np.testing.assert_allclose(
            scaled.nuisance(scale * theta, weight), self.fold.nuisance(theta, weight)
        )
        np.testing.assert_allclose(
            scaled.individual_scores(scale * theta, weight),
            scale * self.fold.individual_scores(theta, weight),
            rtol=1e-13,
            atol=1e-13,
        )


class TwoThresholdArchiveTests(unittest.TestCase):
    def test_audit_defaults_to_committed_archive_without_writes(self):
        args = archive_audit.parse_args([])
        self.assertEqual(args.results_root, archive_audit.DEFAULT_RESULTS)
        self.assertEqual(args.source_snapshot, archive_audit.DEFAULT_SOURCE)
        self.assertIsNone(args.write_comparison)
        legacy = archive_audit.parse_args(["--results-root", "output/tables"])
        self.assertEqual(legacy.results_root, Path("output/tables"))
        self.assertIsNone(legacy.write_comparison)

    def test_audit_rejects_a_source_hash_not_recorded_by_the_experiment(self):
        with self.assertRaisesRegex(AssertionError, "source hash mismatch"):
            archive_audit.audit_result_folder(
                archive_audit.DEFAULT_RESULTS / "two_threshold_pilot", "incorrect"
            )

    def test_full_archived_audit_does_not_modify_its_inputs(self):
        paths = sorted(
            path for path in archive_audit.DEFAULT_RESULTS.rglob("*") if path.is_file()
        ) + [archive_audit.DEFAULT_SOURCE]

        def snapshot():
            return {
                path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in paths
            }

        before = snapshot()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            archive_audit.main([])
        self.assertIn("Audit passed", output.getvalue())
        self.assertEqual(before, snapshot())


if __name__ == "__main__":
    unittest.main()
