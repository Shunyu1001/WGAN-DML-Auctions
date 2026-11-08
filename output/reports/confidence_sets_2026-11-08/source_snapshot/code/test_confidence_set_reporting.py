"""Small deterministic tests for additive confidence-set reporting."""
from __future__ import annotations

import copy
import json
import math
import unittest

from confidence_set_reporting import (REPORTING_RULE_VERSION, contains_target,
                                      describe_confidence_set)
import verify_fresh_score_validation as verifier


class ConfidenceSetReportingTests(unittest.TestCase):
    def test_single_internal_component_is_its_own_hull(self):
        result=describe_confidence_set([[.7,.9]])
        self.assertEqual(result["set_report_status"],"ok")
        self.assertEqual(result["set_component_count"],1)
        self.assertFalse(result["set_is_disconnected"])
        self.assertFalse(result["set_touches_search_boundary"])
        self.assertAlmostEqual(result["set_total_length"],.2)
        self.assertAlmostEqual(result["conservative_hull_length"],.2)
        self.assertEqual(result["conservative_hull_gap_length"],0)

    def test_multiple_components_keep_gap_and_all_members(self):
        components=[[.45,.58],[.78,.96]]
        result=describe_confidence_set(components)
        self.assertTrue(result["set_is_disconnected"])
        self.assertEqual(result["set_component_count"],2)
        self.assertTrue(result["set_touches_lower_bound"])
        self.assertFalse(result["set_touches_upper_bound"])
        self.assertAlmostEqual(result["set_total_length"],.31)
        self.assertAlmostEqual(result["conservative_hull_length"],.51)
        self.assertAlmostEqual(result["conservative_hull_gap_length"],.2)
        for point in (.45,.50,.58,.78,.85,.96):
            self.assertTrue(contains_target(components,point))
            self.assertLessEqual(result["conservative_hull_lower"],point)
            self.assertGreaterEqual(result["conservative_hull_upper"],point)
        self.assertFalse(contains_target(components,.7))
        self.assertLess(result["conservative_hull_lower"],.7)
        self.assertGreater(result["conservative_hull_upper"],.7)

    def test_each_boundary_and_both_boundaries(self):
        for components,lower,upper in [([[.45,.6]],True,False),([[.8,1.15]],False,True),
                                       ([[.45,1.15]],True,True)]:
            with self.subTest(components=components):
                result=describe_confidence_set(components)
                self.assertEqual(result["set_touches_lower_bound"],lower)
                self.assertEqual(result["set_touches_upper_bound"],upper)
                self.assertEqual(result["set_touches_search_boundary"],lower or upper)

    def test_boundary_tolerance_does_not_clip_or_change_endpoints(self):
        left=.45-5e-10
        result=describe_confidence_set([[left,.7]])
        self.assertTrue(result["set_touches_lower_bound"])
        self.assertEqual(result["conservative_hull_lower"],left)

    def test_singleton_is_valid_zero_length_set(self):
        result=describe_confidence_set([[.8,.8]])
        self.assertEqual(result["set_total_length"],0)
        self.assertEqual(result["conservative_hull_length"],0)
        self.assertFalse(result["set_is_disconnected"])

    def test_failed_report_is_missing_not_false_or_zero_length(self):
        result=describe_confidence_set([],status="failed")
        self.assertEqual(result["set_report_status"],"failed")
        self.assertEqual(result["set_component_count"],0)
        for name in ["set_is_disconnected","set_touches_lower_bound","set_touches_upper_bound",
                     "set_touches_search_boundary","set_total_length","conservative_hull_lower",
                     "conservative_hull_upper","conservative_hull_length","conservative_hull_gap_length"]:
            self.assertIsNone(result[name],name)
        self.assertEqual(result["reporting_rule_version"],REPORTING_RULE_VERSION)
        json.dumps(result,allow_nan=False)

    def test_never_mutates_input(self):
        original=[[.45,.58],[.78,.96]]
        before=copy.deepcopy(original)
        describe_confidence_set(original)
        self.assertEqual(original,before)

    def test_bad_components_are_rejected_without_repair(self):
        cases=[[],[[.9,.7]],[[float("nan"),.8]],[[.7,float("inf")]],
               [[.6]],[[.6,.7,.8]],[[.4,.7]],[[.7,1.2]],
               [[.8,.9],[.6,.7]],[[.6,.8],[.7,.9]],[[.6,.8],[.8,.9]]]
        for components in cases:
            with self.subTest(components=components):
                with self.assertRaises(ValueError): describe_confidence_set(components)

    def test_invalid_status_bounds_tolerance_and_failed_components(self):
        for kwargs in ({"status":"unknown"},{"bounds":(.9,.5)},
                       {"bounds":(math.nan,1.)},{"tolerance":-1.},
                       {"tolerance":math.inf},{"status":"failed"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError): describe_confidence_set([[.7,.8]],**kwargs)


class ReportingVerifierCompatibilityTests(unittest.TestCase):
    @staticmethod
    def row(rep=2,point_ok=True,inversion_ok=True,reported=True):
        components=[[.45,.55],[.79,.9]] if inversion_ok else []
        target=.83;estimate=.84;se=.03
        result=dict(auctions=500,repetition=rep,method="Calibrated WGAN-DML",
            status="ok" if point_ok else "failed",failure_reason="" if point_ok else "no_root",
            estimate=estimate if point_ok else None,standard_error=se if point_ok else None,
            target=target,error=estimate-target if point_ok else None,jacobian=-1. if point_ok else None,
            wald_coverage=point_ok,wald_length=3.92*se if point_ok else None,
            inversion_status="ok" if inversion_ok else "failed",
            inversion_failure_reason="" if inversion_ok else "invalid_candidate_variance",
            score_components=json.dumps(components),component_count=len(components),
            score_length=.21 if inversion_ok else None,score_coverage=inversion_ok,
            boundary_truncated=inversion_ok,score_t_at_target=0. if inversion_ok else None,
            refinement_max_difference=None)
        if reported:
            report=describe_confidence_set(components,"ok" if inversion_ok else "failed")
            result.update(report)
            result["conservative_hull_coverage"]=inversion_ok
        return result

    def test_legacy_two_object_rows_remain_supported(self):
        rows=[self.row(reported=False)]
        audit=verifier.Audit()
        verifier.validate_rows(rows,audit,.83)
        self.assertTrue(all(audit.checks.values()),audit.failures)
        summary,_=verifier.independent_statistics(rows)
        self.assertEqual({k[2] for k in summary},{"Wald","Score inversion"})

    def test_reporting_with_two_kinds_of_failures_has_three_objects(self):
        rows=[self.row(),self.row(3,False,False),self.row(4,True,False)]
        audit=verifier.Audit()
        verifier.validate_rows(rows,audit,.83)
        self.assertTrue(all(audit.checks.values()),audit.failures)
        summary,_=verifier.independent_statistics(rows)
        hull=summary[500,"Calibrated WGAN-DML","Conservative envelope"]
        self.assertEqual(hull["interval_failures"],2)
        self.assertEqual(hull["point_failures"],1)
        self.assertEqual(hull["additional_interval_failures"],1)
        self.assertEqual(hull["source_disconnected_sets"],1)
        self.assertEqual(hull["disconnected_sets"],0)
        self.assertEqual(hull["touches_lower_bound"],1)
        self.assertAlmostEqual(hull["coverage"],1/3)
        self.assertAlmostEqual(hull["mean_length"],.45)

    def test_missing_or_false_failure_flag_is_detected(self):
        result=self.row(point_ok=False,inversion_ok=False)
        self.assertTrue(verifier.reporting_row_valid(result,[],False))
        result["set_is_disconnected"]=False
        self.assertFalse(verifier.reporting_row_valid(result,[],False))
        del result["conservative_hull_lower"]
        self.assertFalse(verifier.reporting_row_valid(result,[],False))

    def test_incorrect_hull_or_topology_flag_is_detected(self):
        result=self.row()
        components=json.loads(result["score_components"])
        self.assertTrue(verifier.reporting_row_valid(result,components,True))
        result["set_is_disconnected"]=False
        self.assertFalse(verifier.reporting_row_valid(result,components,True))


if __name__=="__main__": unittest.main()
