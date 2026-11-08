"""Pure reporting of an unchanged confidence set and its conservative hull.

No point estimate, score, critical value or candidate set is changed here.
The helper has no access to a simulation truth.
"""
from __future__ import annotations
import math

REPORTING_RULE_VERSION = "full_set_with_conservative_hull_v1"


def describe_confidence_set(components, status="ok", bounds=(0.45,1.15), tolerance=1e-9):
    lower, upper = map(float,bounds)
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper
            and math.isfinite(tolerance) and tolerance >= 0):
        raise ValueError("Invalid bounds or boundary tolerance")
    if status not in ("ok","failed"):
        raise ValueError("Unknown confidence-set status")
    if status == "failed":
        if components:
            raise ValueError("Failed set must not contain reported components")
        return dict(set_report_status="failed",set_component_count=0,
            set_is_disconnected=None,set_touches_lower_bound=None,set_touches_upper_bound=None,
            set_touches_search_boundary=None,set_total_length=None,
            conservative_hull_lower=None,conservative_hull_upper=None,
            conservative_hull_length=None,conservative_hull_gap_length=None,
            reporting_rule_version=REPORTING_RULE_VERSION)
    if not components:
        raise ValueError("A successful set requires at least one component")
    intervals=[]; previous_right=None
    for component in components:
        if len(component)!=2:
            raise ValueError("Each component must have exactly two endpoints")
        left,right=map(float,component)
        if not (math.isfinite(left) and math.isfinite(right) and left<=right):
            raise ValueError("Nonfinite or reversed component")
        if left<lower-tolerance or right>upper+tolerance:
            raise ValueError("Component lies outside the search bounds")
        if previous_right is not None and left<=previous_right:
            raise ValueError("Components must be strictly ordered and disjoint")
        intervals.append((left,right));previous_right=right
    touches_lower=any(abs(left-lower)<=tolerance for left,right in intervals)
    touches_upper=any(abs(right-upper)<=tolerance for left,right in intervals)
    total=math.fsum(right-left for left,right in intervals)
    hull_lower=intervals[0][0];hull_upper=intervals[-1][1]
    hull_length=hull_upper-hull_lower
    return dict(set_report_status="ok",set_component_count=len(intervals),
        set_is_disconnected=len(intervals)>1,set_touches_lower_bound=touches_lower,
        set_touches_upper_bound=touches_upper,set_touches_search_boundary=touches_lower or touches_upper,
        set_total_length=total,conservative_hull_lower=hull_lower,
        conservative_hull_upper=hull_upper,conservative_hull_length=hull_length,
        conservative_hull_gap_length=math.fsum(intervals[j][0]-intervals[j-1][1]
                                              for j in range(1,len(intervals))),
        reporting_rule_version=REPORTING_RULE_VERSION)


def contains_target(components,target):
    """Simulation evaluation only; never used to construct the reported set."""
    return any(float(a)<=target<=float(b) for a,b in components)
