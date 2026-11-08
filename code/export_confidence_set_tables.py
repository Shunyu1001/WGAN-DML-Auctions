"""Append reporting fields to completed validation results without overwriting them."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import pandas as pd
from confidence_set_reporting import describe_confidence_set, REPORTING_RULE_VERSION


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def export(source,output):
    if output.exists(): raise FileExistsError(f"Refusing to replace {output}")
    files=[source/"summary/raw_results.csv",source/"summary/summary.csv",
           source/"summary/verification.json",source/"protocol.json",source/"run_manifest.json"]
    before={str(p.relative_to(source)):digest(p) for p in files}
    frame=pd.read_csv(files[0],keep_default_na=False,na_values=[""])
    if frame.duplicated(["auctions","repetition","method"]).any():
        raise ValueError("Source contains duplicate result keys")
    output.mkdir(parents=True)
    rows=[]; components=[]
    for original in frame.to_dict("records"):
        row=dict(original)
        parts=json.loads(row["score_components"])
        report=describe_confidence_set(parts,row["inversion_status"])
        if row["inversion_status"]=="ok":
            if report["set_component_count"]!=row["component_count"]:
                raise AssertionError("Original component count mismatch")
            if abs(report["set_total_length"]-row["score_length"])>1e-12:
                raise AssertionError("Original set length mismatch")
            if report["set_touches_search_boundary"]!=row["boundary_truncated"]:
                raise AssertionError("Original boundary flag mismatch")
            hull_cover=report["conservative_hull_lower"]<=row["target"]<=report["conservative_hull_upper"]
            if row["score_coverage"] and not hull_cover: raise AssertionError("Hull lost coverage")
        else: hull_cover=False
        row.update(report,conservative_hull_coverage=bool(hull_cover))
        rows.append(row)
        for number,(left,right) in enumerate(parts,1):
            components.append(dict(auctions=row["auctions"],repetition=row["repetition"],
                method=row["method"],component=number,lower=left,upper=right,length=right-left,
                touches_lower_bound=abs(left-.45)<=1e-9,touches_upper_bound=abs(right-1.15)<=1e-9))
    result=pd.DataFrame(rows)
    result.to_csv(output/"results_with_reporting.csv",index=False)
    pd.DataFrame(components).to_csv(output/"set_components.csv",index=False)
    flags=(result.set_is_disconnected.fillna(False) | result.set_touches_search_boundary.fillna(False))
    flagged=result[flags]
    flagged.to_csv(output/"flagged_cases.csv",index=False)
    summaries=[]
    for (n,method),g in result.groupby(["auctions","method"],sort=True):
        for label,valid,coverage,length in (
            ("Wald",g.status.eq("ok"),g.wald_coverage,g.wald_length),
            ("Full score set",g.inversion_status.eq("ok"),g.score_coverage,g.set_total_length),
            ("Conservative envelope",g.set_report_status.eq("ok"),g.conservative_hull_coverage,g.conservative_hull_length)):
            summaries.append(dict(auctions=n,method=method,reported_object=label,repetitions=len(g),
                valid_count=int(valid.sum()),failure_count=int((~valid).sum()),failure_rate=float((~valid).mean()),
                coverage=float((coverage & valid).mean()),mean_length=float(length[valid].mean()),
                disconnected_count=int(g.set_is_disconnected.fillna(False).sum()) if label=="Full score set" else 0,
                source_disconnected_count=int(g.set_is_disconnected.fillna(False).sum()) if label!="Wald" else 0,
                lower_boundary_count=int(g.set_touches_lower_bound.fillna(False).sum()) if label!="Wald" else 0,
                upper_boundary_count=int(g.set_touches_upper_bound.fillna(False).sum()) if label!="Wald" else 0))
    summary=pd.DataFrame(summaries);summary.to_csv(output/"summary.csv",index=False)
    # Lightweight tables for later manuscript use; deliberately outside paper/.
    labels={"Calibrated WGAN-DML":"Calibrated GAN", "Empirical-local DML":"Empirical DML", "Smooth plug-in":"Smooth plug-in"}
    lines=[r"\begin{tabular}{rlrrrrr}",r"\toprule",
           r"$n$ & Method & Coverage & Set length & Hull length & Disconnected & Boundary \\",r"\midrule"]
    for (n,method),g in result.groupby(["auctions","method"],sort=True):
        lines.append(f"{n} & {labels[method]} & {g.score_coverage.mean():.2f} & {g.set_total_length.mean():.5f} & {g.conservative_hull_length.mean():.5f} & {int(g.set_is_disconnected.sum())} & {int(g.set_touches_search_boundary.sum())} " + chr(92)*2)
    lines += [r"\bottomrule",r"\end{tabular}","% Set length sums all components. Hull is conservative and does not replace the set."]
    (output/"confidence_set_table.tex").write_text("\n".join(lines)+"\n")
    checks=dict(rows_preserved=len(result)==len(frame),
        original_components_preserved=result.score_components.equals(frame.score_components),
        original_estimates_preserved=result.estimate.equals(frame.estimate),
        original_standard_errors_preserved=result.standard_error.equals(frame.standard_error),
        original_score_coverage_preserved=result.score_coverage.equals(frame.score_coverage),
        original_wald_coverage_preserved=result.wald_coverage.equals(frame.wald_coverage),
        component_long_table_complete=len(components)==int(result.set_component_count.sum()),
        hull_covers_every_original_coverage=bool((~result.score_coverage | result.conservative_hull_coverage).all()),
        hull_length_not_smaller=bool(((result.conservative_hull_length+1e-12>=result.set_total_length) | result.set_report_status.eq("failed")).all()),
        source_files_unchanged=all(digest(source/k)==v for k,v in before.items()))
    checks["flagged_rows_match_geometry"]=all(bool(r.set_is_disconnected) or
        bool(r.set_touches_search_boundary) for r in flagged.itertuples())
    (output/"verification.json").write_text(json.dumps(dict(passed=all(checks.values()),checks=checks,
        rows=len(result),components=len(components),flagged_rows=len(flagged)),indent=2)+"\n")
    (output/"manifest.json").write_text(json.dumps(dict(source=str(source),source_sha256=before,
        reporting_rule=REPORTING_RULE_VERSION,bounds=[.45,1.15],boundary_tolerance=1e-9,
        point_estimator_changed=False,confidence_set_changed=False,manuscript_changed=False),indent=2)+"\n")
    if not all(checks.values()):raise AssertionError("Reporting verification failed")
    # Typed data used only to author the optional reader-facing workbook.
    serializable=lambda f: f.astype(object).where(pd.notna(f),None).to_dict("records")
    (output/"workbook_data.json").write_text(json.dumps(dict(raw=serializable(result),
        summary=serializable(summary),flagged=serializable(flagged)),allow_nan=False)+"\n")
    print(summary.to_string(index=False));print("Saved",output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();export(a.source.resolve(),a.output_dir.resolve())
