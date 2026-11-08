"""Validate the separately saved paired coverage diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir",type=Path,required=True)
args=parser.parse_args(); p=args.output_dir
source=p.parent/"fixed_h_100_2026-11-08"
old=pd.read_csv(source/"summary/raw_results.csv").set_index(["auctions","repetition","method"])
new=pd.read_csv(p/"raw_results.csv").set_index(["auctions","repetition","method"])
checks={}
checks["600_unique_complete_rows"]=(len(new)==600 and new.index.is_unique and set(new.index)==set(old.index))
checks["100_per_method_and_size"]=(new.groupby(level=[0,2]).size().eq(100).all())
aligned=old.loc[new.index]
checks["point_estimates_and_se_unchanged"]=bool(np.allclose(new.estimate,aligned.estimate,atol=1e-12,rtol=0)
    and np.allclose(new.standard_error,aligned.standard_error,atol=1e-12,rtol=0))
checks["original_wald_coverage_preserved"]=bool((new.wald_coverage==aligned.covers_target).all())
checks["all_inversions_valid"]=bool(new.inversion_status.eq("ok").all())
checks["test_coverage_exactly_matches_saved_flag"]=bool((new.score_coverage==(abs(new.score_t_at_target)<=1.96)).all())
membership=[]; root_member=[]; lengths=[]; count=[]
for _,r in new.iterrows():
    c=json.loads(r.score_components)
    membership.append(any(a-1e-9<=r.target<=b+1e-9 for a,b in c))
    root_member.append(any(a-1e-9<=r.estimate<=b+1e-9 for a,b in c))
    lengths.append(sum(b-a for a,b in c)); count.append(len(c))
checks["set_membership_matches_test_coverage"]=bool(np.array_equal(membership,new.score_coverage))
checks["all_sets_contain_original_estimate"]=all(root_member)
checks["set_lengths_and_component_counts_match"]=bool(np.allclose(lengths,new.score_length,atol=1e-12,rtol=0)
    and np.array_equal(count,new.component_count))
summary=pd.read_csv(p/"comparison_summary.csv")
metrics=True
for _,r in summary.iterrows():
    g=new.xs((r.auctions,r.method),level=(0,2))
    cov=g.wald_coverage if r.interval=="Wald" else g.score_coverage
    length=g.wald_length if r.interval=="Wald" else g.score_length
    metrics=metrics and abs(cov.mean()-r.coverage)<1e-12 and abs(length.mean()-r.mean_length)<1e-12
    metrics=metrics and abs(g.error.mean()-r.bias)<1e-12 and abs(np.sqrt(np.mean(g.error**2))-r.rmse)<1e-12
checks["summary_metrics_recomputed"]=bool(metrics and len(summary)==12)
refine=pd.to_numeric(new.refinement_max_difference,errors="coerce").dropna()
checks["18_selected_grid_refinements_agree"]=bool(len(refine)==18 and refine.max()<1e-8)
preservation=json.loads((p/"source_integrity_before.json").read_text())
checks["original_source_and_results_unchanged"]=all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v
    for k,v in preservation.items())
cache_files={}
for n in (500,2000):
    cache_files.update(json.loads((p/f"n{n}/cache_sha256.json").read_text()))
checks["600_used_cache_files_unchanged"]=len(cache_files)==600 and all(
    hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in cache_files.items())
report=dict(passed=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},
    rows=len(new),cached_folds_checked=len(cache_files),grid_refinement_rows=len(refine),
    maximum_refinement_difference=float(refine.max()),
    scope="paired exploratory correction on original 100 repetitions per sample size; not fresh-seed validation")
with (p/"verification.json").open("x") as f: json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
if not report["passed"]: raise SystemExit(1)
