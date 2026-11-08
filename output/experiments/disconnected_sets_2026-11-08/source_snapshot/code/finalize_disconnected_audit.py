"""High-precision variance aggregation and read-only input verification."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from diagnose_disconnected_sets import load_case, SOURCE

p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,required=True)
out=p.parse_args().output_dir
probes=pd.read_csv(out/"probe_statistics.csv")
precise=[]
for rep in (29,50):
    observed,tasks,_=load_case(rep)
    for method,moments,weights,ids,plugin in tasks:
        rows=probes[(probes.repetition==rep)&(probes.method==method)&probes.status.eq("ok")]
        for _,r in rows.iterrows():
            values=np.concatenate([f.individual_scores(r.theta,w) for f,w in zip(moments,weights)])
            with localcontext() as ctx:
                ctx.prec=60
                values60=[Decimal.from_float(float(x)) for x in values]
                mean60=sum(values60)/Decimal(len(values60))
                var60=sum((x-mean60)**2 for x in values60)/Decimal(len(values60)-1)
            precise.append(dict(repetition=rep,method=method,theta=r.theta,decimal_precision=60,
                direct_variance=float(np.var(values,ddof=1)),decimal_variance=float(var60),
                absolute_difference=abs(float(var60)-float(np.var(values,ddof=1)))))
precise=pd.DataFrame(precise);precise.to_csv(out/"high_precision_aggregation.csv",index=False)
grid=pd.read_csv(out/"grid_stability.csv")
hull=pd.read_csv(out/"convex_hull_raw.csv")
before=json.loads((out/"source_integrity_before.json").read_text())
checks=dict(
    all_24_grid_comparisons_present=len(grid)==24,
    endpoint_stability=bool(grid.maximum_endpoint_difference.max()<1e-8),
    endpoint_residuals=bool(grid.maximum_boundary_residual.max()<1e-8),
    direct_variance_matches_formula=bool(probes.formula_variance_abs_error.max()<1e-9),
    decimal_variance_matches_direct=bool(precise.absolute_difference.max()<1e-9),
    input_results_unchanged=all(hashlib.sha256((SOURCE/k).read_bytes()).hexdigest()==v for k,v in before.items()),
    hull_never_loses_observed_coverage=bool((~hull.original_coverage|hull.hull_coverage).all()),
    hull_never_shorter=bool((hull.hull_length+1e-12>=hull.original_length).all()),
    six_hull_cells_complete=len(hull)==600 and bool(hull.groupby(["auctions","method"]).size().eq(100).all()))
cache=json.loads((out/"cache_sha256.json").read_text())
checks["six_original_caches_unchanged"]=len(cache)==6 and all(
    hashlib.sha256((SOURCE/k).read_bytes()).hexdigest()==v for k,v in cache.items())
report=dict(passed=all(checks.values()),checks=checks,
    grid_checks=len(grid),high_precision_points=len(precise),
    maximum_grid_endpoint_difference=float(grid.maximum_endpoint_difference.max()),
    maximum_decimal_variance_error=float(precise.absolute_difference.max()),
    high_precision_scope="60-digit aggregation of unchanged double-precision individual scores")
(out/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
if report["passed"]:
    (out/"completed.json").write_text(json.dumps(dict(completed=True,cases=[29,50],grid_checks=len(grid),
        probe_rows=len(probes),verified=True),indent=2)+"\n")
print(json.dumps(report,indent=2))
if not report["passed"]: raise SystemExit(1)
