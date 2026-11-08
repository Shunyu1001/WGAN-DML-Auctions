"""Independent saved-set verification of hull and fixed-variance diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,required=True)
out=p.parse_args().output_dir
source=out.parent/"fixed_h_score_validation_2026-11-08"
original=pd.read_csv(source/"summary/raw_results.csv").set_index(["auctions","repetition","method"])
fixed=pd.read_csv(out/"fixed_variance/raw_results.csv").set_index(["auctions","repetition","method"])
checks={"600_fixed_rows":len(fixed)==600 and fixed.index.is_unique and set(fixed.index)==set(original.index)}
checks["point_estimates_unchanged"]=bool(np.allclose(fixed.estimate,original.loc[fixed.index].estimate,atol=1e-12,rtol=0))
valid=True
for _,r in fixed.iterrows():
    c=json.loads(r.fixed_components)
    if r.fixed_status=="ok":
        valid &= abs(sum(b-a for a,b in c)-r.fixed_length)<1e-12
        valid &= any(a-1e-9<=r.estimate<=b+1e-9 for a,b in c)
        valid &= bool(r.fixed_coverage)==bool(abs(r.fixed_t_at_target)<=1.96)
        valid &= bool(r.fixed_coverage)==any(a-1e-9<=r.target<=b+1e-9 for a,b in c)
    else:
        valid &= not bool(r.fixed_coverage) and len(c)==0
checks["fixed_set_identities"]=bool(valid)
summary=pd.read_csv(out/"fixed_variance/summary.csv")
okay=True
for _,s in summary.iterrows():
    g=fixed.xs((s.auctions,s.method),level=(0,2))
    if s.interval=="Wald": a,b=g.wald_coverage,g.wald_length
    elif s.interval=="Original score": a,b=g.original_score_coverage,g.original_score_length
    else: a,b=g.fixed_coverage,g.fixed_length
    okay &= abs(a.mean()-s.coverage)<1e-12 and abs(b.mean()-s.mean_length)<1e-12
checks["fixed_summary_recomputed"]=bool(okay and len(summary)==18)
cache={}
for n in (500,2000): cache.update(json.loads((out/f"fixed_variance/n{n}/cache_sha256.json").read_text()))
checks["600_cache_hashes_unchanged"]=len(cache)==600 and all(
    hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in cache.items())
before=json.loads((out/"source_integrity_before.json").read_text())
checks["prior_outputs_unchanged"]=all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in before.items())
checks["numeric_audit_passed"]=json.loads((out/"verification.json").read_text())["passed"] is True
result=dict(passed=all(checks.values()),checks=checks)
(out/"treatment_verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
if not result["passed"]:raise SystemExit(1)
