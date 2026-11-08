"""Oracle score comparison for diagnosis only, never for feasible intervals."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from baseline_direct_inversion import SimulationConfig, valuation_distribution
from calibrated_wgan_pilot import scores
from orthogonal_wgan_dml import gaussian_kernel_moments

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir",type=Path,required=True)
args=parser.parse_args()
raw=pd.read_csv(args.output_dir/"raw_results.csv")
target=float(raw.target.iloc[0])
dist=valuation_distribution(SimulationConfig(bidders=5))
population=dist.ppf(((np.arange(100000)+.5)/100000)**.2)
a,g=gaussian_kernel_moments(target,population,.15)
oracle={}
for n in (500,2000):
    for rep in range(100):
        observed=dist.rvs(size=(n,5),random_state=np.random.default_rng(202611080+n*100+rep)).max(axis=1)
        psi=scores(target,observed,a[0],g[0],.15,5)
        oracle[n,rep+1]=(psi.mean(),np.std(psi,ddof=1))
raw["oracle_mean"]=[oracle[n,r][0] for n,r in zip(raw.auctions,raw.repetition)]
raw["oracle_sd"]=[oracle[n,r][1] for n,r in zip(raw.auctions,raw.repetition)]
out=[]
for (n,method),d in raw.groupby(["auctions","method"],sort=False):
    estimated=np.sqrt(n)*d.score_mean_at_target.to_numpy()
    truth=np.sqrt(n)*d.oracle_mean.to_numpy()
    gap=estimated-truth
    out.append(dict(auctions=n,method=method,
        estimated_mean_sd=estimated.std(ddof=1),oracle_mean_sd=truth.std(ddof=1),
        mean_estimated_score_sd=d.score_sd_at_target.mean(),
        mean_oracle_score_sd=d.oracle_sd.mean(),
        estimated_sd_over_oracle_sd=estimated.std(ddof=1)/truth.std(ddof=1),
        mean_estimated_score_sd_over_empirical_sd=d.score_sd_at_target.mean()/estimated.std(ddof=1),
        root_n_gap_mean=gap.mean(),root_n_gap_sd=gap.std(ddof=1),
        gap_sd_over_oracle_mean_sd=gap.std(ddof=1)/truth.std(ddof=1),
        gap_oracle_correlation=np.corrcoef(gap,truth)[0,1],
        variance_oracle=truth.var(ddof=1),variance_gap=gap.var(ddof=1),
        twice_covariance=2*np.cov(truth,gap,ddof=1)[0,1]))
result=pd.DataFrame(out)
result.to_csv(args.output_dir/"oracle_gap_summary.csv",index=False)
print(result.to_string(index=False))
