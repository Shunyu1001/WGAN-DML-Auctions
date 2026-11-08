"""Summarize paired interval changes and visualize the n=500 diagnosis."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir",type=Path,required=True)
args=parser.parse_args()
root=args.output_dir
raw=pd.read_csv(root/"raw_results.csv")
rows=[]
for (n,method),g in raw.groupby(["auctions","method"],sort=False):
    change=g.score_coverage.astype(int)-g.wald_coverage.astype(int)
    rows.append(dict(auctions=n,method=method,repetitions=len(g),
        gained=int((change==1).sum()),lost=int((change==-1).sum()),
        coverage_change=change.mean(),paired_mcse=change.std(ddof=1)/np.sqrt(len(g)),
        mean_length_change=(g.score_length-g.wald_length).mean(),
        mean_length_ratio=g.score_length.mean()/g.wald_length.mean()))
pd.DataFrame(rows).to_csv(root/"paired_interval_comparison.csv",index=False)

g=raw[(raw.auctions==500)&(raw.method=="Calibrated WGAN-DML")]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,3,figsize=(12.3,3.8),layout="constrained")
good=g.wald_coverage.astype(bool)
axes[0].scatter(g.loc[good,"error"],g.loc[good,"standard_error"],s=24,
                color="#287D79",alpha=.75,label="Wald covers")
axes[0].scatter(g.loc[~good,"error"],g.loc[~good,"standard_error"],s=37,
                color="#B65F42",marker="x",label="Wald misses")
axes[0].axvline(0,color="#AAB6B5",lw=1)
axes[0].set(xlabel="Estimation error",ylabel="Estimated standard error",
            title="A  Error and standard error")
axes[0].legend(frameon=False,fontsize=8,loc="upper left")
normal=norm.ppf((np.arange(len(g))+.5)/len(g))
for ax,values,title in zip(axes[1:],[g.error/g.standard_error,g.score_t_at_target],
    ["B  Wald statistic", "C  Score at the target"]):
    empirical=np.sort(values.to_numpy())
    limit=max(3,abs(empirical).max()+.15)
    ax.plot([-limit,limit],[-limit,limit],color="#AAB6B5",lw=1,ls="--")
    ax.scatter(normal,empirical,s=20,color="#287D79",alpha=.8)
    ax.axhline(-1.96,color="#B65F42",lw=.8,ls=":")
    ax.axhline(1.96,color="#B65F42",lw=.8,ls=":")
    ax.set(xlim=(-2.8,2.8),ylim=(-limit,limit),xlabel="Standard normal quantile",
           ylabel="Observed quantile",title=title)
fig.suptitle("Fixed bandwidth diagnosis: calibrated WGAN-DML, n = 500 (100 repetitions)",
             fontsize=12,fontweight="medium")
fig.savefig(root/"coverage_diagnosis.png",dpi=180)
plt.close(fig)
