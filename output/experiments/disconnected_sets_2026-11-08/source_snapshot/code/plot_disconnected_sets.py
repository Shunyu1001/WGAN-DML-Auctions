"""Plot the mean, scale and statistic responsible for disconnected sets."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",type=Path,required=True)
out=p.parse_args().output_dir
curves=pd.read_csv(out/"score_curves.csv")
cases=json.loads((out/"cases.json").read_text())
styles={"Calibrated WGAN-DML":("#176D86","Calibrated GAN"),
        "Empirical-local DML":("#5A9B72","Empirical DML"),
        "Smooth plug-in":("#B67A48","Smooth plug-in")}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
    "axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(2,3,figsize=(12.8,7),layout="constrained")
for row,rep in enumerate((29,50)):
    for method,(color,label) in styles.items():
        x=curves[(curves.repetition==rep)&(curves.method==method)]
        axes[row,0].plot(x.theta,x["mean"],color=color,label=label,lw=1.6)
        axes[row,1].plot(x.theta,x.sd,color=color,lw=1.6)
        axes[row,2].plot(x.theta,x.t,color=color,lw=1.6)
    case=next(c for c in cases if c["repetition"]==rep and c["method"]=="Calibrated WGAN-DML")
    for lower,upper in case["saved_components"]:
        axes[row,2].axvspan(lower,upper,color="#CDE3DC",alpha=.45,zorder=-2)
    for ax in axes[row]:
        ax.axvline(case["target"],color="#89999D",ls=":",lw=1)
        ax.set(xlabel="Candidate reserve",xlim=(.45,1.15))
    axes[row,0].axhline(0,color="#89999D",ls="--",lw=.8)
    axes[row,0].set(title=f"Repetition {rep}: score mean",ylabel="Mean score")
    axes[row,1].set(title="Score standard deviation",ylabel="SD (log scale)",yscale="log")
    axes[row,2].axhline(1.96,color="#AF594C",ls="--",lw=1)
    axes[row,2].axhline(-1.96,color="#AF594C",ls="--",lw=1)
    axes[row,2].set(title="Studentized score",ylabel="Test statistic",ylim=(-5,6))
axes[0,0].legend(frameon=False,fontsize=8)
fig.suptitle("Low-tail variance inflation produces a second acceptance region",fontsize=14)
fig.savefig(out/"disconnected_sets.png",dpi=180)
