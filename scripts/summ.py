"""Summarize a CV result file: per-budget means (benchmark-weighted) and paired ALC differences
against a reference model with SEs clustered by benchmark.

    python scripts/summ.py results/cv_x.parquet [reference_model]
"""
import sys
import numpy as np, pandas as pd
df = pd.read_parquet(sys.argv[1])
cols = ["B0", "B1", "B3", "B7", "B15", "B31", "ALC"]
pb = df.groupby(["model", "bench"])[cols].mean()
tab = pb.groupby("model").mean()
ref = sys.argv[2] if len(sys.argv) > 2 else tab.ALC.idxmin()
d = pb.ALC.unstack("model")
diff = d.sub(d[ref], axis=0)
tab["dALC"] = diff.mean()
tab["se"] = diff.std() / np.sqrt(len(diff))
tab["n_bench"] = d.notna().sum()
pd.set_option("display.width", 220)
print(f"reference: {ref}")
print(tab.sort_values("ALC").round(4).to_string())
