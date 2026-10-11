"""Learning curve: benchmark-weighted ALC by number of training benchmarks K, for hier and hierlev."""
import glob, re
import numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob("results/lc_k*_s*.parquet")):
    k, s = re.search(r"lc_k(\w+)_s(\d+)", f).groups()
    df = pd.read_parquet(f)
    pb = df.groupby(["model", "bench"])[["B0", "B1", "ALC"]].mean()
    d = pb.ALC.unstack("model")
    m = pb.groupby("model").mean()
    hl = [c for c in d.columns if c.startswith("hierlev")][0]
    hi = [c for c in d.columns if c.startswith("hier:")][0]
    diff = d[hl] - d[hi]
    rows.append(dict(K=k, seed=int(s), hier=m.loc[hi, "ALC"], hierlev=m.loc[hl, "ALC"],
                     hierlev_B0=m.loc[hl, "B0"], hier_B0=m.loc[hi, "B0"], diff=diff.mean(), se=diff.std() / np.sqrt(len(diff))))
t = pd.DataFrame(rows)
t["Kn"] = t.K.map(lambda x: 999 if x == "all" else int(x))
print(t.sort_values(["Kn", "seed"]).drop(columns="Kn").round(4).to_string(index=False))
t.to_csv("results/learning_curve.csv", index=False)
