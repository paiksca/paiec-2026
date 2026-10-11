"""Profile the consolidated public pool: benchmarks, pairs, subjects, recency, base rates."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paie.data import ROOT
R = pd.read_parquet(os.path.join(ROOT, "responses.parquet"), columns=["bench", "subject_id", "item_id", "y"])
S = pd.read_parquet(os.path.join(ROOT, "subjects.parquet"))
g = R.groupby(["bench", "subject_id"]).agg(n_items=("item_id", "nunique"), acc=("y", "mean")).reset_index()
g = g.merge(S[["bench", "subject_id", "normalized_name", "release_date", "harness"]], on=["bench", "subject_id"], how="left")
e = g[g.n_items >= 80]
rec = e[e.release_date >= "2025-01"]
out = {
    "benchmarks": int(R.bench.nunique()),
    "benchmarks_with_pair_ge80": int(e.bench.nunique()),
    "pairs_ge80": int(len(e)),
    "pairs_ge80_recent": int(len(rec)),
    "benchmarks_with_recent_pair": int(rec.bench.nunique()),
    "distinct_names": int(S.normalized_name.replace("", np.nan).nunique()),
    "pair_acc_mean_all": round(float(e.acc.mean()), 3),
    "pair_acc_mean_recent": round(float(rec.acc.mean()), 3),
    "recent_bench_level_quantiles": [round(float(x), 3) for x in rec.groupby("bench").acc.mean().quantile([.1, .25, .5, .75, .9])],
    "recent_harness_frac": round(float((rec.harness != "").mean()), 3),
    "recent_acc_harness": round(float(rec[rec.harness != ""].acc.mean()), 3),
    "recent_acc_noharness": round(float(rec[rec.harness == ""].acc.mean()), 3),
}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../results/pool_profile.json"), "w"), indent=1)
