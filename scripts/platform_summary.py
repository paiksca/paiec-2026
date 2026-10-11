"""Platform repeated draws: per-version mean ALC and per-budget Brier with SE over submissions,
and paired differences on (subject, benchmark) pairs that recur across versions."""
import os, re, itertools
import numpy as np, pandas as pd
W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROW = re.compile(r"^(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", re.M)
rows = []
for l in open(os.path.join(W, "results/submissions.tsv")):
    when, sid, zp, sha, label = l.rstrip("\n").split("\t")
    f = os.path.join(W, "results/platform", f"{sid}.txt")
    if not os.path.exists(f):
        continue
    head, *parts = re.split(r"Label budget: (\d+)", open(f).read())
    rec = {}
    for s, b, n, alc, _ in ROW.findall(head):
        rec[(s, b)] = {"ALC": float(alc), "n": int(n)}
    for B, body in zip(parts[0::2], parts[1::2]):
        for s, b, n, br, _ in ROW.findall(body):
            rec[(s, b)][f"B{B}"] = float(br)
    for (s, b), v in rec.items():
        rows.append(dict(sid=sid, version=label, subject=s, bench=b, **v))
df = pd.DataFrame(rows)
cols = ["B0", "B1", "B3", "B7", "B15", "B31", "ALC"]
per_sub = df.groupby(["version", "sid"])[cols].mean()
g = per_sub.groupby("version")
out = g.mean().round(4)
out["sd_ALC"] = g.ALC.std().round(4)
out["n_sub"] = g.size()
print(out.to_string())
pv = df.groupby(["version", "subject", "bench"]).ALC.mean().unstack("version")
for a, b in itertools.combinations(sorted(pv.columns), 2):
    d = (pv[a] - pv[b]).dropna()
    if len(d) > 2:
        print(f"{a} - {b}: {d.mean():+.4f} (SE {d.std() / np.sqrt(len(d)):.4f}, {len(d)} shared pairs)")
