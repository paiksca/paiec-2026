"""Profile public formative feedback logs: pairs per run, benchmarks per run, distinct ids, per-budget Brier."""
import re, json, sys, collections, statistics as st
from pathlib import Path

ROW = re.compile(r"^(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", re.M)
TP = Path(__file__).resolve().parents[1] / "third_party"
files = sorted((TP / "predictive-eval/arms/logs").glob("*.txt")) + sorted((TP / "PAIEC/results/formative").glob("*.txt"))

def parse(text):
    out = {}
    head, *parts = re.split(r"Label budget: (\d+)", text)
    for s, b, n, alc, _ in ROW.findall(head):
        out.setdefault((s, b), {})["alc"] = float(alc); out[(s, b)]["n"] = int(n)
    for budget, body in zip(parts[0::2], parts[1::2]):
        for s, b, n, brier, ece in ROW.findall(body):
            out.setdefault((s, b), {})[int(budget)] = float(brier)
    return out

runs = [parse(f.read_text()) for f in files]
runs = [r for r in runs if r]
npairs = [len(r) for r in runs]
nbench = [len({b for _, b in r}) for r in runs]
bench = collections.Counter(b for r in runs for _, b in r)
subj = collections.Counter(s for r in runs for s, _ in r)
pairs_per_bench = [c for r in runs for c in collections.Counter(b for _, b in r).values()]
ns = [v["n"] for r in runs for v in r.values()]
print(f"runs {len(runs)}; pairs/run {min(npairs)}-{max(npairs)} mean {st.mean(npairs):.1f}; benchmarks/run mean {st.mean(nbench):.1f}")
print(f"distinct benchmark ids {len(bench)}; distinct subject ids {len(subj)}; total pair appearances {sum(npairs)}")
print("pairs of the same benchmark within a run:", collections.Counter(pairs_per_bench))
print(f"eval items per pair: min {min(ns)} median {st.median(ns)} max {max(ns)}")
print("appearances per benchmark id:", sorted(bench.values(), reverse=True))
for B in [0, 1, 3, 7, 15, 31]:
    v = [x[B] for r in runs for x in r.values() if B in x]
    print(f"B{B}: mean Brier {st.mean(v):.4f}")

# Aggregate implied base rates from B31 Brier (low root of r(1-r) = B31). Aggregate only: no per-benchmark use.
import numpy as np
b31 = np.array([x[31] for r in runs for x in r.values() if 31 in x])
q = np.clip(b31, 0, 0.25)
r_low = (1 - np.sqrt(1 - 4 * q)) / 2
print(f"B31 quantiles: {np.round(np.quantile(b31, [.1,.25,.5,.75,.9]), 3)}")
print(f"implied rate (low root) quantiles: {np.round(np.quantile(r_low, [.1,.25,.5,.75,.9]), 3)}; mean {r_low.mean():.3f}")
b0 = np.array([x[0] for r in runs for x in r.values() if 0 in x])
print(f"B0 quantiles: {np.round(np.quantile(b0, [.1,.25,.5,.75,.9]), 3)}")
