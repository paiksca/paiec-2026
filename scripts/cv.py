"""Grouped cross-validation over benchmarks on formative-like simulated runs.

Every fold holds out a set of benchmarks. Model priors are fitted on the other
benchmarks only. Runs are drawn from the held-out benchmarks' pairs with at
least 80 items. All models are scored on identical runs.

    python scripts/cv.py --folds 8 --runs 40 --models const,smooth,hier --out results/cv_base.parquet
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paie.data import Pool
from paie import sim, models_zoo

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=8)
ap.add_argument("--runs", type=int, default=40, help="runs per fold")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--models", default="const,smooth,hier")
ap.add_argument("--out", required=True)
ap.add_argument("--min-train-items", type=int, default=20)
ap.add_argument("--eval-min-date", default="", help="evaluate only pairs whose subject release_date >= this, e.g. 2025-01")
ap.add_argument("--eval-bench-acc-max", type=float, default=1.0,
                help="sensitivity regime: evaluate only benchmarks whose eligible pairs' mean accuracy is below this")
ap.add_argument("--train-k", type=int, default=0, help="learning curve: keep only K random training benchmarks per fold")
ap.add_argument("--train-k-seed", type=int, default=0)
a = ap.parse_args()

t0 = time.time()
pool = Pool()
pairs = pool.pairs(min_items=80)
by_bench = {}
for p in pairs:
    if a.eval_min_date and not (p.subject["release_date"] or "") >= a.eval_min_date:
        continue
    by_bench.setdefault(p.bench, []).append(p)
if a.eval_bench_acc_max < 1.0:
    def _acc(p):
        ys = [y for v in p.ys.values() for y in v]
        return sum(ys) / len(ys)
    by_bench = {b: v for b, v in by_bench.items() if np.mean([_acc(p) for p in v]) < a.eval_bench_acc_max}
elig = sorted(by_bench)
print(f"{len(elig)} benchmarks with eligible pairs, {len(pairs)} pairs; load {time.time()-t0:.0f}s", flush=True)
ptab = pool.pair_table()
ptab = ptab[ptab.n_items >= a.min_train_items]
ptab["k"] = (ptab.acc * ptab.n).round().astype(int)

rng = np.random.default_rng(a.seed)
fold_of = {b: i % a.folds for i, b in enumerate(rng.permutation(elig))}
item_of = pool.item
rows = []
for f in range(a.folds):
    held = [b for b in elig if fold_of[b] == f]
    train = ptab[~ptab.bench.isin(held)]
    if a.train_k:
        tb = sorted(train.bench.unique())
        keep = np.random.default_rng([a.train_k_seed, f, a.train_k]).choice(len(tb), size=min(a.train_k, len(tb)), replace=False)
        train = train[train.bench.isin([tb[i] for i in keep])]
    ms = {name: models_zoo.build(name, train, pool) for name in a.models.split(",")}
    frng = np.random.default_rng([a.seed, f])
    hb = {b: by_bench[b] for b in held}
    for r in range(a.runs):
        run = sim.sample_run(hb, frng, n_bench=min(7, len(hb)))
        for name, m in ms.items():
            res = sim.score_run(run, m, item_of)
            for rp, br in zip(run, res):
                rows.append(dict(fold=f, run=r, model=name, bench=rp.bench, subject_id=rp.subject_id,
                                 n_eval=len(rp.ev), **{f"B{B}": v for B, v in zip(sim.BUDGETS, br)},
                                 ALC=float(br @ np.array(sim.WEIGHTS))))
    print(f"fold {f}: {len(held)} held-out benchmarks, {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
df.to_parquet(a.out)
# benchmark-clustered summary: mean per benchmark, then mean and SE across benchmarks
cols = [f"B{B}" for B in sim.BUDGETS] + ["ALC"]
pb = df.groupby(["model", "bench"])[cols].mean()
summ = pb.groupby("model").agg(["mean", "sem"])
pd.set_option("display.width", 200)
print(summ.xs("mean", axis=1, level=1).round(4))
print("SE (clustered by benchmark):")
print(summ.xs("sem", axis=1, level=1)["ALC"].round(4))
