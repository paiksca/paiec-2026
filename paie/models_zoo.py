"""Model factories for cross-validation: build(name, train_pair_table, pool) -> model."""
import math
import numpy as np

from dataclasses import replace

from paie.model import HierModel, Hyper

_FIT_CACHE = {}


class Const:
    def __init__(self, p=0.5):
        self.p = p

    def predict_pairs(self, targets, labeled):
        return [np.full(len(its), self.p) for _, its in targets]


class Smooth:
    """Own-pair label mean shrunk to a prior mean m with pseudo-count a."""

    def __init__(self, m, a=2.0):
        self.m, self.a = m, a

    def predict_pairs(self, targets, labeled):
        out = []
        for s, its in targets:
            b = its[0]["benchmark_id"]
            k = n = 0
            for (ls, li), y in labeled:
                if li["benchmark_id"] == b and ls == s:
                    k += y
                    n += 1
            out.append(np.full(len(its), (k + self.a * self.m) / (n + self.a)))
        return out


def build(name, train, pool):
    subj_of = pool.subject
    # benchmark-level mean accuracy, averaged equally over benchmarks
    bacc = train.groupby("bench").acc.mean()
    if name == "const":
        return Const(0.5)
    if name == "smooth":
        return Smooth(float(bacc.mean()), 2.0)
    kw = {}
    for part in name.split(":")[1:]:
        k, v = part.split("=")
        kw[k] = float(v)
    base = name.split(":")[0]
    if base in ("hier", "hierlev", "hierbow"):
        fit_kw = {k: kw.pop(k) for k in list(kw) if k in ("se2",)}
        ridge = kw.pop("ridge", 10.0)
        bowridge = kw.pop("bowridge", 1.0)
        infl = kw.pop("infl", 1.3)
        key = (hash((len(train), tuple(sorted(train.bench.unique())))), base, tuple(sorted(fit_kw.items())), ridge, bowridge)
        if key not in _FIT_CACHE:
            m = HierModel.fit(train, subj_of, **fit_kw)
            resid = None
            if base in ("hierlev", "hierbow"):
                lev, resid = fit_level(train, pool, m.L, ridge)
                m.text = lev
            if base == "hierbow":
                m.text, _ = fit_bow(train, pool, m.L, lev, bowridge)  # keep the handcrafted residual sd: the in-sample bow residual is optimistic
            _FIT_CACHE.clear() if len(_FIT_CACHE) > 8 else None
            _FIT_CACHE[key] = (m, resid)
        m0, resid = _FIT_CACHE[key]
        h = replace(m0.h)
        if base in ("hierlev", "hierbow"):
            h.beta, h.mu0 = 1.0, 0.0
            h.s_mu = infl * resid
        for k, v in kw.items():
            setattr(h, k, v)
        m = HierModel(h, m0.sp, m0.text)
        m.L = m0.L
        return m
    raise ValueError(name)


_FEAT_CACHE = {}


def bench_features(pool, bench, train, n_items=60, seed=0):
    key = (bench, n_items, seed)
    if key in _FEAT_CACHE:
        return _FEAT_CACHE[key]
    from paie.level import features
    rng = np.random.default_rng(seed)
    sids = train[train.bench == bench].subject_id.to_numpy()
    its = pool.I[pool.I.bench == bench]
    idx = rng.choice(len(its), size=min(n_items, len(its)), replace=False)
    X = []
    for j in idx:
        r = its.iloc[j]
        it = {"item_content": r.content, "item_features": r.item_features, "interactors": ""}
        s = pool.subject(bench, sids[rng.integers(len(sids))])
        X.append(features(s, it))
    _FEAT_CACHE[key] = np.stack(X)
    _ITEMS_CACHE[key] = [pool.item(bench, its.iloc[j].item_id) for j in idx]
    return _FEAT_CACHE[key]


_ITEMS_CACHE = {}
_BOW_CACHE = {}


def bench_bow(pool, bench, train, dim=4096):
    from paie.level import bow
    bench_features(pool, bench, train)
    key = (bench, dim)
    if key not in _BOW_CACHE:
        _BOW_CACHE[key] = np.stack([bow(it, dim) for it in _ITEMS_CACHE[(bench, 60, 0)]])
    return _BOW_CACHE[key]


def fit_bow(train, pool, L, base, lam, dim=4096):
    """Ridge on hashed words for the residual of the handcrafted level model; benchmark-weighted rows."""
    from paie.level import BowLevelModel
    benches = sorted(L)
    Xs = [bench_bow(pool, b, train, dim) for b in benches]
    Fs = [bench_features(pool, b, train) for b in benches]
    rs, ws = [], []
    for b, F in zip(benches, Fs):
        Z = np.clip((F - base.mean) / base.scale, -base.zclip, base.zclip)
        Z[:, 0] = 1.0
        p = np.clip(Z @ base.coef, base.lo, base.hi)
        rs.append(L[b] - p)
        ws.append(np.full(len(F), 1.0 / len(F)))
    X, r, w = np.concatenate(Xs), np.concatenate(rs), np.concatenate(ws)
    A = X.T @ (X * w[:, None]) + lam * np.eye(dim)
    coef = np.linalg.solve(A, X.T @ (w * r))
    model = BowLevelModel(base, coef, dim, base.lo, base.hi)
    i0 = np.cumsum([0] + [len(x) for x in Xs])
    pb = np.array([(X[i0[k]:i0[k + 1]] @ coef + (L[b] - rs[k])).mean() for k, b in enumerate(benches)])
    yb = np.array([L[b] for b in benches])
    return model, float(np.std(yb - pb))


def fit_level(train, pool, L, ridge):
    """Item-level ridge: rows are up to 60 items per training benchmark, target the benchmark's L."""
    from paie.level import LevelModel
    benches = sorted(L)
    Xs = [bench_features(pool, b, train) for b in benches]
    X = np.concatenate(Xs)
    y = np.concatenate([np.full(len(x), L[b]) for x, b in zip(Xs, benches)])
    w = np.concatenate([np.full(len(x), 1.0 / len(x)) for x in Xs])
    lev = LevelModel.fit(X, y, ridge, w)
    Z = np.clip((X - lev.mean) / lev.scale, -3, 3)
    Z[:, 0] = 1.0
    pred = Z @ lev.coef
    i0 = np.cumsum([0] + [len(x) for x in Xs])
    pb = np.array([pred[i0[k]:i0[k + 1]].mean() for k in range(len(benches))])
    yb = np.array([L[b] for b in benches])
    return lev, float(np.std(yb - pb))
