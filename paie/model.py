"""Hierarchical logistic predictor with priors fitted on the public pool.

    eta = mu_b + theta_s + delta_sb + beta * t_i + e_i,   e_i ~ N(0, se2)
    p   = E[sigmoid(eta)]

mu_b      benchmark level, prior N(mu0, s_mu^2), shared by every subject's
          labels on the anonymous benchmark_id.
theta_s   subject standing keyed on normalized_name, prior mean from the name's
          standing on training benchmarks (shrunk) or, for an unseen name, from
          subject attributes; prior sd s_th.
delta_sb  the pair's own deviation, prior N(0, s_de^2).
t_i       an optional item covariate (a text model's logit), coefficient beta.
e_i       item residual, integrated out by the probit approximation.

Everything is refitted from `labeled` on each call (Laplace approximation by
Newton's method), so predictions are a pure function of (input, labeled).
"""
import json
import math
from dataclasses import dataclass, asdict, field

import numpy as np

PI8 = math.pi / 8


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def subject_key(s):
    n = s.get("normalized_name", "") or ""
    return n if n else json.dumps(s, sort_keys=True)


def pair_key(s):
    return json.dumps(s, sort_keys=True)


def _year(d):
    try:
        y, m = d[:4], d[5:7] if len(d) >= 7 else "06"
        return int(y) + (int(m) - 1) / 12.0 - 2023.0
    except Exception:
        return None


@dataclass
class Hyper:
    mu0: float = -0.5
    s_mu: float = 1.5
    s_th: float = 1.0
    s_de: float = 0.8
    se2: float = 3.0
    beta: float = 0.0
    name_shrink: float = 2.0      # pseudo-count for a name's standing
    attr_scale: float = 1.0       # multiplies the subject prior mean
    level_item: float = 0.0       # 1: text covariate per item; 0: benchmark mean of it as mu_b's prior mean
    line: float = 1.0             # 1: integrate the target's linear predictor exactly along the Laplace line
    s_g: float = 0.0              # sd of a run-level shift shared by every benchmark (0: off)
    s_grp: float = 0.0            # sd of item_features group effects within a benchmark (0: off)


@dataclass
class SubjectPrior:
    standing: dict = field(default_factory=dict)   # normalized_name -> (mean, n_benchmarks)
    coef: list = field(default_factory=list)       # attribute regression
    providers: list = field(default_factory=list)

    def feats(self, s):
        y = _year(s.get("release_date", "") or "")
        x = [1.0, 0.0 if y is None else y, 1.0 if y is None else 0.0,
             1.0 if (s.get("reasoning_effort") or "") else 0.0,
             1.0 if (s.get("harness") or "") else 0.0]
        x += [1.0 if (s.get("provider") or "") == p else 0.0 for p in self.providers]
        return np.array(x)

    def attr_mean(self, s):
        if not self.coef:
            return 0.0
        return float(self.feats(s) @ np.asarray(self.coef))

    def mean(self, s, shrink):
        a = self.attr_mean(s)
        k = s.get("normalized_name", "") or ""
        if k in self.standing:
            m, n = self.standing[k]
            return (n * m + shrink * a) / (n + shrink)
        return a


class HierModel:
    def __init__(self, hyper=None, sprior=None, text=None):
        self.h = hyper or Hyper()
        self.sp = sprior or SubjectPrior()
        self.text = text            # callable item dict -> covariate t_i, or None
        self._cache = {}

    # ---------- offline fitting on training pairs ----------
    @classmethod
    def fit(cls, pair_table, subj_of, ridge=1.0, n_iter=30, text=None, **hk):
        """pair_table: DataFrame bench, subject_id, n, k (successes); subj_of(bench, sid) -> dict.
        Fits logit acc_{s,b} ~ L_b + A_name on the accuracy-logit scale, then maps to eta
        scale with the attenuation of Hyper.se2."""
        h = Hyper(**hk)
        kappa = 1.0 / math.sqrt(1.0 + PI8 * h.se2)
        pt = pair_table.copy()
        pt["z"] = np.log((pt.k + 0.5) / (pt.n - pt.k + 0.5)) / kappa
        subs = [subj_of(b, s) for b, s in zip(pt.bench, pt.subject_id)]
        pt["name"] = [(x.get("normalized_name") or "") for x in subs]
        benches = sorted(pt.bench.unique())
        L = pt.groupby("bench").z.mean().to_dict()
        A = {}
        named = pt.name != ""
        for _ in range(n_iter):
            pt["L"] = pt.bench.map(L)
            r = (pt.z - pt.L)[named]
            g = r.groupby(pt.name[named])
            A = (g.sum() / (g.size() + ridge)).to_dict()
            pt["A"] = pt.name.map(A).fillna(0.0)
            L = (pt.z - pt.A).groupby(pt.bench).mean().to_dict()
        Lv = np.array([L[b] for b in benches])
        h.mu0 = float(Lv.mean()) if "mu0" not in hk else h.mu0
        h.s_mu = float(Lv.std()) if "s_mu" not in hk else h.s_mu
        resid = pt.z - pt.bench.map(L) - pt.A
        if "s_de" not in hk:
            h.s_de = float(np.sqrt(max(resid[named].var() * 0.8, 0.05)))
        # standing per name with the number of benchmarks it was seen on
        nb = pt[named].groupby("name").bench.nunique().to_dict()
        rel = (pt.z - pt.bench.map(L))[named].groupby(pt.name[named]).mean().to_dict()
        sp = SubjectPrior()
        sp.standing = {k: (float(rel[k]), int(nb[k])) for k in rel}
        prov = pd_counts([x.get("provider") or "" for x in subs])
        sp.providers = [p for p, c in prov if p and c >= 20][:12]
        X = np.stack([sp.feats(x) for x in subs])
        y = (pt.z - pt.bench.map(L)).to_numpy()
        lam = np.eye(X.shape[1]) * 5.0
        lam[0, 0] = 1e-6
        sp.coef = list(np.linalg.solve(X.T @ X + lam, X.T @ y))
        if "s_th" not in hk:
            pm = np.array([sp.attr_mean(x) for x in subs])
            h.s_th = float(np.sqrt(max(np.var(y - pm) - h.s_de ** 2, 0.1)))
        out = cls(h, sp, text)
        out.L = L
        return out

    # ---------- online inference ----------
    def _posterior(self, labeled, key=None):
        key = _fingerprint(labeled) if key is None else key
        if key in self._cache:
            return self._cache[key]
        h = self.h
        kappa = 1.0 / math.sqrt(1.0 + PI8 * h.se2)
        bids, sks, pks, grps = {}, {}, {}, {}
        rows = []
        tsum = {}
        for (s, it), y in labeled:
            b = it.get("benchmark_id", "")
            bi = bids.setdefault(b, len(bids))
            sk = subject_key(s)
            si = sks.setdefault(sk, (len(sks), s))[0]
            pk = (b, pair_key(s))
            pi = pks.setdefault(pk, len(pks))
            gl = [grps.setdefault(g, len(grps)) for g in groups_of(b, it)] if h.s_grp > 0 else []
            t = self.text(s, it) if self.text else 0.0
            if h.level_item:
                rows.append((bi, si, pi, gl, h.beta * t, y))
            else:
                rows.append((bi, si, pi, gl, 0.0, y))
                a = tsum.setdefault(bi, [0.0, 0])
                a[0] += t
                a[1] += 1
        nb, ns, npair, ng = len(bids), len(sks), len(pks), len(grps)
        use_g = h.s_g > 0
        o_g = nb + ns + npair
        n = o_g + ng + (1 if use_g else 0)
        gi = n - 1 if use_g else -1
        m0 = np.zeros(n)
        v0 = np.zeros(n)
        m0[:nb], v0[:nb] = h.mu0, h.s_mu ** 2
        for bi, (ts, tn) in tsum.items():
            m0[bi] += h.beta * ts / tn
        for sk, (si, s) in sks.items():
            m0[nb + si] = h.attr_scale * self.sp.mean(s, h.name_shrink)
            v0[nb + si] = h.s_th ** 2
        v0[nb + ns:o_g] = h.s_de ** 2
        v0[o_g:o_g + ng] = h.s_grp ** 2
        if use_g:
            v0[gi] = h.s_g ** 2
        x = m0.copy()
        if rows:
            A = np.zeros((len(rows), n))
            off = np.zeros(len(rows))
            y = np.zeros(len(rows))
            for r, (bi, si, pi, gl, o, yy) in enumerate(rows):
                A[r, bi] = A[r, nb + si] = A[r, nb + ns + pi] = 1.0
                for g in gl:
                    A[r, o_g + g] = 1.0
                if use_g:
                    A[r, gi] = 1.0
                off[r], y[r] = o, yy
            prec0 = 1.0 / v0
            for _ in range(50):
                p = _sig(kappa * (A @ x + off))
                g = A.T @ (kappa * (y - p)) - (x - m0) * prec0
                H = (A.T * (kappa ** 2 * p * (1 - p))) @ A + np.diag(prec0)
                step = np.linalg.solve(H, g)
                x = x + step
                if np.max(np.abs(step)) < 1e-8:
                    break
            p = _sig(kappa * (A @ x + off))
            H = (A.T * (kappa ** 2 * p * (1 - p))) @ A + np.diag(prec0)
            S = np.linalg.inv(H)
            lik = (A, off, y, kappa, m0, prec0)
        else:
            S = np.diag(v0)
            lik = None
        post = (bids, sks, pks, nb, ns, x, S, lik, {}, gi, grps, o_g)
        self._cache = {key: post}
        return post

    def predict_one(self, subject, item, labeled, post=None):
        bids, sks, pks, nb, ns, x, S, lik, lines, gi, grps, o_g = post or self._posterior(labeled)
        h = self.h
        b = item.get("benchmark_id", "")
        m, v = 0.0, 0.0
        ids = []
        if b in bids:
            ids.append(bids[b])
        else:
            m += h.mu0
            v += h.s_mu ** 2
            if self.text and not h.level_item:
                m += h.beta * self.text(subject, item)
        sk = subject_key(subject)
        if sk in sks:
            ids.append(nb + sks[sk][0])
        else:
            m += h.attr_scale * self.sp.mean(subject, h.name_shrink)
            v += h.s_th ** 2
        pk = (b, pair_key(subject))
        if pk in pks:
            ids.append(nb + ns + pks[pk])
        else:
            v += h.s_de ** 2
        if h.s_grp > 0:
            for g in groups_of(b, item):
                if g in grps:
                    ids.append(o_g + grps[g])
                else:
                    v += h.s_grp ** 2
        if h.s_g > 0:
            if gi >= 0:
                ids.append(gi)
            else:
                v += h.s_g ** 2
        if self.text and h.level_item:
            m += h.beta * self.text(subject, item)
        v += h.se2
        if ids and lik is not None and h.line:
            key = tuple(ids)
            if key not in lines:
                lines[key] = _line(x, S, np.array(ids), lik)
            s, w = lines[key]
            return float(w @ _sig((s + m) / math.sqrt(1.0 + PI8 * v)))
        if ids:
            ids = np.array(ids)
            m += x[ids].sum()
            v += S[np.ix_(ids, ids)].sum()
        return float(_sig(m / math.sqrt(1.0 + PI8 * v)))

    def predict_pairs(self, targets, labeled):
        post = self._posterior(labeled, key=("sim", id(labeled), len(labeled)))
        self._cache = {}
        return [np.array([self.predict_one(s, it, labeled, post) for it in its]) for s, its in targets]

    def predict(self, input, labeled=None):
        s, it = input
        try:
            return self.predict_one(s, it, labeled or [])
        except Exception:
            return 0.5


_Z = np.linspace(-6.0, 6.0, 81)


def _line(x, S, ids, lik):
    """Exact log posterior along the Laplace line of a'x (a = indicator of ids): grid values and weights."""
    A, off, y, kappa, m0, prec0 = lik
    Sa = S[:, ids].sum(1)
    va = Sa[ids].sum()
    sh = x[ids].sum()
    sd = math.sqrt(max(va, 1e-12))
    s = sh + sd * _Z
    X = x[None, :] + np.outer(s - sh, Sa / va)
    eta = X @ A.T + off[None, :]
    z = kappa * eta
    ll = (y * -np.logaddexp(0, -z) + (1 - y) * -np.logaddexp(0, z)).sum(1)
    lp = ll - 0.5 * (((X - m0) ** 2) * prec0).sum(1)
    w = np.exp(lp - lp.max())
    return s, w / w.sum()


def groups_of(b, item):
    """(benchmark, key, value) for each non-numeric key=value in item_features."""
    f = item.get("item_features", "") or ""
    out = []
    for kv in f.split(";"):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        v = v.strip()
        if not v or len(v) > 60:
            continue
        try:
            float(v)
            continue
        except ValueError:
            pass
        out.append((b, k.strip(), v))
    return out


def _fingerprint(labeled):
    return hash(json.dumps(labeled, sort_keys=True)) if labeled else 0


def pd_counts(xs):
    from collections import Counter
    return Counter(xs).most_common()


def to_json(model):
    d = {"kind": "paie-hier-v1", "hyper": asdict(model.h),
         "standing": {k: list(v) for k, v in model.sp.standing.items()},
         "coef": [float(c) for c in model.sp.coef], "providers": list(model.sp.providers),
         "level": model.text.to_dict() if model.text is not None else None}
    return d


def from_json(d):
    from .level import LevelModel
    sp = SubjectPrior({k: (float(v[0]), int(v[1])) for k, v in d["standing"].items()},
                      list(d["coef"]), list(d["providers"]))
    from .level import BowLevelModel
    def _lev(lev):
        if lev is None:
            return None
        if lev.get("kind") == "bow":
            return BowLevelModel(_lev(lev["base"]), lev["w"], lev["dim"], lev["lo"], lev["hi"])
        return LevelModel(lev["coef"], lev["mean"], lev["scale"], lev["lo"], lev["hi"], lev["zclip"])
    text = _lev(d.get("level"))
    return HierModel(Hyper(**d["hyper"]), sp, text)
