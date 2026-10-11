"""A benchmark-level prior from features visible at run time.

Each (subject, item) input gives a feature vector: item text statistics,
item_features structure, interactors and whether the subject runs in a
harness. A ridge regression maps a benchmark's mean feature vector to its
fitted level L_b (eta scale), with one row per training benchmark so that
large benchmarks do not dominate. At run time the regression is applied to the
single target input, which gives a covariate t_i on the eta scale.
"""
import math
import re

import numpy as np

_MC = re.compile(r"(^|\n)\s*[\(\[]?[A-Ja-j][\)\].:]\s+\S", re.M)
_CODE = re.compile(r"```|\bdef \w+\(|\bclass \w+|#include|\bfunction\b|\bimport \w+|;\s*$", re.M)
_MATH = re.compile(r"\$[^$]+\$|\\frac|\\sqrt|\\int|\\sum|\^\{")
_REPO = re.compile(r"\b(repository|repo|commit|pull request|issue|github|pytest|unit test)\b", re.I)
_AGENT = re.compile(r"\b(tool|browser|terminal|shell|click|navigate|environment|agent|website|web page|bash)\b", re.I)
_IMG = re.compile(r"<image|\[image|\bimage\b|\bpicture\b|\bfigure\b|\bdiagram\b", re.I)

_DIG = re.compile(r"\d")
_UP = re.compile(r"[A-Z]")
_NA = re.compile(r"[^\x00-\x7f]")

NAMES = ["const", "log_len", "log_lines", "mc", "n_mc", "code", "math", "repo", "agent", "img",
         "has_feat", "n_feat_keys", "interactors", "harness", "reasoning", "digits_frac", "upper_frac",
         "nonascii_frac", "question_mark"]


def features(subject, item):
    c = item.get("item_content", "") or ""
    n = item.get("_len", len(c))  # _len and _nl exist only in simulation, where content is truncated
    f = item.get("item_features", "") or ""
    head = c[:4000]
    mcs = _MC.findall(head)
    nd = len(_DIG.findall(head))
    nu = len(_UP.findall(head))
    na = len(_NA.findall(head))
    L = max(len(head), 1)
    return np.array([
        1.0,
        math.log1p(n),
        math.log1p(item.get("_nl", c.count("\n"))),
        1.0 if len(mcs) >= 2 else 0.0,
        min(len(mcs), 10) / 10.0,
        1.0 if _CODE.search(head) else 0.0,
        1.0 if _MATH.search(head) else 0.0,
        1.0 if _REPO.search(head) else 0.0,
        1.0 if _AGENT.search(head) else 0.0,
        1.0 if _IMG.search(head) else 0.0,
        1.0 if f else 0.0,
        math.log1p(f.count(";") + 1) if f else 0.0,
        1.0 if (item.get("interactors") or "") else 0.0,
        1.0 if (subject.get("harness") or "") else 0.0,
        1.0 if (subject.get("reasoning_effort") or "") else 0.0,
        nd / L, nu / L, na / L,
        1.0 if "?" in head else 0.0,
    ])


class LevelModel:
    def __init__(self, coef, mean, scale, lo=-6.0, hi=6.0, zclip=3.0):
        self.coef, self.mean, self.scale = np.asarray(coef), np.asarray(mean), np.asarray(scale)
        self.lo, self.hi, self.zclip = float(lo), float(hi), float(zclip)

    def __call__(self, subject, item):
        key = (item.get("item_content", ""), item.get("item_features", ""), item.get("interactors", ""),
               bool(subject.get("harness")), bool(subject.get("reasoning_effort")))
        v = self._memo.get(key) if hasattr(self, "_memo") else None
        if v is None:
            x = np.clip((features(subject, item) - self.mean) / self.scale, -self.zclip, self.zclip)
            x[0] = 1.0
            v = float(np.clip(x @ self.coef, self.lo, self.hi))
            if not hasattr(self, "_memo") or len(self._memo) > 200000:
                self._memo = {}
            self._memo[key] = v
        return v

    @classmethod
    def fit(cls, bench_X, bench_L, ridge=10.0, w=None):
        X = np.asarray(bench_X, float)
        w = np.ones(len(X)) if w is None else np.asarray(w, float)
        w = w / w.sum()
        mean = w @ X
        scale = np.sqrt(w @ (X - mean) ** 2) + 1e-6
        mean[0], scale[0] = 0.0, 1.0
        Z = np.clip((X - mean) / scale, -3.0, 3.0)
        Z[:, 0] = 1.0
        lam = np.eye(Z.shape[1]) * ridge
        lam[0, 0] = 1e-6
        y = np.asarray(bench_L, float)
        coef = np.linalg.solve(Z.T @ (Z * w[:, None]) * len(np.unique(y)) + lam,
                               Z.T @ (w * y) * len(np.unique(y)))
        return cls(coef, mean, scale, y.min(), y.max())

    def to_dict(self):
        return {"coef": self.coef.tolist(), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "lo": self.lo, "hi": self.hi, "zclip": self.zclip}


_TOK = re.compile(r"[a-z][a-z0-9_]{1,24}")


def bow(item, dim=4096, head=4000):
    """l2-normalized hashed bag of lowercase word tokens (crc32 buckets, sublinear tf)."""
    import zlib
    c = (item.get("item_content", "") or "")[:head].lower()
    f = (item.get("item_features", "") or "").lower()
    v = np.zeros(dim)
    for tok in _TOK.findall(c):
        v[zlib.crc32(tok.encode()) % dim] += 1.0
    for tok in _TOK.findall(f):
        v[zlib.crc32(("f:" + tok).encode()) % dim] += 1.0
    v = np.log1p(v)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class BowLevelModel:
    """Level = handcrafted LevelModel + ridge on a hashed bag of words fitted to its residual."""

    def __init__(self, base, w, dim, lo, hi):
        self.base, self.w, self.dim, self.lo, self.hi = base, np.asarray(w), int(dim), float(lo), float(hi)
        self._memo = {}

    def __call__(self, subject, item):
        key = (item.get("item_content", ""), item.get("item_features", ""), item.get("interactors", ""),
               bool(subject.get("harness")), bool(subject.get("reasoning_effort")))
        v = self._memo.get(key)
        if v is None:
            v = float(np.clip(self.base(subject, item) + bow(item, self.dim) @ self.w, self.lo, self.hi))
            if len(self._memo) > 200000:
                self._memo = {}
            self._memo[key] = v
        return v

    def to_dict(self):
        return {"kind": "bow", "base": self.base.to_dict(), "w": self.w.tolist(), "dim": self.dim,
                "lo": self.lo, "hi": self.hi}
