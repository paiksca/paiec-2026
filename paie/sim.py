"""Fast simulation of formative-like runs under the official streaming protocol.

A run holds 8 or 9 pairs over about 7 benchmarks; a quarter of benchmark slots
hold two pairs, as in 55 public formative feedback logs. Each pair's items are
cut to twice an evaluation size drawn from the logged sizes, then split 50/50
into acquisition and evaluation pools. The platform's random acquisition policy
takes a uniformly random ordered subset, so the acquired stream is the first
B items of a random permutation of the acquisition pool, graded by each item's
first recorded trial. At budget B the shared `labeled` holds the first B labels
of every pair in the run. Brier counts every trial of every evaluation item.
"""
import hashlib

import numpy as np

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
# evaluation items per pair in 55 public formative logs (scripts/feedback_profile.py)
EVAL_SIZES = (44, 44, 50, 53, 54, 56, 56, 58, 60, 75, 76, 113)


def anon(kind, name, salt=""):
    h = int(hashlib.sha256(f"{salt}|{kind}|{name}".encode()).hexdigest()[:12], 16)
    return f"{kind}_{100000 + h % 900000}"


class RunPair:
    __slots__ = ("bench", "subject_id", "subject", "acq", "ev", "ev_y", "bid")

    def __init__(self, pair, rng, n_eval=None, bid=None):
        items = list(pair.items)
        rng.shuffle(items)
        n_eval = n_eval or int(rng.choice(EVAL_SIZES))
        items = items[: min(len(items), 2 * n_eval)]
        half = len(items) // 2
        self.ev = items[:half]
        self.acq = items[half:]          # already in random stream order
        self.ev_y = [pair.ys[i] for i in self.ev]
        self.bench, self.subject_id, self.subject = pair.bench, pair.subject_id, pair.subject
        self.bid = bid
        self.acq = [(i, pair.ys[i][0]) for i in self.acq]


def sample_run(pairs_by_bench, rng, n_pairs=None, n_bench=None, double_rate=0.25):
    """Pairs for one run from {bench: [Pair]}; benchmarks chosen uniformly."""
    benches = sorted(pairs_by_bench)
    n_pairs = n_pairs or int(rng.choice([8, 9, 9]))
    picks = []
    order = list(rng.permutation(benches))
    while len(picks) < n_pairs and order:
        b = order.pop()
        cand = pairs_by_bench[b]
        k = 2 if (rng.random() < double_rate and len(cand) > 1 and n_pairs - len(picks) >= 2) else 1
        for j in rng.choice(len(cand), size=k, replace=False):
            picks.append(cand[j])
    salt = str(rng.integers(1 << 30))
    return [RunPair(p, rng, bid=anon("benchmark", p.bench, salt)) for p in picks]


def labeled_at(run, B, item_of):
    out = []
    for rp in run:
        for iid, y in rp.acq[:B]:
            it = item_of(rp.bench, iid)
            it["benchmark_id"] = rp.bid
            out.append([[rp.subject, it], int(y)])
    return out


def score_run(run, model, item_of, budgets=BUDGETS):
    """Per-pair Brier at each budget. `model.predict_pairs(targets, labeled)` gets
    targets = [(subject, [item dicts])] and returns a list of prediction arrays."""
    res = np.zeros((len(run), len(budgets)))
    targets = []
    for rp in run:
        its = []
        for iid in rp.ev:
            it = item_of(rp.bench, iid)
            it["benchmark_id"] = rp.bid
            its.append(it)
        targets.append((rp.subject, its))
    for k, B in enumerate(budgets):
        lab = labeled_at(run, B, item_of)
        preds = model.predict_pairs(targets, lab)
        for j, rp in enumerate(run):
            p = np.asarray(preds[j], float)
            se, n = 0.0, 0
            for pi, ys in zip(p, rp.ev_y):
                for y in ys:
                    se += (pi - y) ** 2
                    n += 1
            res[j, k] = se / n
    return res


def alc(brier_rows):
    return np.asarray(brier_rows) @ np.asarray(WEIGHTS)
