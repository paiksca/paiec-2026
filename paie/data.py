"""Load the consolidated public pool and build pair records for simulation."""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.environ.get("PAIE_DATA", os.path.join(os.environ.get("PAIE_DATA_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))), "consolidated"))
SUBJ = ["normalized_name", "provider", "release_date", "access_date", "harness",
        "harness_version", "reasoning_effort", "subject_features_extra"]


@dataclass
class Pair:
    bench: str
    subject_id: str
    subject: dict                      # the eight official subject fields
    items: list                        # item_ids, distinct
    ys: dict                           # item_id -> list of 0/1 over trials (trial order)
    extra: dict = field(default_factory=dict)


class Pool:
    def __init__(self, root=ROOT):
        self.R = pd.read_parquet(os.path.join(root, "responses.parquet"))
        self.S = pd.read_parquet(os.path.join(root, "subjects.parquet"))
        self.I = pd.read_parquet(os.path.join(root, "items.parquet"))
        self.B = pd.read_parquet(os.path.join(root, "benchmarks.parquet"))
        self.R = self.R.sort_values(["bench", "subject_id", "item_id", "trial"], kind="stable")
        self._subj = {(r.bench, r.subject_id): {f: getattr(r, f) for f in SUBJ}
                      for r in self.S.itertuples()}
        self._item = {(r.bench, r.item_id): {"item_content": r.content, "item_features": r.item_features,
                                             "interactors": "", "_len": int(r.content_len), "_nl": int(r.content_nl)}
                      for r in self.I.itertuples()}

    def subject(self, bench, sid):
        return dict(self._subj.get((bench, sid), {f: "" for f in SUBJ}))

    def item(self, bench, iid):
        return dict(self._item[(bench, iid)])

    def pairs(self, min_items=80, benches=None):
        R = self.R if benches is None else self.R[self.R.bench.isin(benches)]
        nu = R.groupby(["bench", "subject_id"]).item_id.nunique()
        ok = nu[nu >= min_items].index
        R = R.set_index(["bench", "subject_id"]).loc[ok].reset_index()
        out = []
        b_arr, s_arr, i_arr, y_arr = (R[c].to_numpy() for c in ("bench", "subject_id", "item_id", "y"))
        brk = np.flatnonzero((b_arr[1:] != b_arr[:-1]) | (s_arr[1:] != s_arr[:-1])) + 1
        for lo, hi in zip(np.r_[0, brk], np.r_[brk, len(R)]):
            ys = {}
            for iid, y in zip(i_arr[lo:hi], y_arr[lo:hi]):
                ys.setdefault(iid, []).append(int(y))
            b, s = b_arr[lo], s_arr[lo]
            out.append(Pair(b, s, self.subject(b, s), list(ys), ys))
        return out

    def pair_table(self):
        """One row per (bench, subject): n_items, n_resp, mean y."""
        g = self.R.groupby(["bench", "subject_id"])
        return pd.DataFrame({"n_items": g.item_id.nunique(), "n": g.y.size(), "acc": g.y.mean()}).reset_index()
