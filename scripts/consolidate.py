"""Consolidate downloaded measurement-db tables into three parquet files under data/consolidated.

Keeps only binary grades (0 or 1). A benchmark directory counts if its metadata
says binary or mixed, item granularity and release public.
"""
import glob, json, os, sys
import pandas as pd

ROOT = os.environ.get("PAIE_DATA_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))
HF, OUT = os.path.join(ROOT, "hf"), os.path.join(ROOT, "consolidated")
os.makedirs(OUT, exist_ok=True)
SUBJ = ["normalized_name", "provider", "release_date", "access_date", "harness",
        "harness_version", "reasoning_effort", "subject_features_extra"]

meta = pd.read_parquet(os.path.join(ROOT, "meta/benchmarks_all.parquet"))
keep = meta[meta.response_type.isin(["binary", "mixed"]) & (meta.granularity == "item") & (meta.release == "public")]
MAX_ITEMS = 3000
HEAD = 4000  # features read only the first 4000 characters, plus full length and line count
R, S, I, info = [], [], [], []
for row in keep.itertuples():
    d = os.path.join(HF, row.dir, "formatted_tables")
    rf = [f for f in (os.path.join(d, "response.parquet"), os.path.join(d, "responses.parquet")) if os.path.exists(f)]
    if not rf or not os.path.exists(os.path.join(d, "items.parquet")):
        print("missing tables", row.dir, file=sys.stderr); continue
    r = pd.read_parquet(rf[0], columns=["subject_id", "item_id", "trial", "response"])
    n_all = len(r)
    r = r[r.response.isin([0.0, 1.0])]
    if r.empty:
        continue
    r = r.assign(bench=row.dir, y=r.response.astype("int8")).drop(columns="response")
    ids = r.item_id.unique()
    if len(ids) > MAX_ITEMS:  # a fixed random subset of items keeps the files small
        keep_ids = pd.Series(ids).sample(MAX_ITEMS, random_state=0)
        r = r[r.item_id.isin(set(keep_ids))]
    s = pd.read_parquet(os.path.join(d, "subjects.parquet"))
    s = s[s.subject_id.isin(r.subject_id.unique())].assign(bench=row.dir)
    it = pd.read_parquet(os.path.join(d, "items.parquet"), columns=["item_id", "content", "item_features", "asset_manifest"])
    it = it[it.item_id.isin(r.item_id.unique())].assign(bench=row.dir)
    c = it.content.astype("string").fillna("")
    it = it.assign(content_len=c.str.len().astype("int64"), content_nl=c.str.count("\n").astype("int64"),
                   content=c.str.slice(0, HEAD), has_assets=it.asset_manifest.notna()).drop(columns="asset_manifest")
    R.append(r); S.append(s); I.append(it)
    info.append(dict(bench=row.dir, response_type=row.response_type, release=row.release, n_resp_all=n_all,
                     n_resp_binary=len(r), domain=list(row.domain) if row.domain is not None else [],
                     modality=list(row.modality) if row.modality is not None else [],
                     multi_single_turn=row.multi_single_turn, release_date=row.release_date))
R = pd.concat(R, ignore_index=True); S = pd.concat(S, ignore_index=True); I = pd.concat(I, ignore_index=True)
for c in SUBJ + ["display_name"]:
    S[c] = S[c].astype("string").fillna("")
I["content"] = I.content.astype("string").fillna(""); I["item_features"] = I.item_features.astype("string").fillna("")
R.to_parquet(os.path.join(OUT, "responses.parquet")); S.to_parquet(os.path.join(OUT, "subjects.parquet"))
I.to_parquet(os.path.join(OUT, "items.parquet")); pd.DataFrame(info).to_parquet(os.path.join(OUT, "benchmarks.parquet"))
print(len(info), "benchmarks", len(R), "binary responses", R.groupby(["bench", "subject_id"]).ngroups, "pairs")
