"""Download measurement-db formatted tables (no traces) at a pinned revision into data/hf.

Needs an HF token whose account has accepted the dataset's gated terms (HF_TOKEN in the environment).
1. Lists every benchmark directory's formatted_tables at the pinned revision (data/meta/tree.json).
2. Downloads benchmarks.parquet everywhere (data/meta/benchmarks_all.parquet).
3. Downloads items, subjects and response tables for binary or mixed, item-granularity benchmarks.
Then run scripts/consolidate.py, after which data/hf can be deleted.
"""
import concurrent.futures as cf, json, os, sys
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

R = "aims-foundations/measurement-db"
REV = "54074453da7b9911be0f48e4c560be547bc84d21"
ROOT = os.environ.get("PAIE_DATA_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))
OUT, META = os.path.join(ROOT, "hf"), os.path.join(ROOT, "meta")
os.makedirs(META, exist_ok=True)
api = HfApi()


def fetch(path):
    return hf_hub_download(R, path, repo_type="dataset", revision=REV, local_dir=OUT)


tree_f = os.path.join(META, "tree.json")
if not os.path.exists(tree_f):
    dirs = [e.path for e in api.list_repo_tree(R, repo_type="dataset", revision=REV) if "." not in e.path]

    def ls(d):
        try:
            return d, [e.path for e in api.list_repo_tree(R, path_in_repo=f"{d}/formatted_tables", repo_type="dataset", revision=REV)]
        except Exception as ex:
            return d, []
    with cf.ThreadPoolExecutor(8) as ex:
        tree = {d: fs for d, fs in ex.map(ls, dirs) if any(f.endswith("benchmarks.parquet") for f in fs)}
    json.dump(tree, open(tree_f, "w"), indent=0)
tree = json.load(open(tree_f))


def bench(d):
    try:
        return pd.read_parquet(fetch(f"{d}/formatted_tables/benchmarks.parquet")).assign(dir=d)
    except Exception as e:
        print("ERR", d, type(e).__name__, file=sys.stderr)
        return None


with cf.ThreadPoolExecutor(8) as ex:
    b = pd.concat([x for x in ex.map(bench, sorted(tree)) if x is not None], ignore_index=True)
b.to_parquet(os.path.join(META, "benchmarks_all.parquet"))
elig = sorted(b[b.response_type.isin(["binary", "mixed"]) & (b.granularity == "item")].dir)
print(len(b), "benchmarks,", len(elig), "binary or mixed with item granularity", flush=True)


def tables(d):
    names = {f.rsplit("/", 1)[-1] for f in tree[d]}
    for t in ["items.parquet", "subjects.parquet", "response.parquet", "responses.parquet"]:
        if t in names:
            fetch(f"{d}/formatted_tables/{t}")
    return d


with cf.ThreadPoolExecutor(8) as ex:
    for i, d in enumerate(ex.map(tables, elig), 1):
        if i % 20 == 0:
            print(i, d, flush=True)
json.dump({"revision": REV, "eligible": elig}, open(os.path.join(META, "eligible.json"), "w"), indent=1)
print("done")
