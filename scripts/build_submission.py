"""Fit the shipped model on every public benchmark and write dist/<name>.zip.

    python scripts/build_submission.py --model hierlev --name v1
"""
import argparse, json, os, sys, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paie.data import Pool
from paie import models_zoo
from paie.model import to_json, from_json

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="hierlev")
ap.add_argument("--name", required=True)
ap.add_argument("--min-train-items", type=int, default=20)
a = ap.parse_args()
W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pool = Pool()
pt = pool.pair_table()
pt = pt[pt.n_items >= a.min_train_items].copy()
pt["k"] = (pt.acc * pt.n).round().astype(int)
m = models_zoo.build(a.model, pt, pool)
prior = to_json(m)
# round trip check: the reloaded model predicts identically
m2 = from_json(prior)
s = {"normalized_name": "OpenAI GPT-5", "provider": "OpenAI", "release_date": "2025-08-07", "access_date": "",
     "harness": "", "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
it = {"item_content": "What is 2+2?", "item_features": "", "interactors": "", "benchmark_id": "benchmark_1"}
assert abs(m.predict_one(s, it, []) - m2.predict_one(s, it, [])) < 1e-12
os.makedirs(os.path.join(W, "dist"), exist_ok=True)
out = os.path.join(W, "dist", f"{a.name}.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(W, "submission/model.py"), "model.py")
    z.writestr("prior.json", json.dumps(prior))
    z.writestr("paie_rt/__init__.py", "")
    for f in ("model.py", "level.py"):
        z.write(os.path.join(W, "paie", f), f"paie_rt/{f}")
json.dump(prior["hyper"], sys.stdout, indent=1)
print("\nwrote", out, os.path.getsize(out), "bytes")
