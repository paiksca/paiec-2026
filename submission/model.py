"""Competition entry point: predict(input, labeled) -> float.

Loads the hierarchical predictor in paie_rt/ with the priors in prior.json.
Any failure falls back to the level prior alone, then to 0.5.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if not sys.path or sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

from paie_rt.model import from_json, HierModel  # noqa: E402

try:
    with open(os.path.join(_HERE, "prior.json")) as _f:
        MODEL = from_json(json.load(_f))
except Exception as _e:  # keep predicting with default hyperparameters
    print(f"prior.json unusable ({_e!r})", file=sys.stderr)
    MODEL = HierModel()


def predict(input, labeled=None):
    try:
        p = float(MODEL.predict_one(input[0], input[1], labeled or []))
        if p != p:
            return 0.5
        return min(max(p, 1e-4), 1 - 1e-4)
    except Exception:
        return 0.5
