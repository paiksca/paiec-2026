"""Fetch status and scoring logs of our submissions (results/submissions.tsv) and cache logs in results/platform/.

    python scripts/feedback.py          # prints id, label, status, ALC and per-budget mean Brier
"""
import json, os, re, sys, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
import cb
W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(W, "results/platform")
os.makedirs(OUT, exist_ok=True)
ROW = re.compile(r"^(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", re.M)


def parse(text):
    out = {}
    head, *parts = re.split(r"Label budget: (\d+)", text)
    for s, b, n, alc, _ in ROW.findall(head):
        out.setdefault((s, b), {})["alc"] = float(alc)
    for budget, body in zip(parts[0::2], parts[1::2]):
        for s, b, n, br, _ in ROW.findall(body):
            out.setdefault((s, b), {})[int(budget)] = float(br)
    return out


tok = cb.token()
rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(W, "results/submissions.tsv"))]
for when, sid, zp, sha, label in rows:
    f = os.path.join(OUT, f"{sid}.txt")
    if os.path.exists(f):
        status = "Finished"
    else:
        try:
            st, d = cb.get(f"/api/submissions/{sid}/", tok)
            status = d.get("status") if isinstance(d, dict) else st
        except Exception as e:
            status = f"error {type(e).__name__}"
    if not os.path.exists(f) and status in ("Finished", "Failed"):
      try:
        s2, det = cb.get(f"/api/submissions/{sid}/get_details/", tok)
        log = next((l for l in det.get("logs", []) if l["name"] == "scoring_stdout"), None)
        if log and log.get("data_file"):
            text = urllib.request.urlopen(log["data_file"], timeout=60).read().decode()
            if "Label budget: 31" in text:  # an infra-failed run can still hold a complete scoring log
                open(f, "w").write(text)
      except Exception as e:
        status = f"log error {type(e).__name__}"
    line = f"{sid}\t{label}\t{status}"
    if os.path.exists(f):
        p = parse(open(f).read())
        alc = sum(v["alc"] for v in p.values()) / len(p)
        bud = [sum(v.get(B, 0) for v in p.values()) / len(p) for B in (0, 1, 3, 7, 15, 31)]
        line += f"\tALC {alc:.4f}\tpairs {len(p)}\tB " + " ".join(f"{x:.3f}" for x in bud)
    print(line)
