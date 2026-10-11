"""Submit a zip to Codabench 17828 with retries; append the submission id to results/submissions.tsv.

    python scripts/submit.py dist/v1_hierlev.zip [label]
Needs CODABENCH_USERNAME and CODABENCH_PASSWORD in the environment. Never prints the token.
"""
import hashlib, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import cb
W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
zp, label = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
sha = hashlib.sha256(open(zp, "rb").read()).hexdigest()[:16]
env = dict(os.environ, CODABENCH_TOKEN=cb.token())
for attempt in range(5):
    r = subprocess.run([sys.executable, os.path.join(W, "third_party/paiec_baseline/submit_api.py"), zp],
                       env=env, capture_output=True, text=True)
    m = re.search(r"Submission created: (\d+)", r.stdout)
    if m:
        with open(os.path.join(W, "results/submissions.tsv"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime())}\t{m.group(1)}\t{os.path.basename(zp)}\t{sha}\t{label}\n")
        print("submitted", m.group(1))
        break
    err = (r.stderr or r.stdout).strip().splitlines()[-1:] 
    print("attempt", attempt, "failed:", err)
    time.sleep(30 * (attempt + 1))
