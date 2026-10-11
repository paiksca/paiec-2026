"""Run the jobs in scripts/jobs.txt locally with 2 workers, skipping jobs whose --out file exists."""
import concurrent.futures as cf, os, re, subprocess
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
jobs = [l.strip() for l in open("scripts/jobs.txt") if l.strip()]
todo = [j for j in jobs if not os.path.exists(re.search(r"--out (\S+)", j).group(1))]
print(len(todo), "jobs", flush=True)
def run(j):
    r = subprocess.run(j, shell=True, env=env)
    return r.returncode, re.search(r"--out (\S+)", j).group(1)
with cf.ThreadPoolExecutor(2) as ex:
    for rc, out in ex.map(run, todo):
        print(rc, out, flush=True)
open("results/LOCAL_DONE", "w").write("done\n")
