"""Codabench API helper. Reads credentials from the environment and never prints them."""
import json, os, sys, urllib.request, urllib.parse

BASE = "https://www.codabench.org"
COMP, PHASE = 17828, 29785

def token():
    data = urllib.parse.urlencode({"username": os.environ["CODABENCH_USERNAME"],
                                   "password": os.environ["CODABENCH_PASSWORD"]}).encode()
    with urllib.request.urlopen(urllib.request.Request(BASE + "/api/api-token-auth/", data=data)) as r:
        return json.load(r)["token"]

def get(path, tok):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Token {tok}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:500].decode(errors="replace")

if __name__ == "__main__":
    tok = token()
    for p in sys.argv[1:]:
        s, d = get(p, tok)
        print(p, s)
        print(json.dumps(d, indent=1)[:4000] if not isinstance(d, str) else d)
