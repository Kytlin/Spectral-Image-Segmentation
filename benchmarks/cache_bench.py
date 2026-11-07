"""Graph-cache benchmark against a running seg_server: cold (miss) vs warm (hit).

    python3 benchmarks/cache_bench.py http://localhost:8080 [REPS]

For each image in data/ and each setting, clears the cache and computes once (miss), then
computes the same parameters again (hit), so the only difference is the reused graph. Changing
only k also hits, but changes the spectral work (e.g. fewer recursive splits), which would
blur the comparison. Reports medians over REPS rounds of the engine time stored with each
result (metrics.compute_ms.total) and the server-side request time.
"""
import json
import os
import statistics
import sys
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SETTINGS = [
    ("recursive, defaults", {"k": 80}),
    ("flat, S=400, k=6", {"k": 6, "method": "flat", "superpixels": 400}),
]


def call(method, path, body=None, ctype=None):
    req = urllib.request.Request(BASE + path, data=body, method=method)
    if ctype:
        req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def upload(path):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{os.path.basename(path)}"'
            f"\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return call("POST", "/api/runs", body, f"multipart/form-data; boundary={boundary}")["data"]["run_id"]


def compute(run_id, params):
    body = call("POST", f"/api/runs/{run_id}/results", json.dumps(params).encode(), "application/json")
    m = body["data"]["metrics"]
    return m["cache"]["graph"], m["compute_ms"]["total"], body["meta"]["request_ms"], m["segments"]


print("| Image | Setting | Miss: engine / request (ms) | Hit: engine / request (ms) | Request speed-up |")
print("|---|---|---|---|---|")
for name in sorted(os.listdir(DATA)):
    run_id = upload(os.path.join(DATA, name))
    for label, params in SETTINGS:
        rows = {"miss": [], "hit": []}
        for _ in range(REPS):
            call("DELETE", "/api/cache")
            for expected in ("miss", "hit"):
                outcome, engine, request, _ = compute(run_id, params)
                assert outcome == expected, (name, label, outcome)
                rows[expected].append((engine, request))
        med = {k: (statistics.median(e for e, _ in v), statistics.median(r for _, r in v)) for k, v in rows.items()}
        print(f"| {name} | {label} | {med['miss'][0]:.0f} / {med['miss'][1]:.0f} | "
              f"{med['hit'][0]:.0f} / {med['hit'][1]:.0f} | {med['miss'][1] / med['hit'][1]:.1f}× |")
