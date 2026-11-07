"""End-to-end test of the complete application under docker compose.

Builds and starts the stack (UI + API + PostgreSQL) under a separate compose project with a
fresh volume, drives the UI in headless Chromium (Playwright image), restarts the API and
then the whole stack, and checks persistence and cache behaviour along the way.

    python3 tests/e2e/run_e2e.py            (needs docker; uses ports 18080 and 15433)
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJECT = "seg-e2e"
# Separate ports so a stack already running on the defaults is left alone.
PORTS = {"SEG_PORT": "18080", "SEG_DB_PORT": "15433"}
os.environ.update(PORTS)
BASE = f"http://127.0.0.1:{PORTS['SEG_PORT']}"
PLAYWRIGHT = "seg-e2e-browser"  # built from tests/e2e/Dockerfile
IMAGE = "/repo/data/downtown_toronto.jpeg"
failures = []


def check(cond, what):
    print(("  PASS " if cond else "  FAIL ") + what)
    if not cond:
        failures.append(what)


def compose(*args):
    subprocess.run(["docker", "compose", "-p", PROJECT, *args], cwd=REPO, check=True, capture_output=True)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())["data"]


def wait_healthy(timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if get("/api/health")["status"] == "ok":
                return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError("API did not become healthy")


def browser(*args):
    p = subprocess.run(["docker", "run", "--rm", "--network", "host", "--ipc", "host", "-v", f"{REPO}:/repo:ro",
                        PLAYWRIGHT, "python", "/repo/tests/e2e/ui_flow.py", BASE, IMAGE, *args],
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        print(p.stdout, p.stderr, sep="\n")
        raise RuntimeError("browser flow failed")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    subprocess.run(["docker", "build", "-q", "-t", PLAYWRIGHT, os.path.join(REPO, "tests", "e2e")],
                   check=True, capture_output=True)
    compose("down", "-v")
    print("building and starting the stack...")
    compose("up", "-d", "--build")
    try:
        wait_healthy()
        index = urllib.request.urlopen(BASE + "/", timeout=10).read().decode()
        check('<div id="root">' in index, "UI is served by seg_server at /")

        print("== upload -> segment -> change k -> change graph -> stale -> save -> history")
        w = browser("workflow")
        check(w["page_errors"] == [], f"no page errors {w['page_errors']}")
        # Recursive NJW: k is the maximum segment count, so check consistency, not equality with k.
        def consistent(r, k):
            return r["method"] == "recursive" and r["k"] == k and r["segments"] == r["stored_segments"] <= k
        check(w["first"]["cache"] == "miss" and consistent(w["first"], 6), f"first: {w['first']}")
        check(w["k_changed"]["cache"] == "hit" and consistent(w["k_changed"], 8),
              f"only k changed -> graph hit: {w['k_changed']}")
        check(w["graph_changed"]["cache"] == "miss", f"superpixels changed -> miss: {w['graph_changed']}")
        ui = w["ui"]
        check(not ui["mentions_cache"], "UI shows no cache information")
        check(ui["removed_inputs"] == 0, "no method / seed / colour sigma / spatial scale inputs")
        check(ui["sliders"] == ["range-blur_sigma", "range-k", "range-max_dim", "range-ncut_threshold",
                                "range-superpixels"], f"one slider per input: {ui['sliders']}")
        check(all(re.fullmatch(r"[0-9a-f]{8}", i) for i in ui["ids_shown"]) and ui["ids_shown"],
              f"only result ids shown: {ui['ids_shown']}")
        check(not re.search(r"[0-9a-f]{8}", ui["run_info"]), f"no run id shown: {ui['run_info']!r}")
        check(w["overlay_loaded"], "result image renders")
        s = w["stale"]
        check(s["held_released"], "older request was held back past the newer one")
        check(s["k_field"] == "5", f"newest result (k=5) stays displayed: {s}")
        check(s["displayed_id"] == s["newest_response_id"], "displayed id is the newest response's id")
        check(w["save"]["put_url"].endswith(f"/api/experiments/{w['save']['displayed']}"),
              f"Save sent the displayed id: {w['save']}")
        check(w["viewing"]["image_loaded"] and w["viewing"]["current_unchanged"],
              f"saved experiment opens separately from the current result: {w['viewing']}")

        saved_id = w["save"]["displayed"]
        experiment = get(f"/api/experiments/{saved_id}")
        check(experiment["params"]["k"] == 5 and experiment["metrics"]["segments"] == w["viewing"]["segments"],
              "API has the saved k=5 result")
        saved_segments = experiment["metrics"]["segments"]

        print("== restart the API container")
        compose("restart", "api")
        wait_healthy()
        check(get("/api/cache")["entries"] == 0, "cache starts empty after restart")
        a = browser("after_restart", saved_id)
        check(a["page_errors"] == [], f"no page errors {a['page_errors']}")
        check(a["viewing_image_loaded"] and a["viewing_segments"] == saved_segments, f"saved history renders after restart: {a}")
        check(a["recompute"]["cache"] == "miss" and a["recompute"]["segments"] == saved_segments,
              f"cold cache after restart -> miss: {a['recompute']}")
        check(get(f"/api/experiments/{saved_id}") == experiment, "saved experiment unchanged by the restart")

        print("== restart the whole stack (volume kept)")
        compose("down")
        compose("up", "-d")
        wait_healthy()
        check(get(f"/api/experiments/{saved_id}") == experiment, "saved experiment survives a full stack restart")

        print("== clear history")
        c = browser("clear")
        check(c["page_errors"] == [], f"no page errors {c['page_errors']}")
        check(c["rows_before"] >= 2 and c["notice"] == f"Deleted {c['rows_before']} saved experiments.",
              f"Clear history deletes every saved experiment: {c}")
        check(c["history_headers"] == ["Saved", "Result", "Superpixels", "Segments", "Engine time"],
              f"history has no method / k columns: {c['history_headers']}")
        check(get("/api/experiments")["count"] == 0, "API log is empty")
        check(c["current_cleared"], "the deleted (saved) current result is no longer shown")
        check(c["after_clear"]["cache"] == "hit", f"graph stays cached after clearing: {c['after_clear']}")
    finally:
        compose("down", "-v")
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
