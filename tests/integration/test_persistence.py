"""Integration tests for the HTTP service with PostgreSQL persistence.

Starts a throwaway PostgreSQL container and the locally built seg_server, then talks HTTP.
Requires docker and a build containing seg_server and segment.

    python3 tests/integration/test_persistence.py -v

Environment: SEG_BUILD_DIR (default: <repo>/build), SEG_PG_IMAGE (default: postgres:16-alpine).
"""
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUILD = os.environ.get("SEG_BUILD_DIR", os.path.join(REPO, "build"))
SERVER_BIN = os.path.join(BUILD, "seg_server")
SEGMENT_BIN = os.path.join(BUILD, "segment")
PG_IMAGE = os.environ.get("SEG_PG_IMAGE", "postgres:16-alpine")
IMAGE = os.path.join(REPO, "data", "downtown_toronto.jpeg")
SLOW_IMAGE = os.path.join(REPO, "data", "coffee_turquoise_cup.jpg")  # default params: ~1.5 s
FAST = {"k": 6, "method": "flat", "superpixels": 400}
FAST_CLI = ["--method", "flat", "--superpixels", "400", "--k", "6"]
IMAGE_NAMES = ("original", "superpixels", "segmentation", "boundaries")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(predicate, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {what}")


class Postgres:
    """A named container whose data survives docker stop/start (removed at the end)."""

    def __init__(self):
        self.name = f"seg-it-pg-{os.getpid()}"
        self.port = free_port()
        self.url = f"postgresql://seg:seg@127.0.0.1:{self.port}/seg"

    def create(self):
        subprocess.run(["docker", "run", "-d", "--name", self.name, "-e", "POSTGRES_USER=seg",
                        "-e", "POSTGRES_PASSWORD=seg", "-e", "POSTGRES_DB=seg",
                        "-p", f"127.0.0.1:{self.port}:5432", PG_IMAGE],
                       check=True, capture_output=True)
        self.wait_ready()

    def wait_ready(self):
        # -h forces TCP: the temporary server of first-time init only listens on a socket.
        def ready():
            return subprocess.run(["docker", "exec", self.name, "pg_isready", "-h", "127.0.0.1", "-U", "seg",
                                   "-d", "seg"], capture_output=True).returncode == 0
        wait_until(ready, 60, "PostgreSQL")

    def stop(self):
        subprocess.run(["docker", "stop", "-t", "10", self.name], check=True, capture_output=True)

    def kill(self):
        subprocess.run(["docker", "kill", self.name], check=True, capture_output=True)

    def start(self):
        subprocess.run(["docker", "start", self.name], check=True, capture_output=True)
        self.wait_ready()

    def remove(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

    def _psql(self, sql):
        return subprocess.run(["docker", "exec", self.name, "psql", "-U", "seg", "-d", "seg", "-v",
                               "ON_ERROR_STOP=1", "-tAc", sql], capture_output=True, text=True)

    def sql(self, sql):
        p = self._psql(sql)
        if p.returncode != 0:
            raise RuntimeError(p.stderr)
        return p.stdout.strip()

    def sql_error(self, sql):
        """Runs a statement that must fail; returns the error text."""
        p = self._psql(sql)
        if p.returncode == 0:
            raise AssertionError(f"statement unexpectedly succeeded: {sql}")
        return p.stderr

    def terminate_server_connections(self):
        return int(self.sql("SELECT count(pg_terminate_backend(pid)) FROM pg_stat_activity "
                            "WHERE application_name = 'seg_server'"))


class Server:
    def __init__(self, db_url, *extra_args):
        self.db_url = db_url
        self.extra_args = list(extra_args)
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc = None
        self.log = tempfile.NamedTemporaryFile(prefix="seg_server_", suffix=".log", delete=False)

    def start(self):
        self.proc = subprocess.Popen([SERVER_BIN, "--host", "127.0.0.1", "--port", str(self.port),
                                      "--db", self.db_url, "--db-wait", "20"] + self.extra_args,
                                     stdout=self.log, stderr=subprocess.STDOUT)

        def healthy():
            if self.proc.poll() is not None:
                raise RuntimeError(f"seg_server exited with {self.proc.returncode}; log: {self.log.name}")
            try:
                return http("GET", "/api/health", timeout=2, base=self.base)[0] == 200
            except OSError:
                return False
        wait_until(healthy, 30, "seg_server")

    def stop(self):
        self.proc.send_signal(signal.SIGTERM)
        return self.proc.wait(timeout=15)


PG = Postgres()
SERVER = Server(PG.url)


def http(method, path, body=None, ctype=None, timeout=120, base=None):
    req = urllib.request.Request((base or SERVER.base) + path, data=body, method=method)
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def api(method, path, payload=None, base=None):
    body = json.dumps(payload).encode() if payload is not None else None
    status, raw = http(method, path, body, "application/json" if body is not None else None, base=base)
    return status, json.loads(raw)


def upload(path):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{os.path.basename(path)}"'
            f"\r\nContent-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    status, raw = http("POST", "/api/runs", body, f"multipart/form-data; boundary={boundary}")
    return status, json.loads(raw)


def without(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


def engine_output(metrics):
    """Metrics that depend only on the inputs: everything except timings and cache status."""
    return without(metrics, "compute_ms", "cache")


def setUpModule():
    for binary in (SERVER_BIN, SEGMENT_BIN):
        if not os.access(binary, os.X_OK):
            raise unittest.SkipTest(f"{binary} not built")
    PG.create()
    SERVER.start()


def tearDownModule():
    if SERVER.proc and SERVER.proc.poll() is None:
        SERVER.stop()
    PG.remove()


class PersistenceTest(unittest.TestCase):
    def new_run(self, path=IMAGE):
        status, body = upload(path)
        self.assertEqual(status, 201, body)
        return body["data"]["run_id"]

    def compute(self, run_id, params, base=None):
        status, body = api("POST", f"/api/runs/{run_id}/results", params, base=base)
        self.assertEqual(status, 201, body)
        # HTTP latency of this request vs the engine time stored with the result.
        self.assertGreaterEqual(body["meta"]["request_ms"], body["data"]["metrics"]["compute_ms"]["total"])
        return body["data"]

    def save(self, result_id):
        status, body = api("PUT", f"/api/experiments/{result_id}")
        return status, body.get("data", body)

    def saved_for_run(self, run_id):
        status, body = api("GET", f"/api/experiments?run_id={run_id}")
        self.assertEqual(status, 200, body)
        return body["data"]["experiments"]

    def count(self, table, where="TRUE"):
        return int(PG.sql(f"SELECT count(*) FROM {table} WHERE {where}"))

    def images(self, result_id):
        out = {}
        for name in IMAGE_NAMES:
            status, raw = http("GET", f"/api/results/{result_id}/images/{name}")
            self.assertEqual(status, 200, raw[:200])
            out[name] = raw
        return out

    # 1 ---------------------------------------------------------------------------------------
    def test_1_create_save_retrieve(self):
        run_id = self.new_run()
        result = self.compute(run_id, FAST)
        rid = result["result_id"]
        self.assertIsNone(result["saved_at"])
        self.assertEqual(result["params"]["k"], 6)

        status, body = api("GET", f"/api/experiments/{rid}")
        self.assertEqual((status, body["error"]["code"]), (404, "experiment_not_found"))
        status, body = api("GET", f"/api/results/{rid}")
        self.assertEqual((status, body["data"]), (200, result))

        status, saved = self.save(rid)
        self.assertEqual(status, 201, saved)
        self.assertIsNotNone(saved["saved_at"])
        self.assertEqual(without(saved, "saved_at"), without(result, "saved_at"))

        status, body = api("GET", f"/api/experiments/{rid}")
        self.assertEqual((status, body["data"]), (200, saved))
        status, body = api("GET", "/api/experiments")
        self.assertIn(rid, [e["result_id"] for e in body["data"]["experiments"]])

        # The stored metrics are exactly what the engine returned at creation.
        stored = json.loads(PG.sql(f"SELECT metrics::text FROM results WHERE id = '{rid}'"))
        self.assertEqual(stored, result["metrics"])

    # 2 ---------------------------------------------------------------------------------------
    def test_2_save_twice_is_idempotent(self):
        run_id = self.new_run()
        rid = self.compute(run_id, FAST)["result_id"]
        s1, first = self.save(rid)
        s2, second = self.save(rid)
        self.assertEqual((s1, s2), (201, 200))
        self.assertEqual(first, second)  # same saved_at, params and metrics
        self.assertEqual(PG.sql(f"SELECT iso_utc(saved_at) FROM results WHERE id = '{rid}'"), first["saved_at"])
        self.assertEqual([e["result_id"] for e in self.saved_for_run(run_id)], [rid])

        # Concurrent saves of one fresh result: exactly one performs the save.
        rid2 = self.compute(run_id, FAST)["result_id"]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            replies = list(pool.map(lambda _: self.save(rid2), range(8)))
        self.assertEqual(sorted(s for s, _ in replies), [200] * 7 + [201])
        self.assertEqual(len({d["saved_at"] for _, d in replies}), 1)
        self.assertEqual(sorted(e["result_id"] for e in self.saved_for_run(run_id)), sorted([rid, rid2]))

        status, body = self.save("00000000-0000-4000-8000-000000000000")
        self.assertEqual((status, body["error"]["code"]), (404, "result_not_found"))

    # 3 ---------------------------------------------------------------------------------------
    def test_3_several_saved_results_for_one_run(self):
        run_id = self.new_run()
        results = [self.compute(run_id, dict(FAST, k=k)) for k in (4, 6, 8)]
        ids = [r["result_id"] for r in results]
        self.assertEqual(len(set(ids)), 3)
        saved = {}
        for rid in ids:
            status, body = self.save(rid)
            self.assertEqual(status, 201)
            saved[rid] = body

        experiments = self.saved_for_run(run_id)
        self.assertEqual([e["result_id"] for e in experiments], ids[::-1])  # most recently saved first
        for r in results:
            e = next(e for e in experiments if e["result_id"] == r["result_id"])
            self.assertEqual(e, saved[r["result_id"]])
            self.assertEqual(without(e, "saved_at"), without(r, "saved_at"))
            # Each experiment reports its own engine time, never a total over the run.
            self.assertEqual(e["metrics"]["compute_ms"], r["metrics"]["compute_ms"])
        totals = [r["metrics"]["compute_ms"]["total"] for r in results]
        for e in experiments:
            self.assertNotAlmostEqual(e["metrics"]["compute_ms"]["total"], sum(totals))

        status, body = api("GET", f"/api/runs/{run_id}")
        self.assertEqual(body["data"]["result_ids"], ids)  # creation order

    # 4 ---------------------------------------------------------------------------------------
    def test_4_save_older_result_after_computing_newer(self):
        run_id = self.new_run()
        older = self.compute(run_id, dict(FAST, k=4))
        newer = self.compute(run_id, dict(FAST, k=8))

        status, saved = self.save(older["result_id"])
        self.assertEqual(status, 201)
        self.assertEqual(saved["result_id"], older["result_id"])
        self.assertEqual(saved["params"]["k"], 4)
        self.assertEqual(saved["metrics"], older["metrics"])
        self.assertNotEqual(saved["metrics"], newer["metrics"])

        status, body = api("GET", f"/api/experiments/{newer['result_id']}")
        self.assertEqual(status, 404)
        self.assertEqual([e["result_id"] for e in self.saved_for_run(run_id)], [older["result_id"]])

        status, _ = self.save(newer["result_id"])
        self.assertEqual(status, 201)
        experiments = self.saved_for_run(run_id)
        self.assertEqual([e["result_id"] for e in experiments], [newer["result_id"], older["result_id"]])
        self.assertEqual(experiments[1], saved)

    # 5 ---------------------------------------------------------------------------------------
    def test_5_restart_keeps_saved_history(self):
        run_id = self.new_run()
        rid = self.compute(run_id, FAST)["result_id"]
        self.assertEqual(self.save(rid)[0], 201)
        status, before = api("GET", f"/api/experiments/{rid}")
        images_before = self.images(rid)

        self.assertEqual(SERVER.stop(), 0)
        SERVER.start()

        status, after = api("GET", f"/api/experiments/{rid}")
        self.assertEqual(status, 200)
        self.assertEqual(after["data"], before["data"])
        status, body = api("GET", "/api/experiments")
        self.assertIn(rid, [e["result_id"] for e in body["data"]["experiments"]])
        self.assertEqual(self.images(rid), images_before)

        # Images re-rendered from storage match the standalone CLI byte for byte.
        with tempfile.TemporaryDirectory() as out:
            subprocess.run([SEGMENT_BIN, "run", IMAGE, out] + FAST_CLI, check=True, capture_output=True)
            for name in IMAGE_NAMES:
                with open(os.path.join(out, f"{name}.png"), "rb") as f:
                    self.assertEqual(images_before[name], f.read(), name)

        # The upload is persisted too, so old runs can still be segmented.
        self.compute(run_id, dict(FAST, k=5))

    # 6: simulated database failures ---------------------------------------------------------
    def test_6a_terminated_connections_reconnect(self):
        run_id = self.new_run()
        self.compute(run_id, FAST)
        self.assertGreater(PG.terminate_server_connections(), 0)
        self.assertEqual(api("GET", "/api/health")[0], 200)
        rid = self.compute(run_id, FAST)["result_id"]
        PG.terminate_server_connections()
        self.assertEqual(self.save(rid)[0], 201)

    def test_6b_connection_lost_during_compute(self):
        run_id = self.new_run(SLOW_IMAGE)
        results_before = self.count("results")
        reply = {}
        worker = threading.Thread(target=lambda: reply.update(r=api("POST", f"/api/runs/{run_id}/results",
                                                                    {"k": 80})))
        worker.start()
        time.sleep(0.4)  # the run image has been read; the engine is running
        self.assertGreater(PG.terminate_server_connections(), 0)
        self.assertTrue(worker.is_alive(), "compute finished before the connection was killed")
        worker.join()
        status, body = reply["r"]
        self.assertEqual(status, 201, body)  # the insert reconnected and retried once
        self.assertEqual(self.count("results"), results_before + 1)
        self.assertEqual(self.count("results", f"id = '{body['data']['result_id']}'"), 1)

    def test_6c_database_down_then_back(self):
        run_id = self.new_run()
        rid = self.compute(run_id, FAST)["result_id"]
        counts = (self.count("runs"), self.count("results"), self.count("results", "saved_at IS NOT NULL"))

        PG.stop()
        try:
            for label, (status, body) in {
                "health": api("GET", "/api/health"),
                "upload": upload(IMAGE),
                "compute": api("POST", f"/api/runs/{run_id}/results", FAST),
                "save": api("PUT", f"/api/experiments/{rid}"),
                "get result": api("GET", f"/api/results/{rid}"),
                "list": api("GET", "/api/experiments"),
            }.items():
                self.assertEqual((status, body["error"]["code"]), (503, "database_unavailable"), label)
            self.assertIsNone(SERVER.proc.poll(), "server must keep running")
        finally:
            PG.start()

        wait_until(lambda: api("GET", "/api/health")[0] == 200, 20, "health after DB restart")
        # Nothing was half-written while the database was down.
        self.assertEqual((self.count("runs"), self.count("results"), self.count("results", "saved_at IS NOT NULL")),
                         counts)
        self.assertEqual(PG.sql(f"SELECT saved_at IS NULL FROM results WHERE id = '{rid}'"), "t")
        self.assertEqual(self.save(rid)[0], 201)

    def test_6d_database_crash_during_compute(self):
        run_id = self.new_run(SLOW_IMAGE)
        results_before = self.count("results")
        reply = {}
        worker = threading.Thread(target=lambda: reply.update(r=api("POST", f"/api/runs/{run_id}/results",
                                                                    {"k": 80})))
        worker.start()
        time.sleep(0.4)
        self.assertTrue(worker.is_alive(), "compute finished before the database was killed")
        PG.kill()  # SIGKILL: no clean shutdown; PostgreSQL recovers from its WAL on start
        try:
            worker.join()
            status, body = reply["r"]
            self.assertEqual((status, body["error"]["code"]), (503, "database_unavailable"))
        finally:
            PG.start()
        wait_until(lambda: api("GET", "/api/health")[0] == 200, 20, "health after DB crash")
        self.assertEqual(self.count("results"), results_before)  # the failed result was not stored
        self.assertEqual(self.count("runs", f"id = '{run_id}'"), 1)  # committed data survived the crash
        self.compute(run_id, FAST)

    def test_6e_startup_without_database_fails(self):
        p = subprocess.run([SERVER_BIN, "--port", str(free_port()), "--db",
                            f"postgresql://seg:seg@127.0.0.1:{free_port()}/seg", "--db-wait", "2"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 1)
        self.assertIn("cannot connect to database", p.stderr)

    # 7 ---------------------------------------------------------------------------------------
    def test_7_database_enforces_immutability(self):
        run_id = self.new_run()
        rid = self.compute(run_id, FAST)["result_id"]
        self.assertEqual(self.save(rid)[0], 201)
        where = f"WHERE id = '{rid}'"
        unsaved = self.compute(run_id, FAST)["result_id"]
        for sql, error in ((f"UPDATE results SET metrics = '{{}}' {where}", "immutable"),
                           (f"UPDATE results SET params = '{{}}' {where}", "immutable"),
                           (f"UPDATE results SET saved_at = now() {where}", "immutable"),
                           (f"UPDATE results SET saved_at = NULL {where}", "immutable"),
                           (f"DELETE FROM results WHERE id = '{unsaved}'", "only saved results can be deleted"),
                           (f"UPDATE runs SET filename = 'x' WHERE id = '{run_id}'", "cannot be updated or deleted"),
                           (f"DELETE FROM runs WHERE id = '{run_id}'", "cannot be updated or deleted")):
            self.assertIn(error, PG.sql_error(sql), sql)


    # 8: graph cache --------------------------------------------------------------------------
    def clear_cache(self, base=None):
        status, body = api("DELETE", "/api/cache", base=base)
        self.assertEqual((status, body["data"]["entries"]), (200, 0))

    def test_8a_only_clustering_changes_reuse_the_graph(self):
        self.clear_cache()
        run_id = self.new_run()
        first = self.compute(run_id, FAST)
        self.assertEqual(first["metrics"]["cache"]["graph"], "miss")
        built = first["metrics"]["cache"]["graph_build_ms"]
        self.assertGreaterEqual(first["metrics"]["compute_ms"]["total"], built)  # a miss pays for the graph

        for change in ({"k": 8}, {"method": "recursive", "k": 20},
                       {"method": "recursive", "k": 20, "ncut_threshold": 0.1}, {}):
            r = self.compute(run_id, dict(FAST, **change))
            cache, c = r["metrics"]["cache"], r["metrics"]["compute_ms"]
            self.assertEqual(cache["graph"], "hit", change)
            self.assertEqual((c["preprocess"], c["slic"], c["graph"]), (0, 0, 0), change)
            self.assertEqual(cache["graph_build_ms"], built, change)  # the reused graph's own build cost
            self.assertLess(c["total"], first["metrics"]["compute_ms"]["total"], change)

        for change in ({"superpixels": 500}, {"blur_sigma": 2}, {"max_dim": 300}):
            r = self.compute(run_id, dict(FAST, **change))
            self.assertEqual(r["metrics"]["cache"]["graph"], "miss", change)
        # Same parameters on another image never share a graph.
        self.assertEqual(self.compute(self.new_run(), FAST)["metrics"]["cache"]["graph"], "miss")

    def test_8b_cached_results_equal_uncached(self):
        self.clear_cache()
        run_id = self.new_run()
        params = {"k": 30, "superpixels": 1500}  # recursive
        self.compute(run_id, dict(params, k=10))  # builds the graph
        cached = self.compute(run_id, params)
        repeated = self.compute(run_id, params)
        self.clear_cache()  # invalidation: the next request must rebuild
        cold = self.compute(run_id, params)
        self.assertEqual([r["metrics"]["cache"]["graph"] for r in (cached, repeated, cold)], ["hit", "hit", "miss"])
        self.assertEqual(len({cached["result_id"], repeated["result_id"], cold["result_id"]}), 3)
        for r in (cached, repeated):
            self.assertEqual(engine_output(r["metrics"]), engine_output(cold["metrics"]))
            self.assertEqual(r["params"], cold["params"])
            self.assertEqual(self.images(r["result_id"]), self.images(cold["result_id"]))

    def test_8c_concurrent_requests_build_the_graph_once(self):
        self.clear_cache()
        run_id = self.new_run(SLOW_IMAGE)
        before = api("GET", "/api/cache")[1]["data"]
        ks = [4, 5, 6, 7, 8, 9]
        with concurrent.futures.ThreadPoolExecutor(len(ks)) as pool:
            replies = list(pool.map(lambda k: api("POST", f"/api/runs/{run_id}/results", dict(FAST, k=k)), ks))
        self.assertEqual([s for s, _ in replies], [201] * len(ks))
        results = [b["data"] for _, b in replies]
        outcomes = sorted(r["metrics"]["cache"]["graph"] for r in results)
        self.assertEqual(outcomes.count("miss"), 1, outcomes)
        self.assertTrue(set(outcomes) <= {"miss", "hit", "shared"})
        after = api("GET", "/api/cache")[1]["data"]
        self.assertEqual(after["misses"] - before["misses"], 1)
        self.assertEqual(self.count("results", f"run_id = '{run_id}'"), len(ks))
        # Each concurrent result equals a sequential cold computation with its own k.
        self.clear_cache()
        for r in results[:2]:
            cold = self.compute(run_id, dict(FAST, k=r["params"]["k"]))
            self.assertEqual(engine_output(r["metrics"]), engine_output(cold["metrics"]))

    def test_8d_fixed_parameters_are_rejected(self):
        run_id = self.new_run()
        for field, value in (("seed", 7), ("sigma_color", 8), ("spatial_scale", 0)):
            status, body = api("POST", f"/api/runs/{run_id}/results", dict(FAST, **{field: value}))
            self.assertEqual((status, body["error"]["code"], body["error"]["details"]["field"]),
                             (400, "invalid_parameter", field))
        params = self.compute(run_id, FAST)["params"]  # the fixed values are still recorded
        self.assertEqual((params["seed"], params["sigma_color"], params["spatial_scale"]), (42, 5.0, 2.0))

    def lru_scenario(self, *server_args):
        small = Server(PG.url, *server_args)
        small.start()
        try:
            base = small.base
            run_id = self.new_run()
            for s in (400, 500, 600):
                r = self.compute(run_id, dict(FAST, superpixels=s), base=base)
                self.assertEqual(r["metrics"]["cache"]["graph"], "miss")
            stats = api("GET", "/api/cache", base=base)[1]["data"]
            self.assertEqual(stats["entries"], 2)
            self.assertEqual(stats["evictions"], 1)
            self.assertLessEqual(stats["bytes"], stats["capacity_bytes"])
            self.assertEqual(self.compute(run_id, dict(FAST, superpixels=600), base=base)["metrics"]["cache"]["graph"],
                             "hit")
            self.assertEqual(self.compute(run_id, dict(FAST, superpixels=400), base=base)["metrics"]["cache"]["graph"],
                             "miss")  # least recently used, evicted first
        finally:
            self.assertEqual(small.stop(), 0)

    def test_8e_holds_at_most_n_graphs(self):
        self.lru_scenario("--cache-graphs", "2", "--cache-mb", "1024")

    def test_8f_byte_cap_also_bounds_the_cache(self):
        # A ~1.4 MB graph per request (387x516 image + labels): 3 MB holds two.
        self.lru_scenario("--cache-graphs", "100", "--cache-mb", "3")

    # 9 ---------------------------------------------------------------------------------------
    def test_9_clear_history_deletes_saved_results_only(self):
        run_id = self.new_run()
        saved = [self.compute(run_id, dict(FAST, k=k))["result_id"] for k in (4, 5)]
        unsaved = self.compute(run_id, dict(FAST, k=6))["result_id"]
        for rid in saved:
            self.assertEqual(self.save(rid)[0], 201)
        total_saved = self.count("results", "saved_at IS NOT NULL")
        runs = self.count("runs")

        status, body = api("DELETE", "/api/experiments")
        self.assertEqual((status, body["data"]["deleted"]), (200, total_saved))
        self.assertEqual(api("GET", "/api/experiments")[1]["data"]["count"], 0)
        for rid in saved:
            self.assertEqual(api("GET", f"/api/results/{rid}")[0], 404)
            self.assertEqual(self.save(rid)[0], 404)
        self.assertEqual(api("GET", f"/api/results/{unsaved}")[0], 200)  # unsaved results stay
        self.assertEqual(api("GET", f"/api/runs/{run_id}")[1]["data"]["result_ids"], [unsaved])
        self.assertEqual(self.count("runs"), runs)
        self.assertEqual(api("DELETE", "/api/experiments")[1]["data"]["deleted"], 0)  # idempotent
        # The graph cache is independent of the log: the same graph is still cached.
        self.assertEqual(self.compute(run_id, dict(FAST, k=7))["metrics"]["cache"]["graph"], "hit")


if __name__ == "__main__":
    unittest.main()
