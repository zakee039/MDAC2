"""Run with the isolated interpreter. All mutable data lives in a temp directory."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time


def consumer():
    from huey.consumer import Consumer
    from probe_tasks import huey
    runner = Consumer(huey, workers=1, worker_type="thread", periodic=False,
                      initial_delay=.01, max_delay=.05, shutdown_timeout=2)
    def stop():
        sys.stdin.readline()
        runner.stop(graceful=True)
    threading.Thread(target=stop, daemon=True).start()
    runner.run()


def main():
    with tempfile.TemporaryDirectory(prefix="mdac-huey-") as directory:
        os.environ["MDAC_HUEY_PROBE"] = directory
        from probe_tasks import db, huey, work
        with db() as connection:
            connection.executescript("""
                CREATE TABLE jobs(key TEXT PRIMARY KEY, state TEXT);
                CREATE TABLE outbox(key TEXT PRIMARY KEY, sent INTEGER DEFAULT 0);
                CREATE TABLE events(kind TEXT, key TEXT);
                CREATE TABLE releases(key TEXT PRIMARY KEY);
            """)
        children = []
        def start():
            child = subprocess.Popen([sys.executable, __file__, "consumer"], stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            children.append(child)
            return child
        def stop(child):
            if child.poll() is None:
                child.communicate(b"stop\n", timeout=5)
            assert child.returncode == 0, child.returncode
        def rows(sql, args=()):
            with db() as connection:
                return connection.execute(sql, args).fetchall()
        def wait(predicate):
            deadline = time.monotonic() + 6
            while not predicate():
                assert time.monotonic() < deadline, "probe deadline exceeded"
                time.sleep(.01)
        def create(key):
            with db() as connection:
                connection.execute("INSERT INTO jobs VALUES (?, 'queued')", (key,))
                connection.execute("INSERT INTO outbox(key) VALUES (?)", (key,))
        def done(key):
            return rows("SELECT state FROM jobs WHERE key=?", (key,)) == [("succeeded",)]
        try:
            for key in ("low", "high", "retry", "duplicate", "revoked"):
                create(key)
            work("low", priority=1)
            work("high", priority=10)
            work("retry", fail_first=True)
            # Simulate dispatcher dying after enqueue, before outbox ACK.
            work("duplicate")
            work("duplicate")
            revoked = work("revoked")
            revoked.revoke()
            child = start()
            wait(lambda: all(done(key) for key in ("low", "high", "retry", "duplicate")))
            wait(lambda: huey.pending_count() == 0)
            stop(child)
            effects = rows("SELECT key FROM events WHERE kind='effect' ORDER BY rowid")
            assert effects[:2] == [("high",), ("low",)], effects
            assert rows("SELECT COUNT(*) FROM events WHERE key='retry' AND kind='attempt'") == [(2,)]
            assert rows("SELECT COUNT(*) FROM events WHERE key='duplicate' AND kind='effect'") == [(1,)]
            assert not done("revoked")
            # revoke cannot cancel an already-running function.
            create("running")
            running = work("running", hold=True)
            child = start()
            wait(lambda: rows("SELECT 1 FROM events WHERE key='running' AND kind='attempt'"))
            running.revoke()
            with db() as connection:
                connection.execute("INSERT INTO releases VALUES ('running')")
            wait(lambda: done("running"))
            stop(child)
            # Kill the exact owned consumer while one task is in flight.
            create("crash")
            create("pending")
            work("crash", hold=True)
            work("pending")
            child = start()
            wait(lambda: rows("SELECT 1 FROM events WHERE key='crash' AND kind='attempt'"))
            child.kill()
            child.wait(timeout=5)
            child = start()
            wait(lambda: done("pending"))
            stop(child)
            assert not done("crash")
            assert rows("SELECT COUNT(*) FROM events WHERE key='crash' AND kind='attempt'") == [(1,)]
            print(json.dumps({"huey": "3.4.0", "windows_consumer": "pass",
                "priority": "pass", "retry": "pass", "duplicate_outbox_effects": 1,
                "queued_revoke": "pass", "running_revoke_stops_work": False,
                "pending_survives_restart": True, "inflight_redelivered_after_kill": False}))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
            huey.storage.close()


if __name__ == "__main__":
    consumer() if sys.argv[1:] == ["consumer"] else main()
