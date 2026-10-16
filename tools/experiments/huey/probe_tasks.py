"""Isolated synthetic ledger; not a production backend."""
import os
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import time

from huey import SqliteHuey

root = Path(os.environ["MDAC_HUEY_PROBE"])
huey = SqliteHuey("probe", filename=str(root / "queue.sqlite"))


@contextmanager
def db():
    connection = sqlite3.connect(root / "business.sqlite", timeout=3)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def record(kind, key):
    with db() as connection:
        connection.execute("INSERT INTO events(kind,key) VALUES (?,?)", (kind, key))


@huey.task(retries=1)
def work(key, fail_first=False, hold=False):
    record("attempt", key)
    with db() as connection:
        count = connection.execute("SELECT COUNT(*) FROM events WHERE kind='attempt' AND key=?", (key,)).fetchone()[0]
    if fail_first and count == 1:
        raise RuntimeError("intentional first-attempt failure")
    if hold:
        deadline = time.monotonic() + 10
        while True:
            with db() as connection:
                released = connection.execute("SELECT 1 FROM releases WHERE key=?", (key,)).fetchone()
            if released:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("probe release deadline")
            time.sleep(.01)
    with db() as connection:
        connection.execute("BEGIN IMMEDIATE")
        changed = connection.execute("UPDATE jobs SET state='succeeded' WHERE key=? AND state='queued'", (key,)).rowcount
        if changed:
            connection.execute("INSERT INTO events(kind,key) VALUES ('effect',?)", (key,))
    return key
