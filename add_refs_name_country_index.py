"""Create idx_refs_name_country on existing train SQLite. Does not rebuild refs."""
import os
import sqlite3
import time

DB = os.path.abspath("data/indexes/s2s3.sqlite")


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def main():
    if not os.path.isfile(DB):
        raise SystemExit(f"missing {DB}")
    print(f"db={DB} size_mb={os.path.getsize(DB)/1e6:.1f} rss={rss_mb():.1f}", flush=True)
    conn = sqlite3.connect(DB)
    n = conn.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
    print(f"refs={n}", flush=True)
    if n != 10320219:
        raise SystemExit(f"refusing: unexpected refs count {n}")
    print("CREATE INDEX IF NOT EXISTS idx_refs_name_country ON refs(name, country)...", flush=True)
    t0 = time.time()
    rss0 = rss_mb()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_name_country ON refs(name, country)")
    conn.commit()
    elapsed = time.time() - t0
    rss1 = rss_mb()
    print(
        f"index_done elapsed_s={elapsed:.1f} rss_before={rss0:.1f} rss_after={rss1:.1f} "
        f"size_mb={os.path.getsize(DB)/1e6:.1f}",
        flush=True,
    )
    plan = conn.execute(
        "EXPLAIN QUERY PLAN SELECT row_id FROM refs WHERE name = ? AND country = ? LIMIT ?",
        ("acme corp", "us", 40),
    ).fetchall()
    print("EXPLAIN QUERY PLAN:", plan, flush=True)
    plan_txt = " ".join(str(x) for row in plan for x in row).lower()
    if "idx_refs_name_country" not in plan_txt:
        raise SystemExit(f"index not used: {plan}")
    if "scan refs" in plan_txt and "idx_refs_name_country" not in plan_txt:
        raise SystemExit(f"still scanning refs: {plan}")
    print("plan_ok: idx_refs_name_country", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
