"""Build FULL test S2+S3 SQLite index with RSS logging. No S1 matching."""
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.blocking.sqlite_index import SqliteReferenceIndex

    db = os.path.abspath("data/indexes/test_s2s3.sqlite")
    os.makedirs(os.path.dirname(db), exist_ok=True)
    t0 = time.time()
    print(f"RSS before: {rss_mb():.1f} MB")
    idx = SqliteReferenceIndex.build_from_files(
        ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"],
        db,
        chunksize=10000,
    )
    elapsed = time.time() - t0
    print(f"RSS after: {rss_mb():.1f} MB")
    print(f"rows={idx.n} elapsed_s={elapsed:.1f} db={db} size_mb={os.path.getsize(db)/1e6:.1f}")
    r, n, hit = idx.retrieve("acme corp", "main street", "us")
    print(f"smoke retrieve len={len(r)} raw={n} cap={hit}")


if __name__ == "__main__":
    main()
