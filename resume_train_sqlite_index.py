"""Resume train S2+S3 SQLite ingest. Never deletes s2s3.sqlite or test_s2s3.sqlite."""
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

    db = os.path.abspath("data/indexes/s2s3.sqlite")
    test_db = os.path.abspath("data/indexes/test_s2s3.sqlite")
    if not os.path.isfile(test_db):
        raise SystemExit(f"refusing to continue: missing {test_db}")
    expected = 5034616 + 5285603
    print(f"RSS before: {rss_mb():.1f} MB")
    t0 = time.time()
    idx = SqliteReferenceIndex.resume_from_files(
        ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"],
        db,
        chunksize=10000,
        expected_n=expected,
    )
    elapsed = time.time() - t0
    print(f"RSS after: {rss_mb():.1f} MB")
    print(
        f"rows={idx.n} expected={expected} elapsed_s={elapsed:.1f} db={db} "
        f"size_mb={os.path.getsize(db)/1e6:.1f} test_db_ok={os.path.isfile(test_db)}"
    )
    r, n, hit = idx.retrieve("acme corp", "main street", "us")
    print(f"smoke retrieve len={len(r)} raw={n} cap={hit}")


if __name__ == "__main__":
    main()
