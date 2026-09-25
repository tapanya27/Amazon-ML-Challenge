"""
Streaming SQLite index for ~10M S2/S3 rows.

Each CSV chunk is normalized, inserted, committed, and discarded.
No full-dataset Python lists, no in-RAM inverted index, no keep-token sets.
"""

from __future__ import annotations

import gc
import logging
import os
import sqlite3
import time
from collections import Counter
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.preprocessing import preprocess_dataframe

logger = logging.getLogger(__name__)

LEGALISH = {
    "limited", "ltd", "company", "co", "inc", "incorporated", "corp", "corporation",
    "pvt", "private", "llc", "llp", "the", "and", "of", "for", "in", "at", "dba", "group",
}

MIN_DF = 2
MAX_DF = 800
MAX_POSTING = 400
CHUNK = 10000


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def _db_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def _tokens(text: str, min_len: int, stop: Set[str]) -> Set[str]:
    if not text:
        return set()
    out = set()
    for t in str(text).split():
        if t in stop:
            continue
        if len(t) < min_len and not t.isdigit():
            continue
        out.add(t)
    return out


def _log_progress(label: str, rows: int, t0: float, db_path: str) -> None:
    logger.info(
        "%s rows=%d elapsed=%.0fs rss_mb=%.1f sqlite_mb=%.1f",
        label, rows, time.time() - t0, _rss_mb(), _db_mb(db_path),
    )


class SqliteReferenceIndex:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA cache_size = -32000")  # 32MB
        self.conn.execute("PRAGMA temp_store = FILE")
        self.n = int(self.conn.execute("SELECT COUNT(*) FROM refs").fetchone()[0])

    @staticmethod
    def _pragma_write(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = FILE")
        conn.execute("PRAGMA cache_size = -32000")
        conn.execute("PRAGMA mmap_size = 0")

    @staticmethod
    def _insert_chunk(conn: sqlite3.Connection, chunk: pd.DataFrame, row_id: int) -> int:
        p = preprocess_dataframe(chunk, "idx_chunk")
        conn.execute("BEGIN")
        cn_name = Counter()
        cn_addr = Counter()
        tr_name: List[Tuple[str, int]] = []
        tr_addr: List[Tuple[str, int]] = []
        it = zip(
            p["entity_id"].tolist(),
            p["name_normalized"].fillna("").tolist(),
            p["address_normalized"].fillna("").tolist(),
            p["country_normalized"].fillna("").astype(str).str.lower().tolist(),
        )
        ref_batch = []
        for eid, nm, ad, co in it:
            nm = "" if not nm or str(nm) == "nan" else str(nm)
            ad = "" if not ad or str(ad) == "nan" else str(ad)
            co = "" if not co or str(co) == "nan" else str(co)
            ref_batch.append((row_id, str(eid), nm, ad, co))
            ntoks = _tokens(nm, 3, LEGALISH)
            atoks = _tokens(ad, 2, set())
            cn_name.update(ntoks)
            cn_addr.update(atoks)
            for t in ntoks:
                tr_name.append((t, row_id))
            for t in atoks:
                tr_addr.append((t, row_id))
            row_id += 1
        conn.executemany("INSERT INTO refs VALUES (?,?,?,?,?)", ref_batch)
        conn.executemany(
            "INSERT INTO tokc_name(token,c) VALUES(?,?) ON CONFLICT(token) DO UPDATE SET c=c+excluded.c",
            list(cn_name.items()),
        )
        conn.executemany(
            "INSERT INTO tokc_addr(token,c) VALUES(?,?) ON CONFLICT(token) DO UPDATE SET c=c+excluded.c",
            list(cn_addr.items()),
        )
        conn.executemany("INSERT INTO tok_row_name VALUES (?,?)", tr_name)
        conn.executemany("INSERT INTO tok_row_addr VALUES (?,?)", tr_addr)
        conn.commit()
        del p, ref_batch, cn_name, cn_addr, tr_name, tr_addr, it
        gc.collect()
        return row_id

    @staticmethod
    def _stream_tsv_chunks(
        conn: sqlite3.Connection,
        paths: Sequence[str],
        chunksize: int,
        skip_rows: int,
        row_id: int,
        t0: float,
        db_path: str,
    ) -> int:
        skip = skip_rows
        for path in paths:
            logger.info("reading %s skip_remaining=%d row_id=%d", path, skip, row_id)
            for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize):
                n_chunk = len(chunk)
                if skip >= n_chunk:
                    skip -= n_chunk
                    continue
                if skip > 0:
                    chunk = chunk.iloc[skip:]
                    skip = 0
                row_id = SqliteReferenceIndex._insert_chunk(conn, chunk, row_id)
                if row_id % 100000 < chunksize:
                    _log_progress("ingest", row_id, t0, db_path)
        if skip != 0:
            raise RuntimeError(f"skip leftover={skip}: DB has more rows than the source files")
        return row_id

    @staticmethod
    def _finalize_token_indexes(conn: sqlite3.Connection, row_id: int, t0: float, db_path: str) -> None:
        existing = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "inv_name" in existing or "inv_addr" in existing:
            raise RuntimeError("inv_* already present; refusing to rebuild token tables in-place")
        logger.info("SQL filter rare tokens (df %d-%d) — no Python token set", MIN_DF, MAX_DF)
        conn.execute(
            f"""CREATE TABLE inv_name AS
                SELECT r.token, r.row_id
                FROM tok_row_name r
                JOIN tokc_name c ON c.token = r.token
                WHERE c.c BETWEEN {MIN_DF} AND {MAX_DF}"""
        )
        conn.execute("DROP TABLE tok_row_name")
        conn.commit()
        gc.collect()
        _log_progress("after inv_name", row_id, t0, db_path)

        conn.execute(
            f"""CREATE TABLE inv_addr AS
                SELECT r.token, r.row_id
                FROM tok_row_addr r
                JOIN tokc_addr c ON c.token = r.token
                WHERE c.c BETWEEN {MIN_DF} AND {MAX_DF}"""
        )
        conn.execute("DROP TABLE tok_row_addr")
        conn.commit()
        gc.collect()
        _log_progress("after inv_addr", row_id, t0, db_path)

        logger.info("Creating retrieval indexes (token + exact name,country)...")
        conn.execute("CREATE INDEX inv_name_tok ON inv_name(token)")
        _log_progress("after_idx_inv_name_tok", row_id, t0, db_path)
        conn.execute("CREATE INDEX inv_addr_tok ON inv_addr(token)")
        _log_progress("after_idx_inv_addr_tok", row_id, t0, db_path)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_name_country ON refs(name, country)")
        _log_progress("after_idx_refs_name_country", row_id, t0, db_path)
        conn.execute(f"DELETE FROM tokc_name WHERE c < {MIN_DF} OR c > {MAX_DF}")
        conn.execute(f"DELETE FROM tokc_addr WHERE c < {MIN_DF} OR c > {MAX_DF}")
        conn.commit()
        _log_progress("index_ready", row_id, t0, db_path)
        logger.info("SQLite indexes: inv_name_tok, inv_addr_tok, idx_refs_name_country")

    @classmethod
    def build_from_files(
        cls,
        paths: Sequence[str],
        db_path: str,
        chunksize: int = CHUNK,
    ) -> "SqliteReferenceIndex":
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        if os.path.isfile(db_path):
            os.remove(db_path)

        conn = sqlite3.connect(db_path)
        cls._pragma_write(conn)
        conn.execute(
            """CREATE TABLE refs (
                row_id INTEGER PRIMARY KEY,
                entity_id TEXT NOT NULL,
                name TEXT,
                addr TEXT,
                country TEXT
            )"""
        )
        conn.execute("CREATE TABLE tokc_name (token TEXT PRIMARY KEY, c INTEGER)")
        conn.execute("CREATE TABLE tokc_addr (token TEXT PRIMARY KEY, c INTEGER)")
        conn.execute("CREATE TABLE tok_row_name (token TEXT NOT NULL, row_id INTEGER NOT NULL)")
        conn.execute("CREATE TABLE tok_row_addr (token TEXT NOT NULL, row_id INTEGER NOT NULL)")

        logging.getLogger("src.preprocessing.normalize").setLevel(logging.ERROR)
        t0 = time.time()
        logger.info("Streaming ingest from %s chunksize=%d rss_mb=%.1f", list(paths), chunksize, _rss_mb())
        row_id = cls._stream_tsv_chunks(conn, paths, chunksize, skip_rows=0, row_id=0, t0=t0, db_path=db_path)
        logger.info("Pass 1 complete: %d rows rss_mb=%.1f", row_id, _rss_mb())
        cls._finalize_token_indexes(conn, row_id, t0, db_path)
        conn.close()
        return cls(db_path)

    @classmethod
    def resume_from_files(
        cls,
        paths: Sequence[str],
        db_path: str,
        chunksize: int = CHUNK,
        expected_n: int | None = None,
    ) -> "SqliteReferenceIndex":
        """Continue Pass-1 ingest into an existing DB. Never deletes or recreates the file."""
        if not os.path.isfile(db_path):
            raise FileNotFoundError(db_path)
        conn = sqlite3.connect(db_path)
        cls._pragma_write(conn)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        required = {"refs", "tokc_name", "tokc_addr", "tok_row_name", "tok_row_addr"}
        missing = required - tables
        if missing:
            raise RuntimeError(f"cannot resume; missing tables {sorted(missing)}")
        if "inv_name" in tables or "inv_addr" in tables:
            raise RuntimeError("cannot resume ingest; inv_* already exist")

        n_refs = int(conn.execute("SELECT COUNT(*) FROM refs").fetchone()[0])
        mn, mx = conn.execute("SELECT MIN(row_id), MAX(row_id) FROM refs").fetchone()
        if n_refs == 0:
            raise RuntimeError("refs is empty; use build_from_files")
        if mn != 0 or mx != n_refs - 1:
            raise RuntimeError(f"refs row_id not contiguous 0..{n_refs-1}: min={mn} max={mx}")

        logging.getLogger("src.preprocessing.normalize").setLevel(logging.ERROR)
        t0 = time.time()
        logger.info(
            "RESUME ingest db=%s existing_refs=%d skip_rows=%d chunksize=%d rss_mb=%.1f",
            db_path, n_refs, n_refs, chunksize, _rss_mb(),
        )
        row_id = cls._stream_tsv_chunks(
            conn, paths, chunksize, skip_rows=n_refs, row_id=n_refs, t0=t0, db_path=db_path
        )
        logger.info("Pass 1 complete: %d rows rss_mb=%.1f", row_id, _rss_mb())
        if expected_n is not None and row_id != expected_n:
            raise RuntimeError(f"refs count {row_id} != expected {expected_n}")
        cls._finalize_token_indexes(conn, row_id, t0, db_path)
        conn.close()
        return cls(db_path)

    def posting(self, token: str, which: str, limit: int = MAX_POSTING) -> np.ndarray:
        table = "inv_name" if which == "name" else "inv_addr"
        rows = self.conn.execute(
            f"SELECT row_id FROM {table} WHERE token = ? LIMIT ?", (token, limit)
        ).fetchall()
        if not rows:
            return np.empty(0, dtype=np.int32)
        return np.fromiter((r[0] for r in rows), dtype=np.int32, count=len(rows))

    def token_df(self, token: str, which: str) -> int:
        table = "tokc_name" if which == "name" else "tokc_addr"
        r = self.conn.execute(f"SELECT c FROM {table} WHERE token = ?", (token,)).fetchone()
        return int(r[0]) if r else 0

    def fetch_rows(self, row_ids: Iterable[int]) -> List[Tuple[int, str, str, str, str]]:
        ids = [int(x) for x in row_ids]
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        return self.conn.execute(
            f"SELECT row_id, entity_id, name, addr, country FROM refs WHERE row_id IN ({q})",
            ids,
        ).fetchall()

    def exact_name_rows(self, name: str, country: str, limit: int = 40) -> List[int]:
        if not name:
            return []
        if country:
            rows = self.conn.execute(
                "SELECT row_id FROM refs WHERE name = ? AND country = ? LIMIT ?",
                (name, country, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT row_id FROM refs WHERE name = ? LIMIT ?",
                (name, limit),
            ).fetchall()
        return [r[0] for r in rows]

    def retrieve(
        self,
        name: str,
        addr: str,
        country: str,
        retrieve_cap: int = 350,
        max_tokens: int = 6,
        require_country: bool = True,
    ) -> Tuple[np.ndarray, int, bool]:
        scores = {}
        country = (country or "").lower()

        def add_tokens(text: str, which: str, min_len: int, stop: set, weight: int):
            toks = list(_tokens(text, min_len, stop))
            toks.sort(key=lambda t: self.token_df(t, which) or 10**9)
            used = 0
            for t in toks:
                posting = self.posting(t, which)
                if posting.size == 0:
                    continue
                used += 1
                for rid in posting:
                    scores[int(rid)] = scores.get(int(rid), 0) + weight
                if used >= max_tokens or len(scores) >= retrieve_cap * 3:
                    break

        for rid in self.exact_name_rows(name, country if require_country else ""):
            scores[rid] = scores.get(rid, 0) + 50
        add_tokens(name, "name", 3, LEGALISH, 3)
        add_tokens(addr, "addr", 2, set(), 2)
        if require_country and country and scores:
            recs = self.fetch_rows(list(scores.keys())[: retrieve_cap * 3])
            scores = {rid: scores[rid] for rid, _e, _n, _a, co in recs if co == country or not co}
        if not scores:
            return np.empty(0, dtype=np.int32), 0, False
        raw_n = len(scores)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:retrieve_cap]
        return np.asarray([r for r, _ in ranked], dtype=np.int32), raw_n, raw_n >= retrieve_cap

    def cheap_prerank(self, q_name, q_addr, q_country, rows: np.ndarray, top_k: int) -> np.ndarray:
        if rows.size == 0:
            return rows
        recs = {r[0]: r for r in self.fetch_rows(rows.tolist())}
        qn = set(q_name.split()) if q_name else set()
        qa = set(q_addr.split()) if q_addr else set()
        q_country = (q_country or "").lower()
        scored = []
        for rid in rows:
            rec = recs.get(int(rid))
            if not rec:
                continue
            _rid, _eid, cname, caddr, cco = rec
            cname = cname or ""
            caddr = caddr or ""
            sc = len(qn & set(cname.split())) * 3 + len(qa & set(caddr.split())) * 4
            if q_name and cname == q_name:
                sc += 40
            if q_addr and caddr == q_addr:
                sc += 40
            if q_country and cco == q_country:
                sc += 5
            scored.append((sc, int(rid)))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return np.asarray([r for _, r in scored[:top_k]], dtype=np.int32)

    def entity_id(self, row_id: int) -> str:
        r = self.conn.execute("SELECT entity_id FROM refs WHERE row_id=?", (int(row_id),)).fetchone()
        return r[0] if r else ""
