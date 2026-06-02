# src/lcsh_benchmark/scaleup/store.py
"""SQLite-backed index + consensus store for the production bulk pipeline.

Stdlib sqlite3, WAL mode. Provides a resumable chunk ledger, batched record
insert, the SQL consensus prefilter (>=N distinct sources AND >=M distinct
non-empty agencies), a small-group members reader, a consensus writer, and
pool-stat queries that drive the data-driven size decision. No parsing here.
"""
import json
import sqlite3
from collections.abc import Iterable, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL, chunk_id TEXT NOT NULL,
  match_key TEXT NOT NULL,
  oclc TEXT, lccn TEXT, lang TEXT, lc_class TEXT, agency TEXT,
  has_input INTEGER NOT NULL, is_monograph INTEGER NOT NULL DEFAULT 0,
  n_headings INTEGER NOT NULL,
  title TEXT, headings_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_records_key ON records(match_key);
CREATE TABLE IF NOT EXISTS chunks (
  source TEXT, chunk_id TEXT, status TEXT,
  n_records INTEGER, finished_at TEXT,
  PRIMARY KEY (source, chunk_id)
);
CREATE TABLE IF NOT EXISTS consensus (
  match_key TEXT PRIMARY KEY, lang TEXT, lc_class TEXT,
  catalog_count INTEGER, n_agencies INTEGER,
  has_input INTEGER, is_monograph INTEGER DEFAULT 0,
  title TEXT, gt_json TEXT, graded_json TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _add_column_if_missing(cx: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    cols = {r["name"] for r in cx.execute(f"PRAGMA table_info({table})")}
    if col not in cols:
        cx.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def connect(path: str) -> sqlite3.Connection:
    cx = sqlite3.connect(path)
    cx.row_factory = sqlite3.Row
    cx.execute("PRAGMA journal_mode=WAL")
    cx.executescript(_SCHEMA)
    # Migrate pre-A3 databases so the monograph filter can't silently drop all.
    _add_column_if_missing(cx, "records", "is_monograph", "INTEGER NOT NULL DEFAULT 0")
    _add_column_if_missing(cx, "consensus", "is_monograph", "INTEGER DEFAULT 0")
    cx.commit()
    return cx


def set_meta(cx: sqlite3.Connection, key: str, value: str) -> None:
    cx.execute("INSERT INTO meta(key,value) VALUES(?,?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    cx.commit()


def get_meta(cx: sqlite3.Connection, key: str) -> str | None:
    row = cx.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def mark_chunk(cx: sqlite3.Connection, source: str, chunk_id: str, status: str, n: int = 0) -> None:
    cx.execute(
        "INSERT INTO chunks(source,chunk_id,status,n_records,finished_at) "
        "VALUES(?,?,?,?,datetime('now')) "
        "ON CONFLICT(source,chunk_id) DO UPDATE SET "
        "status=excluded.status, n_records=excluded.n_records, finished_at=excluded.finished_at",
        (source, chunk_id, status, n))
    cx.commit()


def chunk_done(cx: sqlite3.Connection, source: str, chunk_id: str) -> bool:
    row = cx.execute("SELECT status FROM chunks WHERE source=? AND chunk_id=?",
                     (source, chunk_id)).fetchone()
    return row is not None and row["status"] == "done"


def insert_records(cx: sqlite3.Connection, rows: Iterable[dict]) -> None:
    cx.executemany(
        "INSERT INTO records(source,chunk_id,match_key,oclc,lccn,lang,lc_class,"
        "agency,has_input,is_monograph,n_headings,title,headings_json) VALUES("
        ":source,:chunk_id,:match_key,:oclc,:lccn,:lang,:lc_class,"
        ":agency,:has_input,:is_monograph,:n_headings,:title,:headings_json)", rows)
    cx.commit()


def delete_chunk_records(cx: sqlite3.Connection, source: str, chunk_id: str) -> None:
    """Remove any previously-inserted rows for a (source, chunk_id) shard so a
    re-load is idempotent (guards the insert/mark-loaded atomicity gap)."""
    cx.execute("DELETE FROM records WHERE source=? AND chunk_id=?", (source, chunk_id))
    cx.commit()


def iter_consensus_keys(cx: sqlite3.Connection, min_sources: int = 2, min_agencies: int = 2) -> Iterator[str]:
    cur = cx.execute(
        "SELECT match_key FROM records WHERE match_key != '' GROUP BY match_key "
        "HAVING COUNT(DISTINCT source) >= ? "
        "AND COUNT(DISTINCT NULLIF(agency,'')) >= ?", (min_sources, min_agencies))
    for row in cur:
        yield row["match_key"]


def members_for_key(cx: sqlite3.Connection, match_key: str) -> list[dict]:
    cur = cx.execute("SELECT * FROM records WHERE match_key=? ORDER BY source, oclc", (match_key,))
    return [{"source": r["source"], "oclc": r["oclc"], "lccn": r["lccn"],
             "lang": r["lang"], "lc_class": r["lc_class"], "agency": r["agency"],
             "title": r["title"], "has_input": bool(r["has_input"]),
             "is_monograph": bool(r["is_monograph"]),
             "headings": json.loads(r["headings_json"])} for r in cur]


def write_consensus(cx: sqlite3.Connection, matched: list[dict]) -> None:
    rows = [{
        "match_key": m["key"], "lang": m["lang"], "lc_class": m["lc_class"],
        "catalog_count": m["catalog_count"], "n_agencies": m["n_agencies"],
        "has_input": int(bool(m["has_input"])),
        "is_monograph": int(bool(m.get("is_monograph"))), "title": m["title"],
        "gt_json": json.dumps(m["ground_truth_lcsh"], ensure_ascii=False),
        "graded_json": json.dumps({"exact": m["consensus_exact"],
                                   "root": m["consensus_root"]}, ensure_ascii=False),
    } for m in matched]
    cx.executemany(
        "INSERT OR REPLACE INTO consensus(match_key,lang,lc_class,catalog_count,"
        "n_agencies,has_input,is_monograph,title,gt_json,graded_json) VALUES("
        ":match_key,:lang,:lc_class,:catalog_count,:n_agencies,:has_input,"
        ":is_monograph,:title,:gt_json,:graded_json)", rows)
    cx.commit()


def clear_consensus(cx: sqlite3.Connection) -> None:
    """Drop all consensus rows so a re-run of the match phase recomputes the
    full pool from scratch (the match phase is a pure function of `records`)."""
    cx.execute("DELETE FROM consensus")
    cx.commit()


def _grouped(cx: sqlite3.Connection, col: str) -> dict:
    cur = cx.execute(f"SELECT {col} AS k, COUNT(*) AS n FROM consensus "
                     f"GROUP BY {col} ORDER BY n DESC")
    return {row["k"]: row["n"] for row in cur}


def counts_by_lang(cx: sqlite3.Connection) -> dict:
    return _grouped(cx, "lang")


def counts_by_class(cx: sqlite3.Connection) -> dict:
    return _grouped(cx, "lc_class")


def counts_by_lang_class(cx: sqlite3.Connection) -> dict:
    cur = cx.execute("SELECT lang, lc_class, COUNT(*) AS n FROM consensus "
                     "GROUP BY lang, lc_class ORDER BY lang, n DESC")
    return {f"{row['lang']}/{row['lc_class']}": row["n"] for row in cur}


def iter_consensus_rows(cx: sqlite3.Connection) -> Iterator[dict]:
    for r in cx.execute("SELECT * FROM consensus"):
        graded = json.loads(r["graded_json"])
        yield {"key": r["match_key"], "lang": r["lang"], "lc_class": r["lc_class"],
               "has_input": bool(r["has_input"]),
               "is_monograph": bool(r["is_monograph"]), "title": r["title"],
               "catalog_count": r["catalog_count"], "n_agencies": r["n_agencies"],
               "ground_truth_lcsh": json.loads(r["gt_json"]),
               "consensus_exact": graded.get("exact", {}),
               "consensus_root": graded.get("root", {})}
