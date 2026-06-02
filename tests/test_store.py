# tests/test_store.py
import json
from lcsh_benchmark.scaleup import store


def _row(source, key, agency, lang="eng", cls="Q", headings=None, is_monograph=1):
    headings = headings or {"650": ["X"]}
    return {"source": source, "chunk_id": "c1", "match_key": key,
            "oclc": key[2:], "lccn": "", "lang": lang, "lc_class": cls,
            "agency": agency, "has_input": 1, "is_monograph": is_monograph,
            "n_headings": sum(len(v) for v in headings.values()),
            "title": "T", "headings_json": json.dumps(headings)}


def test_is_monograph_round_trips_through_records_and_consensus():
    cx = store.connect(":memory:")
    store.insert_records(cx, [_row("harvard", "o:1", "DLC", is_monograph=1),
                              _row("columbia", "o:2", "NNC", is_monograph=0)])
    members = {m["source"]: m["is_monograph"] for m in store.members_for_key(cx, "o:1")}
    assert members == {"harvard": True}
    assert store.members_for_key(cx, "o:2")[0]["is_monograph"] is False

    store.write_consensus(cx, [{
        "key": "o:9", "lang": "eng", "lc_class": "Q", "catalog_count": 2,
        "n_agencies": 2, "has_input": True, "is_monograph": True, "title": "T",
        "ground_truth_lcsh": {"a": ["X"]}, "consensus_exact": {}, "consensus_root": {}}])
    assert list(store.iter_consensus_rows(cx))[0]["is_monograph"] is True


def test_connect_migrates_pre_a3_db_missing_is_monograph(tmp_path):
    import sqlite3
    db = str(tmp_path / "old.db")
    # Simulate a pre-A3 records table with NO is_monograph column.
    raw = sqlite3.connect(db)
    raw.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, source TEXT, "
                "chunk_id TEXT, match_key TEXT, oclc TEXT, lccn TEXT, lang TEXT, "
                "lc_class TEXT, agency TEXT, has_input INTEGER, n_headings INTEGER, "
                "title TEXT, headings_json TEXT)")
    raw.commit()
    raw.close()
    # connect() must add the column rather than crash on the next insert.
    cx = store.connect(db)
    cols = {r["name"] for r in cx.execute("PRAGMA table_info(records)")}
    assert "is_monograph" in cols
    store.insert_records(cx, [_row("harvard", "o:1", "DLC", is_monograph=1)])
    assert store.members_for_key(cx, "o:1")[0]["is_monograph"] is True


def test_chunk_ledger_marks_and_skips():
    cx = store.connect(":memory:")
    assert store.chunk_done(cx, "columbia", "001") is False
    store.mark_chunk(cx, "columbia", "001", "done", 42)
    assert store.chunk_done(cx, "columbia", "001") is True
    store.mark_chunk(cx, "columbia", "002", "failed", 0)
    assert store.chunk_done(cx, "columbia", "002") is False


def test_insert_and_consensus_prefilter():
    cx = store.connect(":memory:")
    store.insert_records(cx, [
        _row("harvard", "o:111", "DLC"),
        _row("columbia", "o:111", "NNC"),     # 111: 2 sources, 2 agencies -> kept
        _row("harvard", "o:222", "DLC"),
        _row("columbia", "o:222", "DLC"),     # 222: 2 sources, 1 agency  -> dropped
        _row("princeton", "o:333", "DLC"),    # 333: 1 source            -> dropped
    ])
    keys = sorted(store.iter_consensus_keys(cx, min_sources=2, min_agencies=2))
    assert keys == ["o:111"]
    members = store.members_for_key(cx, "o:111")
    assert {m["source"] for m in members} == {"harvard", "columbia"}
    assert members[0]["headings"] == {"650": ["X"]}   # JSON round-trips


def test_delete_chunk_records_removes_only_that_shard():
    cx = store.connect(":memory:")
    store.insert_records(cx, [
        _row("harvard", "o:1", "DLC"), _row("harvard", "o:2", "DLC"),
        _row("columbia", "o:3", "NNC"),
    ])
    # _row sets chunk_id="c1"; delete only harvard/c1
    store.delete_chunk_records(cx, "harvard", "c1")
    remaining = [m["source"] for k in ["o:1", "o:2", "o:3"]
                 for m in store.members_for_key(cx, k)]
    assert remaining == ["columbia"]   # only the columbia row survives


def test_write_consensus_and_pool_stats():
    cx = store.connect(":memory:")
    store.write_consensus(cx, [{
        "key": "o:111", "lang": "eng", "lc_class": "Q",
        "catalog_count": 2, "n_agencies": 2, "has_input": True, "title": "T",
        "ground_truth_lcsh": {"harvard": ["X"], "columbia": ["X"]},
        "consensus_exact": {}, "consensus_root": {}}])
    assert store.counts_by_lang(cx) == {"eng": 1}
    assert store.counts_by_class(cx) == {"Q": 1}
    rows = list(store.iter_consensus_rows(cx))
    assert rows[0]["key"] == "o:111" and rows[0]["has_input"] is True
    assert rows[0]["ground_truth_lcsh"]["harvard"] == ["X"]
    assert rows[0]["consensus_exact"] == {}
    assert rows[0]["consensus_root"] == {}
    assert "graded" not in rows[0]


def test_clear_consensus_empties_table():
    cx = store.connect(":memory:")
    store.write_consensus(cx, [{
        "key": "o:1", "lang": "eng", "lc_class": "Q", "catalog_count": 2,
        "n_agencies": 2, "has_input": True, "title": "T",
        "ground_truth_lcsh": {"a": ["X"]}, "consensus_exact": {}, "consensus_root": {}}])
    assert store.counts_by_lang(cx) == {"eng": 1}
    store.clear_consensus(cx)
    assert store.counts_by_lang(cx) == {}
    assert list(store.iter_consensus_rows(cx)) == []
