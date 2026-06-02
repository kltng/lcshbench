# tests/test_consensus_parity.py
import json
from lcsh_benchmark.scaleup import store
from lcsh_benchmark.scaleup.match import match_consensus, match_key


def _idx(source, oclc, headings, agency):
    return {"source": source, "oclc": oclc, "lccn": "", "lang": "eng",
            "lc_class": "Q", "title": "T", "headings": headings,
            "has_input": True, "agency": agency}


def _to_row(idx):
    return {"source": idx["source"], "chunk_id": "c", "match_key": match_key(idx),
            "oclc": idx["oclc"], "lccn": idx["lccn"], "lang": idx["lang"],
            "lc_class": idx["lc_class"], "agency": idx["agency"], "has_input": 1,
            "is_monograph": int(bool(idx.get("is_monograph"))),
            "n_headings": sum(len(v) for v in idx["headings"].values()),
            "title": idx["title"], "headings_json": json.dumps(idx["headings"])}


def test_sql_path_matches_in_memory_path():
    records = [
        _idx("harvard", "111", {"650": ["Sociology"]}, "DLC"),
        _idx("columbia", "111", {"650": ["Sociology", "Networks"]}, "NNC"),
        _idx("harvard", "222", {"650": ["Botany"]}, "DLC"),
        _idx("columbia", "222", {"650": ["Botany"]}, "DLC"),    # same agency -> dropped
        _idx("princeton", "333", {"650": ["Physics"]}, "CtY"),  # 1 source -> dropped
    ]
    # in-memory path
    mem = match_consensus(records, min_sources=2, require_independent_agencies=2)
    mem_keys = sorted(r["key"] for r in mem)

    # SQL path
    cx = store.connect(":memory:")
    store.insert_records(cx, [_to_row(r) for r in records])
    sql = []
    for k in store.iter_consensus_keys(cx, 2, 2):
        sql += match_consensus(store.members_for_key(cx, k),
                               min_sources=2, require_independent_agencies=2)
    sql_keys = sorted(r["key"] for r in sql)

    assert sql_keys == mem_keys == ["o:111"]
    # graded consensus identical for the kept key
    mem_g = next(r for r in mem if r["key"] == "o:111")["consensus_exact"]
    sql_g = next(r for r in sql if r["key"] == "o:111")["consensus_exact"]
    assert mem_g == sql_g
