# tests/test_run.py
import gzip
import json
from pathlib import Path

from pymarc import Record, Field, Subfield
from pymarc import marcxml

from lcsh_benchmark.scaleup import run, store


def _marcxml_bytes(recs):
    """Build a MARCXML <collection> as bytes from a list of pymarc Records.
    Uses record_to_xml (available in pymarc >=5) instead of XMLWriter which
    was removed; round-trips correctly through map_xml."""
    import io
    parts = [b'<?xml version="1.0" encoding="UTF-8"?>'
             b'<collection xmlns="http://www.loc.gov/MARC21/slim">']
    for r in recs:
        parts.append(marcxml.record_to_xml(r, namespace=True))
    parts.append(b'</collection>')
    return b"".join(parts)


def _rec(oclc, agency, heading):
    r = Record()
    r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
    r.add_field(Field(tag="035", indicators=[" ", " "], subfields=[Subfield("a", f"(OCoLC){oclc}")]))
    r.add_field(Field(tag="040", indicators=[" ", " "], subfields=[Subfield("a", agency)]))
    r.add_field(Field(tag="050", indicators=[" ", "0"], subfields=[Subfield("a", "QA76")]))
    r.add_field(Field(tag="245", indicators=["1", "0"], subfields=[Subfield("a", "T")]))
    r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", heading)]))
    return r


def test_index_chunk_to_shard_writes_jsonl(tmp_path):
    gz = tmp_path / "c.xml.gz"
    gz.write_bytes(gzip.compress(_marcxml_bytes([_rec("111", "DLC", "Sociology")])))
    n, shard = run.index_chunk_to_shard(str(gz), "columbia", "001", "gz_marcxml", str(tmp_path / "corpus"))
    assert n == 1
    rows = [json.loads(l) for l in Path(shard).read_text().splitlines()]
    assert rows[0]["agency"] == "DLC" and rows[0]["headings"]["650"] == ["Sociology"]


def test_load_then_match_end_to_end(tmp_path):
    # two sources, same OCLC, distinct agencies -> one consensus record
    corpus = tmp_path / "corpus"
    for src, agency in [("harvard", "DLC"), ("columbia", "NNC")]:
        gz = tmp_path / f"{src}.xml.gz"
        gz.write_bytes(gzip.compress(_marcxml_bytes([_rec("111", agency, "Sociology")])))
        run.index_chunk_to_shard(str(gz), src, "001", "gz_marcxml", str(corpus))
    cx = store.connect(str(tmp_path / "v2.db"))
    run.load_shards(cx, str(corpus))
    n = run.run_match(cx, min_sources=2, min_agencies=2)
    assert n == 1
    assert store.counts_by_lang(cx) == {"eng": 1}


def test_load_shards_idempotent_when_meta_lost(tmp_path):
    corpus = tmp_path / "corpus"
    gz = tmp_path / "h.xml.gz"
    gz.write_bytes(gzip.compress(_marcxml_bytes([_rec("111", "DLC", "Sociology")])))
    run.index_chunk_to_shard(str(gz), "harvard", "001", "gz_marcxml", str(corpus))
    cx = store.connect(str(tmp_path / "v2.db"))
    run.load_shards(cx, str(corpus))
    # simulate a crash AFTER inserting rows but BEFORE the loaded-shards meta persisted
    store.set_meta(cx, "loaded_shards", "[]")
    run.load_shards(cx, str(corpus))   # re-run must NOT double-insert
    n = cx.execute("SELECT COUNT(*) AS c FROM records").fetchone()["c"]
    assert n == 1


def test_run_match_is_idempotent(tmp_path):
    import gzip
    corpus = tmp_path / "corpus"
    for src, agency in [("harvard", "DLC"), ("columbia", "NNC")]:
        gz = tmp_path / f"{src}.xml.gz"
        gz.write_bytes(gzip.compress(_marcxml_bytes([_rec("111", agency, "Sociology")])))
        run.index_chunk_to_shard(str(gz), src, "001", "gz_marcxml", str(corpus))
    cx = store.connect(str(tmp_path / "v2.db"))
    run.load_shards(cx, str(corpus))
    n1 = run.run_match(cx, min_sources=2, min_agencies=2)
    n2 = run.run_match(cx, min_sources=2, min_agencies=2)   # re-run
    assert n1 == n2 == 1
    assert sum(store.counts_by_lang(cx).values()) == 1   # no duplication


def test_index_chunk_to_shard_binary_marc_gz(tmp_path):
    import gzip as _gz
    gz = tmp_path / "x.mrc.gz"
    gz.write_bytes(_gz.compress(_rec("111", "DLC", "Sociology").as_marc()))
    n, shard = run.index_chunk_to_shard(str(gz), "loc", "001", "gz_marc", str(tmp_path / "corpus"))
    assert n == 1
    import json as _j
    assert _j.loads(Path(shard).read_text().splitlines()[0])["headings"]["650"] == ["Sociology"]


def test_process_chunk_caches_remote_raw_and_reuses(tmp_path, monkeypatch):
    """With a cache_dir, a remote chunk's raw is kept and a re-run skips the download."""
    cache = tmp_path / "raw_cache"
    corpus = tmp_path / "corpus"
    raw = gzip.compress(_marcxml_bytes([_rec("111", "DLC", "Sociology")]))
    calls = []

    def fake_download(url, dest, retries=4):
        calls.append(url)
        Path(dest).write_bytes(raw)

    monkeypatch.setattr(run, "_download", fake_download)
    desc = {"source": "columbia", "chunk_id": "001",
            "url": "https://x/001", "container": "gz_marcxml"}

    res1 = run._process_chunk(desc, str(corpus), cache_dir=str(cache))
    assert res1["status"] == "done" and res1["n"] == 1
    assert (cache / "columbia" / "001").exists()   # raw kept, not discarded
    assert len(calls) == 1

    res2 = run._process_chunk(desc, str(corpus), cache_dir=str(cache))
    assert res2["status"] == "done" and res2["n"] == 1
    assert len(calls) == 1                          # cache hit -> no re-download


def test_process_chunk_does_not_cache_local_file(tmp_path, monkeypatch):
    """A local file:// chunk is already persistent: index from a temp copy, don't
    duplicate it into the cache."""
    cache = tmp_path / "raw_cache"
    corpus = tmp_path / "corpus"
    raw = gzip.compress(_marcxml_bytes([_rec("222", "DLC", "Sociology")]))
    dests = []

    def fake_download(url, dest, retries=4):
        dests.append(dest)
        Path(dest).write_bytes(raw)

    monkeypatch.setattr(run, "_download", fake_download)
    desc = {"source": "harvard", "chunk_id": "h1.xml.gz",
            "url": "file:///incoming/harvard/h1.xml.gz", "container": "gz_marcxml"}

    res = run._process_chunk(desc, str(corpus), cache_dir=str(cache))
    assert res["status"] == "done" and res["n"] == 1
    assert not (cache / "harvard").exists()          # not duplicated into cache
    assert all(str(cache) not in d for d in dests)   # copied to a temp path, not cache


def test_sample_writes_manifest(tmp_path):
    cx = store.connect(":memory:")
    store.write_consensus(cx, [{
        "key": f"o:{i}", "lang": "eng", "lc_class": "Q", "catalog_count": 2,
        "n_agencies": 2, "has_input": True, "is_monograph": True, "title": "T",
        "ground_truth_lcsh": {"a": ["X"], "b": ["X"]},
        "consensus_exact": {}, "consensus_root": {}} for i in range(20)])
    out = tmp_path / "manifest.json"
    counts = run.run_sample(cx, str(out), core={"eng": 5}, breadth={"eng": 5})
    assert counts["core"] == 5 and counts["breadth"] == 5
    assert out.exists()
