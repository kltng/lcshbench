# tests/test_sources.py
from lcsh_benchmark.scaleup import sources


def test_princeton_enumerates_fixed_range():
    chunks = list(sources.princeton_chunks())
    assert len(chunks) == 31
    assert chunks[0]["source"] == "princeton"
    assert chunks[0]["chunk_id"] == "00"
    assert chunks[0]["container"] == "tar_marcxml"
    assert chunks[0]["url"].endswith("download/0")
    assert chunks[-1]["url"].endswith("download/30")


def test_columbia_probes_until_missing():
    present = {1, 2, 3}
    chunks = list(sources.columbia_chunks(exists=lambda url, n: n in present, max_n=10))
    assert [c["chunk_id"] for c in chunks] == ["001", "002", "003"]
    assert chunks[0]["container"] == "gz_marcxml"
    assert chunks[2]["url"].endswith("extract-003.xml.gz")


def test_harvard_parses_dataverse_file_list():
    fake = {"data": {"latestVersion": {"files": [
        {"dataFile": {"id": 42, "filename": "marc_01.xml"}},
        {"dataFile": {"id": 43, "filename": "marc_02.xml"}},
    ]}}}
    chunks = list(sources.harvard_chunks(fetch_json=lambda url: fake))
    assert [c["chunk_id"] for c in chunks] == ["42", "43"]
    assert chunks[0]["container"] == "gz_marcxml"   # MARCXML treated as map_xml input
    assert "access/datafile/42" in chunks[0]["url"]


def test_local_chunks_infers_source_and_container(tmp_path):
    (tmp_path / "harvard").mkdir()
    (tmp_path / "loc").mkdir()
    (tmp_path / "harvard" / "bibdata.tar.gz").write_bytes(b"x")
    (tmp_path / "loc" / "books_part1.xml.gz").write_bytes(b"x")
    (tmp_path / "loc" / "README.txt").write_bytes(b"ignore me")
    chunks = sorted(sources.local_chunks(str(tmp_path)), key=lambda c: (c["source"], c["chunk_id"]))
    assert [(c["source"], c["chunk_id"], c["container"]) for c in chunks] == [
        ("harvard", "bibdata.tar.gz", "tar_marc"),
        ("loc", "books_part1.xml.gz", "gz_marcxml"),
    ]
    assert chunks[0]["url"].startswith("file://")


def test_local_chunks_missing_dir_is_empty():
    assert list(sources.local_chunks("/nonexistent/path/xyz")) == []


def test_iter_chunks_dispatches():
    assert list(sources.iter_chunks("princeton"))[0]["source"] == "princeton"
