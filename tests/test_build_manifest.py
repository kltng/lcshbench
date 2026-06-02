import gzip
import json
import tarfile
import tempfile

from lcsh_benchmark.scaleup.build_manifest import (
    stream_index_marcxml_gz, stream_index_marcxml_tar, build)

MARCXML = '''<?xml version="1.0" encoding="UTF-8"?>
<collection xmlns="http://www.loc.gov/MARC21/slim">
<record><leader>00000nam a2200000 a 4500</leader>
<controlfield tag="008">000000s2010    xxu           000 0 eng d</controlfield>
<datafield tag="035" ind1=" " ind2=" "><subfield code="a">(OCoLC)555</subfield></datafield>
<datafield tag="650" ind1=" " ind2="0"><subfield code="a">Testing</subfield></datafield>
</record></collection>'''


def test_stream_index_gz(tmp_path):
    gzp = tmp_path / "c.xml.gz"
    with gzip.open(gzp, "wb") as f:
        f.write(MARCXML.encode())
    out = tmp_path / "idx.jsonl"
    with open(out, "w", encoding="utf-8") as oj:
        n = stream_index_marcxml_gz(str(gzp), "columbia", oj)
    assert n == 1
    rec = json.loads(open(out).read().strip())
    assert rec["oclc"] == "555" and rec["headings"]["650"] == ["Testing"]


def test_stream_index_tar(tmp_path):
    member = tmp_path / "POD_1"
    member.write_text(MARCXML, encoding="utf-8")
    tarp = tmp_path / "p.tar.gz"
    with tarfile.open(tarp, "w:gz") as tf:
        tf.add(str(member), arcname="POD_1")
    out = tmp_path / "idx.jsonl"
    with open(out, "w", encoding="utf-8") as oj:
        n = stream_index_marcxml_tar(str(tarp), "princeton", oj)
    assert n == 1


def test_build_produces_manifest_with_graded_consensus(tmp_path):
    # two sources, same OCLC, both assign Sociology -> matched + unanimous
    idx = tmp_path / "idx.jsonl"
    rows = [
        {"source": "harvard", "oclc": "9", "lccn": "", "lang": "eng", "lc_class": "H",
         "title": "T", "headings": {"650": ["Sociology"]}, "has_input": True, "is_monograph": True},
        {"source": "columbia", "oclc": "9", "lccn": "", "lang": "eng", "lc_class": "H",
         "title": "T", "headings": {"650": ["Sociology"]}, "has_input": True, "is_monograph": True},
    ]
    with open(idx, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "manifest.json"
    counts = build([str(idx)], str(out))
    assert counts["matched"] == 1
    man = json.loads(open(out).read())
    rec = (man["core"] + man["breadth"])[0]
    assert rec["consensus_exact"]["sociology"]["tier"] == "unanimous"


def test_build_excludes_non_monograph_matches(tmp_path):
    # A consensus-matched serial (is_monograph False) must NOT be sampled (A3).
    idx = tmp_path / "idx.jsonl"
    rows = [
        {"source": "harvard", "oclc": "7", "lccn": "", "lang": "eng", "lc_class": "H",
         "title": "Serial", "headings": {"650": ["Periodicals"]}, "has_input": True, "is_monograph": False},
        {"source": "columbia", "oclc": "7", "lccn": "", "lang": "eng", "lc_class": "H",
         "title": "Serial", "headings": {"650": ["Periodicals"]}, "has_input": True, "is_monograph": False},
    ]
    with open(idx, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "manifest.json"
    counts = build([str(idx)], str(out))
    assert counts["matched"] == 1            # it matched...
    assert counts["monographs"] == 0         # ...but was filtered out of selection
    man = json.loads(open(out).read())
    assert man["core"] + man["breadth"] == []
