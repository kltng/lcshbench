# tests/test_binary_marc.py
import gzip, io, tarfile, json
from pathlib import Path
from pymarc import Record, Field, Subfield
from lcsh_benchmark.scaleup.build_manifest import stream_index_marc_gz, stream_index_marc_tar


def _rec(oclc, agency, heading):
    r = Record()
    r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
    r.add_field(Field(tag="035", indicators=[" ", " "], subfields=[Subfield("a", f"(OCoLC){oclc}")]))
    r.add_field(Field(tag="040", indicators=[" ", " "], subfields=[Subfield("a", agency)]))
    r.add_field(Field(tag="050", indicators=[" ", "0"], subfields=[Subfield("a", "QA76")]))
    r.add_field(Field(tag="245", indicators=["1", "0"], subfields=[Subfield("a", "T")]))
    r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", heading)]))
    return r


def test_stream_index_marc_gz(tmp_path):
    gz = tmp_path / "loc.mrc.gz"
    gz.write_bytes(gzip.compress(_rec("111", "DLC", "Sociology").as_marc()))
    out = tmp_path / "shard.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        n = stream_index_marc_gz(str(gz), "loc", fh)
    assert n == 1
    row = json.loads(out.read_text().splitlines()[0])
    assert row["agency"] == "DLC" and row["headings"]["650"] == ["Sociology"]


def test_stream_index_marc_tar(tmp_path):
    # tar.gz of two binary MARC member files (Harvard-style)
    tar_path = tmp_path / "harvard.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for i, rec in enumerate([_rec("111", "DLC", "Sociology"), _rec("222", "DLC", "Botany")]):
            data = rec.as_marc()
            info = tarfile.TarInfo(name=f"part{i}.mrc"); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    out = tmp_path / "shard.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        n = stream_index_marc_tar(str(tar_path), "harvard", fh)
    assert n == 2
    headings = [json.loads(l)["headings"]["650"][0] for l in out.read_text().splitlines()]
    assert set(headings) == {"Sociology", "Botany"}
