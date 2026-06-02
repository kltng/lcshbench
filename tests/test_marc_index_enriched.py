# tests/test_marc_index_enriched.py
from pymarc import Record, Field, Subfield
from lcsh_benchmark.scaleup.marc_index import index_record


def _rec():
    r = Record()
    r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
    r.add_field(Field(tag="040", indicators=[" ", " "],
                      subfields=[Subfield("a", "DLC"), Subfield("d", "NjP")]))
    r.add_field(Field(tag="245", indicators=["1", "0"], subfields=[Subfield("a", "Title")]))
    r.add_field(Field(tag="505", indicators=["0", " "], subfields=[Subfield("a", "Ch1. Intro -- Ch2. More")]))
    r.add_field(Field(tag="520", indicators=[" ", " "], subfields=[Subfield("a", "A summary.")]))
    r.add_field(Field(tag="500", indicators=[" ", " "], subfields=[Subfield("a", "A note.")]))
    r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "Sociology")]))
    return r


def test_index_record_captures_agency_and_input_text():
    out = index_record(_rec(), "columbia")
    assert out["agency"] == "DLC"            # 040$a, uppercased
    assert out["abstract"] == "A summary."   # 520
    assert out["toc"] == "Ch1. Intro -- Ch2. More"  # 505
    assert out["notes"] == "A note."         # 500
    # existing keys unchanged
    assert out["headings"]["650"] == ["Sociology"]
    assert out["has_input"] is True


def test_agency_falls_back_to_040d_then_empty():
    r = Record()
    r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
    r.add_field(Field(tag="040", indicators=[" ", " "], subfields=[Subfield("d", "NjP")]))
    r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "X")]))
    assert index_record(r, "princeton")["agency"] == "NJP"   # $d uppercased
    r2 = Record()
    r2.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "X")]))
    assert index_record(r2, "loc")["agency"] == ""           # no 040
