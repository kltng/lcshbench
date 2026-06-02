# tests/test_marc_index_inputs.py
from pymarc import Record, Field, Subfield
from lcsh_benchmark.scaleup.marc_index import index_record


def test_index_record_captures_bibliographic_inputs():
    r = Record()
    r.add_field(Field(tag="008", data="000101s2014    xxu           000 0 eng d"))  # date1 = 2014 at [7:11]
    r.add_field(Field(tag="100", indicators=["1", " "], subfields=[Subfield("a", "Manzo, Gianluca")]))
    r.add_field(Field(tag="700", indicators=["1", " "], subfields=[Subfield("a", "Editor, Jane")]))
    r.add_field(Field(tag="264", indicators=[" ", "1"], subfields=[Subfield("b", "Springer,"), Subfield("c", "2014.")]))
    r.add_field(Field(tag="300", indicators=[" ", " "], subfields=[Subfield("a", "x, 250 p. ;"), Subfield("c", "24 cm.")]))
    r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "Sociology")]))
    out = index_record(r, "columbia")
    assert out["authors"] == ["Manzo, Gianluca", "Editor, Jane"]
    assert out["date"] == "2014"
    assert out["publisher"] == "Springer,"
    assert out["physical_description"] == "x, 250 p. ; 24 cm."


def test_date_falls_back_to_264c_when_008_blank():
    r = Record()
    r.add_field(Field(tag="008", data="000101s        xxu           000 0 eng d"))  # no 4-digit date1
    r.add_field(Field(tag="264", indicators=[" ", "1"], subfields=[Subfield("c", "[2018]")]))
    r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "X")]))
    out = index_record(r, "harvard")
    assert out["date"] == "[2018]"
    assert out["authors"] == []          # no 1xx/7xx
