from pymarc import Record, Field, Subfield
from lcsh_benchmark.scaleup.marc_index import index_record


def test_index_record_captures_880_vernacular():
    """245/1xx hold romanized text; linked 880 fields hold the original script."""
    r = Record()
    r.add_field(Field(tag="008", data="000101s2014    ja            000 0 jpn d"))
    r.add_field(Field(tag="100", indicators=["1", " "],
                      subfields=[Subfield("6", "880-01"), Subfield("a", "Okada, Jo,")]))
    r.add_field(Field(tag="245", indicators=["1", "0"],
                      subfields=[Subfield("6", "880-02"), Subfield("a", "Nihon no bijutsu /")]))
    r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "Art, Japanese")]))
    # 880s link back to their romanized field via $6 "<tag>-<occ>/<script>"
    r.add_field(Field(tag="880", indicators=["1", " "],
                      subfields=[Subfield("6", "100-01/$1"), Subfield("a", "岡田譲,")]))
    r.add_field(Field(tag="880", indicators=["1", "0"],
                      subfields=[Subfield("6", "245-02/$1"), Subfield("a", "日本の美術")]))
    out = index_record(r, "harvard")
    assert out["title"] == "Nihon no bijutsu /"          # romanized preserved
    assert out["title_vernacular"] == "日本の美術"          # original script added
    assert out["authors"] == ["Okada, Jo,"]
    assert out["authors_vernacular"] == ["岡田譲,"]


def test_vernacular_empty_when_no_880():
    r = Record()
    r.add_field(Field(tag="008", data="000101s2014    xxu           000 0 eng d"))
    r.add_field(Field(tag="245", indicators=["1", "0"],
                      subfields=[Subfield("a", "Plain English title /")]))
    r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "X")]))
    out = index_record(r, "columbia")
    assert out["title_vernacular"] == ""
    assert out["authors_vernacular"] == []
