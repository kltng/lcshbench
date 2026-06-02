"""H2: keep only LCSH (ind2==0, or ind2==7 $2=lcsh) 6xx and LCGFT (655 $2=lcgft).

Drop MeSH, FAST, foreign/unspecified thesauri so non-LCSH headings never enter
the consensus ground truth.
"""
from pymarc import Record, Field, Subfield
from lcsh_benchmark.scaleup.marc_index import index_record


def _rec(*fields):
    r = Record()
    r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
    for f in fields:
        r.add_field(f)
    return r


def _f(tag, ind2, subs):
    return Field(tag=tag, indicators=[" ", ind2],
                 subfields=[Subfield(c, v) for c, v in subs])


def test_keeps_lcsh_ind2_0():
    out = index_record(_rec(_f("650", "0", [("a", "Python")])), "c")
    assert out["headings"]["650"] == ["Python"]


def test_drops_mesh_ind2_2():
    out = index_record(_rec(_f("650", "2", [("a", "Neoplasms")])), "c")
    assert "650" not in out["headings"]


def test_drops_fast_ind2_7_source_fast():
    out = index_record(_rec(_f("650", "7", [("a", "Cooking"), ("2", "fast")])), "c")
    assert "650" not in out["headings"]


def test_keeps_lcsh_ind2_7_source_lcsh():
    out = index_record(_rec(_f("650", "7", [("a", "Cooking"), ("2", "lcsh")])), "c")
    assert out["headings"]["650"] == ["Cooking"]


def test_keeps_lcgft_655_source_lcgft():
    out = index_record(_rec(_f("655", "7", [("a", "Detective fiction"), ("2", "lcgft")])), "c")
    assert out["headings"]["655"] == ["Detective fiction"]


def test_drops_655_ind2_0_without_lcgft():
    out = index_record(_rec(_f("655", "0", [("a", "Fiction")])), "c")
    assert "655" not in out["headings"]


def test_drops_655_other_source_gsafd():
    out = index_record(_rec(_f("655", "7", [("a", "Mystery"), ("2", "gsafd")])), "c")
    assert "655" not in out["headings"]


def test_drops_geographic_651_mesh_keeps_lcsh():
    out = index_record(_rec(
        _f("651", "0", [("a", "United States")]),
        _f("651", "2", [("a", "Europe")]),
    ), "c")
    assert out["headings"]["651"] == ["United States"]
