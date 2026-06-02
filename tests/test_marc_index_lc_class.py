"""A4: validate LC classification against the real LCC top-level letter set.

`lc_class` took the first alphabetic character with no validation, so the 7.6%
of dev records whose 050/090 starts with a non-LCC letter (I/O/W/X/Y) were
treated as real disciplines. Valid LCC top-level classes are
{A,B,C,D,E,F,G,H,J,K,L,M,N,P,Q,R,S,T,U,V,Z}; anything else -> "" (unknown).
"""
from pymarc import Record, Field, Subfield
from lcsh_benchmark.scaleup.marc_index import index_record


def _rec(*class_fields):
    r = Record()
    r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
    for f in class_fields:
        r.add_field(f)
    r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "X")]))
    return r


def _cls(*class_fields):
    return index_record(_rec(*class_fields), "test")["lc_class"]


def _f050(val):
    return Field(tag="050", indicators=[" ", "0"], subfields=[Subfield("a", val)])


def _f090(val):
    return Field(tag="090", indicators=[" ", " "], subfields=[Subfield("a", val)])


def test_valid_lcc_letter_kept():
    assert _cls(_f050("QA76.73 .P98")) == "Q"
    assert _cls(_f050("PR6045")) == "P"


def test_non_lcc_letters_mapped_to_unknown():
    # I, O, W, X, Y are not used in LCC top-level classification
    assert _cls(_f050("X123")) == ""
    assert _cls(_f050("WB 100")) == ""   # W = NLM, not LCC
    assert _cls(_f050("I 19.2")) == ""   # I = SuDoc, not LCC


def test_blank_or_nonalpha_class_is_unknown():
    assert _cls() == ""                   # no 050/090 at all
    assert _cls(_f050("12345")) == ""     # starts with a digit


def test_050_preferred_over_090():
    # LC-assigned 050 takes precedence over local 090
    assert _cls(_f050("QA76"), _f090("Z699")) == "Q"
