from pymarc import Record, Field, Subfield
from lcsh_benchmark.scaleup.marc_index import index_record


def _rec():
    r = Record()
    r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
    r.add_field(Field(tag="010", indicators=[" ", " "], subfields=[Subfield("a", "  2010012345 ")]))
    r.add_field(Field(tag="035", indicators=[" ", " "], subfields=[Subfield("a", "(OCoLC)123456789")]))
    r.add_field(Field(tag="050", indicators=[" ", "0"], subfields=[Subfield("a", "QA76.73"), Subfield("b", ".P98")]))
    r.add_field(Field(tag="245", indicators=["1", "0"], subfields=[Subfield("a", "Python programming")]))
    r.add_field(Field(tag="520", indicators=[" ", " "], subfields=[Subfield("a", "An intro to Python.")]))
    r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "Python (Computer program language)")]))
    r.add_field(Field(tag="651", indicators=[" ", "0"], subfields=[Subfield("a", "United States")]))
    return r


def test_index_record_extracts_keys_class_lang_and_tagged_headings():
    out = index_record(_rec(), "columbia")
    assert out["source"] == "columbia"
    assert out["oclc"] == "123456789"
    assert out["lccn"] == "2010012345"
    assert out["lang"] == "eng"
    assert out["lc_class"] == "Q"
    assert out["headings"]["650"] == ["Python (Computer program language)"]
    assert out["headings"]["651"] == ["United States"]
    assert out["has_input"] is True
    assert out["title"] == "Python programming"


def test_heading_preserves_document_subfield_order():
    from pymarc import Record, Field, Subfield
    from lcsh_benchmark.scaleup.marc_index import index_record
    r = Record()
    r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
    r.add_field(Field(tag="650", indicators=[" ", "0"],
                      subfields=[Subfield("a", "Science"), Subfield("z", "United States"),
                                 Subfield("x", "History")]))
    out = index_record(r, "loc")
    assert out["headings"]["650"] == ["Science--United States--History"]


def test_oclc_prefixes_stripped():
    def _rec_oclc(val):
        r = Record()
        r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
        r.add_field(Field(tag="035", indicators=[" ", " "], subfields=[Subfield("a", val)]))
        r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "X")]))
        return r
    assert index_record(_rec_oclc("(OCoLC)ocm00012345"), "p")["oclc"] == "12345"
    assert index_record(_rec_oclc("(OCoLC)ocn123456789"), "p")["oclc"] == "123456789"
    assert index_record(_rec_oclc("(OCoLC)on1234567890"), "p")["oclc"] == "1234567890"
    assert index_record(_rec_oclc("(OCoLC)777"), "p")["oclc"] == "777"


def test_all_zero_oclc_rejected():
    """The o:0 bug: (OCoLC)0 / all-zero forms must NOT key as oclc '0' — they
    collapse unrelated records into one false-matched group. Reject -> ''."""
    def _rec_oclc(val):
        r = Record()
        r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
        r.add_field(Field(tag="035", indicators=[" ", " "], subfields=[Subfield("a", val)]))
        r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "X")]))
        return r
    assert index_record(_rec_oclc("(OCoLC)0"), "p")["oclc"] == ""
    assert index_record(_rec_oclc("(OCoLC)000"), "p")["oclc"] == ""
    assert index_record(_rec_oclc("(OCoLC)ocm00000000"), "p")["oclc"] == ""
    assert index_record(_rec_oclc("(OCoLC)on0000000000"), "p")["oclc"] == ""
