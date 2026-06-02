# tests/test_match_independence.py
from lcsh_benchmark.scaleup.match import match_consensus, match_key


def _idx(source, oclc, headings, agency, lang="eng", cls="Q"):
    return {"source": source, "oclc": oclc, "lccn": "", "lang": lang,
            "lc_class": cls, "title": "T", "headings": headings,
            "has_input": True, "agency": agency}


def test_distinct_agencies_kept_same_agency_dropped():
    records = [
        # key 111: 2 sources but BOTH agency DLC -> copy-cataloged -> dropped
        _idx("harvard", "111", {"650": ["Sociology"]}, "DLC"),
        _idx("columbia", "111", {"650": ["Sociology"]}, "DLC"),
        # key 222: 2 sources, distinct agencies -> independent -> kept
        _idx("harvard", "222", {"650": ["Botany"]}, "DLC"),
        _idx("columbia", "222", {"650": ["Botany", "Plants"]}, "NNC"),
    ]
    out = match_consensus(records, min_sources=2, require_independent_agencies=2)
    assert [r["oclc"] for r in out] == ["222"]
    assert out[0]["n_agencies"] == 2
    assert out[0]["catalog_count"] == 2


def test_empty_agency_does_not_count_toward_independence():
    records = [
        _idx("harvard", "333", {"650": ["X"]}, "DLC"),
        _idx("columbia", "333", {"650": ["X"]}, ""),     # empty agency
    ]
    out = match_consensus(records, min_sources=2, require_independent_agencies=2)
    assert out == []   # only 1 distinct non-empty agency


def test_default_is_back_compatible_without_agency_field():
    # Phase 5a-style records with NO agency key must still match (default=0)
    records = [
        {"source": "harvard", "oclc": "9", "lccn": "", "lang": "eng",
         "lc_class": "Q", "title": "T", "headings": {"650": ["Y"]}, "has_input": True},
        {"source": "columbia", "oclc": "9", "lccn": "", "lang": "eng",
         "lc_class": "Q", "title": "T", "headings": {"650": ["Y"]}, "has_input": True},
    ]
    out = match_consensus(records, min_sources=2)   # default require_independent_agencies=0
    assert len(out) == 1 and out[0]["n_agencies"] == 0


def test_match_key_precedence():
    assert match_key({"oclc": "5", "lccn": "x"}) == "o:5"
    assert match_key({"oclc": "", "lccn": "x"}) == "l:x"
    assert match_key({"oclc": "", "lccn": ""}) == ""


def test_non_assigning_member_agency_does_not_count():
    records = [
        _idx("harvard", "444", {"650": ["X"]}, "DLC"),
        _idx("columbia", "444", {}, "NNC"),   # present but assigns no headings
    ]
    # NNC must not count toward independence (no headings) -> only 1 agency
    out = match_consensus(records, min_sources=1, require_independent_agencies=2)
    assert out == []
    out0 = match_consensus(records, min_sources=1, require_independent_agencies=0)
    assert out0[0]["n_agencies"] == 1
