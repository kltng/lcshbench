from lcsh_benchmark.scaleup.match import match_consensus


def _idx(source, oclc, headings, lang="eng", cls="Q"):
    return {"source": source, "oclc": oclc, "lccn": "", "lang": lang,
            "lc_class": cls, "title": "T", "headings": headings, "has_input": True}


def test_match_keeps_only_records_with_two_assigning_sources():
    records = [
        _idx("harvard", "111", {"650": ["Sociology"]}),
        _idx("columbia", "111", {"650": ["Sociology", "Social networks"]}),
        _idx("princeton", "222", {"650": ["Botany"]}),
        _idx("loc", "333", {}),
        _idx("harvard", "333", {"650": ["Physics"]}),
    ]
    out = match_consensus(records, min_sources=2)
    assert len(out) == 1
    rec = out[0]
    assert rec["oclc"] == "111"
    assert set(rec["ground_truth_lcsh"]) == {"harvard", "columbia"}
    assert rec["catalog_count"] == 2
    assert rec["lc_class"] == "Q" and rec["lang"] == "eng"


def test_match_propagates_is_monograph_as_any_of_members():
    # A grouped record is a monograph if ANY contributing catalog coded it as
    # one (mirrors has_input). Lets the selection stage filter on it.
    def idx(src, mono):
        return {"source": src, "oclc": "111", "lccn": "", "lang": "eng",
                "lc_class": "Q", "title": "T", "headings": {"650": ["A"]},
                "has_input": True, "is_monograph": mono}
    out = match_consensus([idx("harvard", False), idx("columbia", True)], min_sources=2)
    assert out[0]["is_monograph"] is True

    out2 = match_consensus([idx("harvard", False), idx("columbia", False)], min_sources=2)
    assert out2[0]["is_monograph"] is False


def test_graded_consensus_tiers_exact_and_root():
    from lcsh_benchmark.scaleup.match import match_consensus
    def idx(src, h):
        return {"source": src, "oclc": "111", "lccn": "", "lang": "eng",
                "lc_class": "H", "title": "T", "headings": {"650": h}, "has_input": True}
    recs = [
        idx("harvard", ["Sociology", "Sociology--Research"]),
        idx("columbia", ["Sociology", "Sociology--Research"]),
        idx("princeton", ["Sociology", "Botany"]),
    ]
    out = match_consensus(recs, min_sources=2)
    assert len(out) == 1
    r = out[0]
    assert r["catalog_count"] == 3
    ce = r["consensus_exact"]
    assert ce["sociology"]["votes"] == 3 and ce["sociology"]["tier"] == "unanimous"
    assert ce["sociology--research"]["votes"] == 2 and ce["sociology--research"]["tier"] == "majority"
    assert ce["botany"]["votes"] == 1 and ce["botany"]["tier"] == "single"
    cr = r["consensus_root"]
    # all three assigned a 'Sociology' base -> root unanimous even though subdivisions differ
    assert cr["sociology"]["votes"] == 3 and cr["sociology"]["tier"] == "unanimous"
    assert cr["botany"]["votes"] == 1


def test_same_source_twice_unions_headings():
    recs = [
        {"source": "harvard", "oclc": "111", "lccn": "", "lang": "eng", "lc_class": "Q",
         "title": "T", "headings": {"650": ["A"]}, "has_input": True},
        {"source": "harvard", "oclc": "111", "lccn": "", "lang": "eng", "lc_class": "Q",
         "title": "T", "headings": {"650": ["B"]}, "has_input": True},
        {"source": "columbia", "oclc": "111", "lccn": "", "lang": "eng", "lc_class": "Q",
         "title": "T", "headings": {"650": ["A"]}, "has_input": True},
    ]
    out = match_consensus(recs, min_sources=2)
    assert len(out) == 1
    assert out[0]["ground_truth_lcsh"]["harvard"] == ["A", "B"]   # unioned, not overwritten


from lcsh_benchmark.scaleup.match import pick_stratify_fields


def test_pick_stratify_fields_is_deterministic_by_source_precedence():
    members = [
        {"source": "princeton", "lang": "fre", "lc_class": "D"},
        {"source": "harvard",   "lang": "eng", "lc_class": "P"},
        {"source": "columbia",  "lang": "eng", "lc_class": "P"},
    ]
    import random
    for _ in range(5):
        random.shuffle(members)
        assert pick_stratify_fields(members) == {"lang": "eng", "lc_class": "P"}
