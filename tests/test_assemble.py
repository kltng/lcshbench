# tests/test_assemble.py
from lcsh_benchmark.scaleup.assemble import assemble_record, merge_inputs, tag_for_heading


def _corpus(source, oclc, title, headings, **kw):
    base = {"source": source, "oclc": oclc, "lccn": "", "lang": "eng", "lc_class": "H",
            "title": title, "title_vernacular": "", "authors": [], "authors_vernacular": [],
            "date": "", "publisher": "", "physical_description": "", "abstract": "",
            "toc": "", "notes": "", "headings": headings}
    base.update(kw)
    return base


def test_merge_inputs_takes_longest_non_empty_with_provenance():
    per = {
        "columbia": {"title": "Short", "abstract": "", "authors": ["A"]},
        "harvard": {"title": "A longer title", "abstract": "An abstract.", "authors": ["A", "B"]},
    }
    merged, prov = merge_inputs(per)
    assert merged["title"] == "A longer title" and prov["title"] == "harvard"
    assert merged["abstract"] == "An abstract." and prov["abstract"] == "harvard"
    assert merged["authors"] == ["A", "B"] and prov["authors"] == "harvard"


def test_merge_inputs_carries_vernacular_fields():
    """880 vernacular (original script) flows through; authors_vernacular is a list."""
    per = {
        "columbia": {"title": "Nihon no bijutsu", "title_vernacular": "",
                     "authors": ["Okada, Jo"], "authors_vernacular": []},
        "harvard": {"title": "Nihon no bijutsu", "title_vernacular": "日本の美術",
                    "authors": ["Okada, Jo"], "authors_vernacular": ["岡田譲"]},
    }
    merged, prov = merge_inputs(per)
    assert merged["title_vernacular"] == "日本の美術" and prov["title_vernacular"] == "harvard"
    assert merged["authors_vernacular"] == ["岡田譲"] and prov["authors_vernacular"] == "harvard"


def test_tag_for_heading_priority():
    per = {"columbia": {"650": ["Sociology"]}, "harvard": {"651": ["Sociology"]}}
    assert tag_for_heading("Sociology", per) == "650"   # 650 beats 651
    assert tag_for_heading("Nowhere", per) == ""


def test_assemble_builds_three_gt_views_and_types():
    manifest_rec = {
        "key": "o:111", "lang": "eng", "lc_class": "H", "n_agencies": 2,
        "ground_truth_lcsh": {"columbia": ["Sociology", "Social networks"],
                              "harvard": ["Sociology", "Group theory"]},
        "consensus_exact": {
            "sociology": {"surface": "Sociology", "votes": 2, "sources": ["columbia", "harvard"], "tier": "unanimous"},
            "social networks": {"surface": "Social networks", "votes": 1, "sources": ["columbia"], "tier": "single"},
            "group theory": {"surface": "Group theory", "votes": 1, "sources": ["harvard"], "tier": "single"},
        },
        "consensus_root": {},
    }
    per_source = {
        "columbia": _corpus("columbia", "111", "Analytical sociology",
                            {"650": ["Sociology", "Social networks"]}, authors=["Manzo, G"]),
        "harvard": _corpus("harvard", "111", "Analytical sociology: actions and networks",
                           {"650": ["Sociology"], "651": ["Group theory"]}),
    }
    rec = assemble_record("o:111", manifest_rec, per_source)
    assert rec["id"] == "o:111" and rec["oclc"] == "111"
    assert rec["language_code"] == "eng" and rec["language"] == "English"
    assert rec["title"] == "Analytical sociology: actions and networks"   # longest
    assert rec["authors"] == ["Manzo, G"]
    assert set(rec["ground_truth_lcsh_merged"]) == {"Sociology", "Social networks", "Group theory"}
    assert rec["ground_truth_lcsh_unanimous"] == ["Sociology"]            # only the unanimous one
    assert rec["catalog_count"] == 2 and rec["n_agencies"] == 2
    assert rec["heading_types"]["Sociology"]["type"] == "topical"
    assert rec["heading_types"]["Sociology"]["tag"] == "650"


def test_assemble_drops_record_with_only_artifacts():
    manifest_rec = {"key": "o:9", "lang": "eng", "lc_class": "", "n_agencies": 2,
                    "ground_truth_lcsh": {"columbia": ["653-11/"], "harvard": ["880-12"]},
                    "consensus_exact": {}, "consensus_root": {}}
    per_source = {"columbia": _corpus("columbia", "9", "T", {}),
                  "harvard": _corpus("harvard", "9", "T", {})}
    assert assemble_record("o:9", manifest_rec, per_source) is None


def test_unanimous_excludes_artifact_even_if_unanimous_tier():
    # An artifact string that (pathologically) reached consensus_exact as unanimous
    # must NOT appear in the released unanimous GT, and must be cleaned from all views.
    manifest_rec = {
        "key": "o:5", "lang": "eng", "lc_class": "H", "n_agencies": 2,
        "ground_truth_lcsh": {"columbia": ["Sociology", "653-11/"],
                              "harvard": ["Sociology", "653-11/"]},
        "consensus_exact": {
            "sociology": {"surface": "Sociology", "votes": 2, "sources": ["columbia", "harvard"], "tier": "unanimous"},
            "653-11/": {"surface": "653-11/", "votes": 2, "sources": ["columbia", "harvard"], "tier": "unanimous"},
        },
        "consensus_root": {},
    }
    per_source = {
        "columbia": _corpus("columbia", "5", "T", {"650": ["Sociology"]}),
        "harvard": _corpus("harvard", "5", "T", {"650": ["Sociology"]}),
    }
    rec = assemble_record("o:5", manifest_rec, per_source)
    assert rec["ground_truth_lcsh_unanimous"] == ["Sociology"]          # artifact excluded
    assert "653-11/" not in rec["ground_truth_lcsh_merged"]             # cleaned everywhere
