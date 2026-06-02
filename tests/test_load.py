from lcsh_benchmark.load import to_canonical


def test_to_canonical_500_style_flat_identifiers():
    rec = {
        "id": 1, "lccn": "123", "isbn": "978", "title": "T", "authors": ["A"],
        "language_code": "eng", "language": "English", "date": "2020",
        "ground_truth_lcsh": {"harvard": ["Sociology"], "columbia": ["Sociology", "Music"]},
        "ground_truth_lcsh_merged": ["Sociology", "Music"],
    }
    out = to_canonical(rec, source="multi500")
    assert out["id"] == "multi500-1"
    assert out["lccn"] == "123" and out["isbn"] == "978"
    assert out["catalogs"] == ["columbia", "harvard"]  # sorted keys
    assert out["catalog_count"] == 2


def test_to_canonical_100_style_identifiers_dict():
    rec = {
        "id": 5, "title": "T", "authors": ["A"], "language_code": "eng",
        "language": "English", "date": "2014",
        "identifiers": {"lccn": "987", "isbn": "111", "oclc": "222"},
        "ground_truth_lcsh": {"harvard": ["Art"]},
        "ground_truth_lcsh_merged": ["Art"],
    }
    out = to_canonical(rec, source="eng100")
    assert out["id"] == "eng100-5"
    assert out["lccn"] == "987" and out["isbn"] == "111" and out["oclc"] == "222"
    assert out["catalogs"] == ["harvard"]
    assert out["catalog_count"] == 1
