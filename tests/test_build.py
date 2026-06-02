from lcsh_benchmark.build import process_record


def test_process_record_cleans_merges_and_types():
    lookup = {"sociology": "lcsh", "france": "lcnaf"}.get
    rec = {
        "id": "eng100-1", "title": "T", "authors": [], "lccn": "", "isbn": "",
        "oclc": "", "language_code": "eng", "language": "English", "date": "",
        "publisher": "", "physical_description": "", "abstract": "", "toc": "",
        "notes": "", "genres": [], "catalogs": ["harvard", "columbia"],
        "catalog_count": 2,
        "ground_truth_lcsh": {
            "harvard": ["Sociology", "653-11/"],
            "columbia": ["Sociology--Research", "France"],
        },
        "ground_truth_lcsh_merged": [],
    }
    out, dropped = process_record(rec, lookup)
    assert "653-11/" not in out["ground_truth_lcsh"]["harvard"]
    assert dropped == ["653-11/"]
    assert out["ground_truth_lcsh_merged"] == ["France", "Sociology", "Sociology--Research"]
    assert out["heading_types"]["Sociology--Research"]["authority"] == "lcsh"
    assert out["heading_types"]["France"]["authority"] == "lcnaf"
