from lcsh_benchmark.clean import is_marc_artifact, clean_headings


def test_detects_marc_field_tag_artifacts():
    assert is_marc_artifact("653-11/")
    assert is_marc_artifact("880-12 -- Shahon")
    assert not is_marc_artifact("Sociology--Research")
    assert not is_marc_artifact("20th century")  # leading digits but not NNN[-/]


def test_clean_headings_partitions_kept_and_dropped():
    kept, dropped = clean_headings(["Sociology", "653-11/", "Music--History"])
    assert kept == ["Sociology", "Music--History"]
    assert dropped == ["653-11/"]
