from lcsh_benchmark.consensus import catalog_count, merge_union


def test_catalog_count_counts_catalogs_with_any_heading():
    gt = {"harvard": ["Sociology"], "columbia": ["Sociology", "Music"], "loc": []}
    assert catalog_count(gt) == 2  # loc has none


def test_merge_union_dedupes_case_insensitively_keeps_first_surface():
    gt = {"harvard": ["Sociology", "Music"], "columbia": ["sociology", "Art"]}
    assert merge_union(gt) == ["Art", "Music", "Sociology"]
