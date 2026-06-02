from lcsh_benchmark.scaleup.sample import stratified_sample, monograph_only


def test_monograph_only_keeps_books_drops_other_material():
    recs = [
        {"key": "o:1", "is_monograph": True},
        {"key": "o:2", "is_monograph": False},   # serial/map/score/etc.
        {"key": "o:3", "is_monograph": True},
        {"key": "o:4"},                            # missing flag -> excluded
    ]
    out = monograph_only(recs)
    assert [r["key"] for r in out] == ["o:1", "o:3"]


def _r(i, lang, cls, has_input=True):
    return {"key": f"o:{i}", "lang": lang, "lc_class": cls,
            "has_input": has_input, "ground_truth_lcsh": {"a": ["X"], "b": ["X"]},
            "catalog_count": 2}


def test_stratified_sample_respects_targets_music_cap_and_disjoint():
    pool = [_r(i, "eng", "M") for i in range(60)] + [_r(100 + i, "eng", "P") for i in range(40)]
    core, breadth = stratified_sample(
        pool, core_targets={"eng": 10}, breadth_targets={"eng": 10},
        music_cap=0.15, seed=13)
    assert len(core) == 10
    assert sum(1 for r in core if r["lc_class"] == "M") <= 2
    assert not ({r["key"] for r in core} & {r["key"] for r in breadth})


def test_core_prefers_records_with_input():
    pool = [_r(i, "eng", "P", has_input=(i < 5)) for i in range(20)]
    core, _ = stratified_sample(pool, {"eng": 5}, {"eng": 0}, music_cap=0.15, seed=13)
    assert all(r["has_input"] for r in core)


def test_music_cap_holds_even_when_pool_is_all_music():
    pool = [{"key": f"o:{i}", "lang": "eng", "lc_class": "M",
             "has_input": True, "ground_truth_lcsh": {"a": ["X"], "b": ["X"]},
             "catalog_count": 2} for i in range(100)]
    core, _ = stratified_sample(pool, {"eng": 10}, {"eng": 0}, music_cap=0.15, seed=13)
    assert sum(1 for r in core if r["lc_class"] == "M") <= int(10 * 0.15)  # <= 1
