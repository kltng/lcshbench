from lcsh_benchmark.scaleup import composition as C


def test_targets_sum_close_to_5k():
    assert 2900 <= sum(C.CORE_LANG_TARGETS.values()) <= 3100
    assert 1900 <= sum(C.BREADTH_LANG_TARGETS.values()) <= 2100
    assert 0 < C.MUSIC_CAP < 1
    assert C.MUSIC_CLASS == "M"
    assert C.SEED == 13
