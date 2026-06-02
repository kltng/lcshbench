from lcsh_benchmark.metrics import keys, prf, micro_prf


def test_keys_exact_vs_root():
    hs = ["Sociology--Research", "Music"]
    assert keys(hs, "exact") == ["sociology--research", "music"]
    assert keys(hs, "root") == ["sociology", "music"]


def test_prf_basic():
    gt = {"a", "b", "c"}
    pred = {"a", "b", "x"}
    p, r, f1 = prf(gt, pred)
    assert round(p, 3) == 0.667 and round(r, 3) == 0.667 and round(f1, 3) == 0.667


def test_prf_empty_prediction_is_zero():
    assert prf({"a"}, set()) == (0.0, 0.0, 0.0)


def test_micro_prf_pools_counts():
    p, r, f1 = micro_prf([({"a", "b"}, {"a"}), ({"c"}, {"c", "d"})])
    assert round(p, 3) == 0.667 and round(r, 3) == 0.667
