from lcsh_benchmark.stats import bootstrap_mean_ci


def test_bootstrap_mean_ci_constant_is_degenerate():
    point, lo, hi = bootstrap_mean_ci([0.5] * 200, n_boot=500, seed=1)
    assert point == 0.5 and lo == 0.5 and hi == 0.5


def test_bootstrap_mean_ci_brackets_point_and_is_deterministic():
    vals = [0.0, 1.0] * 100          # mean 0.5
    p1 = bootstrap_mean_ci(vals, n_boot=1000, seed=7)
    p2 = bootstrap_mean_ci(vals, n_boot=1000, seed=7)
    assert p1 == p2
    point, lo, hi = p1
    assert abs(point - 0.5) < 1e-9
    assert lo < point < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_mean_ci_empty():
    assert bootstrap_mean_ci([], n_boot=10, seed=0) == (0.0, 0.0, 0.0)


from lcsh_benchmark.stats import bootstrap_micro_ci


def test_bootstrap_micro_ci_perfect():
    counts = [(2, 2, 2)] * 50
    for metric in ("p", "r", "f1"):
        point, lo, hi = bootstrap_micro_ci(counts, metric, n_boot=300, seed=3)
        assert abs(point - 1.0) < 1e-9 and lo == 1.0 and hi == 1.0


def test_bootstrap_micro_ci_half_recall_brackets():
    counts = [(1, 2, 2)] * 100 + [(0, 0, 2)] * 100
    point, lo, hi = bootstrap_micro_ci(counts, "f1", n_boot=800, seed=5)
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0
    assert abs(point - 1 / 3) < 1e-6     # micro F1 = 2*0.5*0.25/(0.5+0.25)


from lcsh_benchmark.stats import paired_randomization_pvalue


def test_paired_pvalue_identical_is_high():
    a = [0.3, 0.5, 0.7, 0.2] * 25
    assert paired_randomization_pvalue(a, list(a), n_iter=2000, seed=0) > 0.9


def test_paired_pvalue_large_separation_is_low():
    a = [0.9] * 100
    b = [0.1] * 100
    assert paired_randomization_pvalue(a, b, n_iter=2000, seed=0) < 0.01


def test_bootstrap_micro_ci_empty():
    assert bootstrap_micro_ci([], "f1", n_boot=10, seed=0) == (0.0, 0.0, 0.0)


def test_paired_pvalue_empty_is_one():
    assert paired_randomization_pvalue([], [], n_iter=10, seed=0) == 1.0
