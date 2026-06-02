from lcsh_benchmark.seed_variance import variance_across_runs


def test_variance_across_runs_zero_when_identical():
    runs = [{"f1": 0.30}, {"f1": 0.30}, {"f1": 0.30}]
    out = variance_across_runs(runs)
    assert out["f1"]["mean"] == 0.30 and out["f1"]["std"] == 0.0


def test_variance_across_runs_reports_spread():
    runs = [{"f1": 0.28}, {"f1": 0.30}, {"f1": 0.32}]
    out = variance_across_runs(runs)
    assert abs(out["f1"]["mean"] - 0.30) < 1e-9
    assert out["f1"]["std"] > 0.0
