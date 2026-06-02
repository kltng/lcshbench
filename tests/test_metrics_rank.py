from lcsh_benchmark.metrics import recall_at_k, precision_at_k, r_precision, mrr


def test_recall_at_k():
    gt = {"a", "b", "c"}
    ranked = ["a", "x", "b", "y"]
    assert recall_at_k(gt, ranked, 2) == 1 / 3
    assert round(recall_at_k(gt, ranked, 3), 3) == 0.667


def test_precision_at_k_divides_by_k():
    gt = {"a", "b"}
    ranked = ["a", "x", "b"]
    assert precision_at_k(gt, ranked, 2) == 0.5


def test_r_precision_at_len_gt():
    gt = {"a", "b", "c"}
    ranked = ["a", "b", "x", "c"]
    assert round(r_precision(gt, ranked), 3) == 0.667


def test_mrr_first_hit_position():
    assert mrr({"b"}, ["a", "b", "c"]) == 0.5
    assert mrr({"z"}, ["a", "b", "c"]) == 0.0
