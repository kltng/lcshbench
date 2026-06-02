from lcsh_benchmark.gt_tiers import heading_tiers


def test_heading_tiers_counts():
    rec = {"ground_truth_lcsh": {
        "harvard": ["A", "B"], "columbia": ["A", "B"], "princeton": ["A", "C"]}}
    t = heading_tiers(rec)
    assert t["unanimous"] == ["A"]
    assert sorted(t["majority"]) == ["A", "B"]
    assert sorted(t["single"]) == ["C"]
