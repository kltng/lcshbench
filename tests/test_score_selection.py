from lcsh_benchmark.score import score_selection


def _rec(rid, lang, merged):
    return {"id": rid, "language_code": lang,
            "ground_truth_lcsh_merged": merged,
            "heading_types": {h: {"authority": "lcsh"} for h in merged}}


def test_score_selection_ranks():
    records = [_rec("r1", "eng", ["a", "b"])]
    preds = {"r1": ["a", "x", "b"]}
    out = score_selection(records, preds, "exact", ks=[1, 2, 3])
    assert out["recall@1"] == 0.5
    assert out["recall@3"] == 1.0
    assert out["mrr"] == 1.0
    assert round(out["r_precision"], 3) == 0.5
    assert "eng" in out["per_language_recall"]
