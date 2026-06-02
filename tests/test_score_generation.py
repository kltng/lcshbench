from lcsh_benchmark.score import score_generation


def _rec(rid, lang, merged, auth):
    return {"id": rid, "language_code": lang,
            "ground_truth_lcsh_merged": merged,
            "heading_types": {h: {"authority": auth[h]} for h in merged}}


def test_score_generation_exact_perfect_and_typed():
    records = [_rec("r1", "eng", ["Sociology", "France"],
                    {"Sociology": "lcsh", "France": "lcnaf"})]
    preds = {"r1": ["Sociology", "France"]}
    out = score_generation(records, preds, "exact")
    assert out["micro"] == (1.0, 1.0, 1.0)
    assert out["per_language"]["eng"][1] == 1.0
    assert out["per_type_recall"]["lcsh"] == (1.0, 1)
    assert out["per_type_recall"]["lcnaf"] == (1.0, 1)


def test_score_generation_partial_and_missing_record():
    records = [_rec("r1", "eng", ["Sociology", "Music"],
                    {"Sociology": "lcsh", "Music": "lcsh"})]
    preds = {}
    out = score_generation(records, preds, "exact")
    assert out["micro"] == (0.0, 0.0, 0.0)
    assert out["per_type_recall"]["lcsh"] == (0.0, 2)
