import hashlib
from lcsh_benchmark.score import (hash_preds, score_generation_hashed,
                                  score_selection, score_selection_hashed)
from lcsh_benchmark.build import hash_gt


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def test_hash_preds_matches_normalized_sha256():
    assert hash_preds(["Sociology."]) == [_h("sociology")]


def test_score_generation_hashed_pr():
    hashed_gt = {"r1": [_h("sociology"), _h("music")]}
    test_records = [{"id": "r1", "language_code": "eng"}]
    preds = {"r1": ["Sociology"]}
    out = score_generation_hashed(test_records, preds, hashed_gt)
    assert out["micro"][0] == 1.0 and out["micro"][1] == 0.5
    assert out["per_language"]["eng"][1] == 0.5


def test_score_selection_hashed_equals_plaintext_exact():
    """Rank metrics only need set membership + order, so scoring hashed GT vs
    hashed predictions must match plaintext exact selection scoring exactly."""
    records = [
        {"id": "r1", "language_code": "eng", "ground_truth_lcsh_merged": ["Music", "Art"]},
        {"id": "r2", "language_code": "fre", "ground_truth_lcsh_merged": ["Botany"]},
    ]
    preds = {"r1": ["Music", "Noise", "Art"], "r2": ["Botany", "Weeds"]}
    ks = [5, 10]
    plain = score_selection(records, preds, "exact", ks)

    hashed_gt = {r["id"]: hash_gt(r["ground_truth_lcsh_merged"]) for r in records}
    test_records = [{"id": r["id"], "language_code": r["language_code"]} for r in records]
    hashed = score_selection_hashed(test_records, preds, hashed_gt, ks)

    for key in ("mrr", "r_precision", "recall@5", "p@5", "recall@10", "p@10"):
        assert abs(hashed[key] - plain[key]) < 1e-9, key
    assert hashed["per_language_recall"]["eng"] == plain["per_language_recall"]["eng"]
