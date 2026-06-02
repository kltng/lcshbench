from lcsh_benchmark import ci
from lcsh_benchmark.retrieval.score_retrieval import VocabKeys

VOCAB = VocabKeys(exact={"a", "b"}, root={"a", "b"})


def _rec(rid, merged):
    return {"id": rid, "language_code": "eng",
            "ground_truth_lcsh_merged": merged,
            "heading_types": {h: {"type": "topical"} for h in merged}}


def test_selection_values_recall_and_mrr():
    recs = [_rec("r1", ["A", "B"])]
    preds = {"r1": ["A", "X", "B"]}
    assert ci.selection_values(recs, preds, "exact", "recall@10") == [1.0]
    assert ci.selection_values(recs, preds, "exact", "mrr") == [1.0]


def test_generation_micro_counts():
    recs = [_rec("r1", ["A", "B"])]
    preds = {"r1": ["A", "C"]}
    assert ci.generation_micro_counts(recs, preds, "exact") == [(1, 2, 2)]


def test_metric_cis_returns_point_lo_hi():
    recs = [_rec(f"r{i}", ["A", "B"]) for i in range(50)]
    preds = {f"r{i}": ["A", "B"] for i in range(50)}
    out = ci.metric_cis(recs, preds, task="selection", n_boot=200, seed=0, vocab=VOCAB)
    pt, lo, hi = out["recall@10"]["exact"]
    assert pt == 1.0 and lo == 1.0 and hi == 1.0


def test_compare_systems_significance():
    recs = [_rec(f"r{i}", ["A", "B"]) for i in range(80)]
    strong = {f"r{i}": ["A", "B"] for i in range(80)}
    weak = {f"r{i}": ["X", "Y"] for i in range(80)}
    res = ci.compare_systems(recs, strong, weak, "selection", "recall@10",
                             mode="exact", n_iter=2000, seed=0, vocab=VOCAB)
    assert res["delta"] > 0.9 and res["p_value"] < 0.01
