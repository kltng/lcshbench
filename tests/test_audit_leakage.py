from lcsh_benchmark.audit_leakage import audit
from lcsh_benchmark.retrieval.score_retrieval import VocabKeys


def test_audit_flags_unreachable_and_nonlatin():
    recs = [{"id": "r1", "heading_types": {"Music": {"type": "topical"},
                                           "Música": {"type": "topical"},
                                           "Жзл": {"type": "topical"}},
             "ground_truth_lcsh_merged": ["Music", "Música", "Жзл"]}]
    vocab = VocabKeys(exact={"music"}, root={"music"})
    rep = audit(recs, vocab)
    assert rep["total"] == 3
    assert rep["root_unreachable"] == 2
    assert rep["non_latin"] == 1
    assert 0.66 < rep["root_unreachable_frac"] < 0.67
