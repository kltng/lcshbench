import numpy as np
from lcsh_benchmark.retrieval import retrieve
from lcsh_benchmark.retrieval.embed_index import Index


def test_build_query_text_folds_fields():
    rec = {"title": "Sociology of Music", "authors": ["Doe, Jane"],
           "abstract": "An abstract.", "toc": "Ch1\nCh2",
           "title_vernacular": "音楽社会学"}
    q = retrieve.build_query_text(rec)
    assert "Sociology of Music" in q and "Doe, Jane" in q
    assert "An abstract." in q and "音楽社会学" in q


def test_topk_ranks_by_cosine():
    # 3-label index; query identical to label 1 -> it ranks first.
    vecs = np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = Index(vecs, ["A", "B", "C"], ["u/a", "u/b", "u/c"],
                ["lcsh", "lcsh", "lcgft"])
    qv = np.array([0, 1], dtype=np.float32)
    ranked = retrieve.top_k(idx, qv, k=2)
    assert ranked[0] == "B"
    assert len(ranked) == 2


def test_run_emits_selection_submission(tmp_path):
    vecs = np.eye(3, dtype=np.float32)
    idx = Index(vecs, ["A", "B", "C"], ["u/a", "u/b", "u/c"],
                ["lcsh", "lcsh", "lcsh"])

    class StubBackend:
        name = "stub"
        def encode(self, texts):
            # one query -> aligns with label "C"
            return np.array([[0, 0, 1]], dtype=np.float32)
        def cost_usd(self, texts):
            return 0.0

    records = [{"id": "r1", "title": "x", "authors": [], "language_code": "eng"}]
    sub = retrieve.run(records, idx, StubBackend(), k=3, system="stub")
    assert sub["task"] == "selection"
    assert sub["predictions"]["r1"][0] == "C"
