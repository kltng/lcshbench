from lcsh_benchmark.retrieval import rerank
from lcsh_benchmark.retrieval.rerank_backends import FakeReranker


def test_rerank_reorders_top_n_per_record():
    records = [{"id": "r1", "title": "music history", "authors": [],
                "language_code": "eng"}]
    l1 = {"system": "ret", "task": "selection",
          "predictions": {"r1": ["Cooking", "Music--History", "Physics", "France"]}}
    out = rerank.run(records, l1, FakeReranker(), top_n=4, system="rr")
    assert out["task"] == "selection"
    assert out["predictions"]["r1"][0] == "Music--History"   # lexical overlap wins
    assert out["system"] == "rr"


def test_rerank_only_touches_top_n_keeps_tail_order():
    records = [{"id": "r1", "title": "music", "authors": [], "language_code": "eng"}]
    l1 = {"predictions": {"r1": ["A", "Music", "B", "C", "D"]}}
    out = rerank.run(records, l1, FakeReranker(), top_n=2, system="rr")
    # only first 2 reranked; tail [B,C,D] appended in original order
    assert out["predictions"]["r1"][2:] == ["B", "C", "D"]
