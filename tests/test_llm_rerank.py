from lcsh_benchmark.chat_backend import FakeChat
from lcsh_benchmark.retrieval import llm_rerank


def test_llm_rerank_reorders_within_candidates_only():
    # model returns a reordering; anything it invents is dropped, missing appended.
    fc = FakeChat(reply='["France--History", "Music"]')
    records = [{"id": "r1", "title": "t", "authors": [], "language_code": "eng"}]
    l1 = {"system": "ret", "predictions": {"r1": ["Music", "France--History", "Z"]}}
    out = llm_rerank.run(records, l1, fc, top_n=3, system="llm-rr")
    assert out["task"] == "selection"
    assert out["predictions"]["r1"][:2] == ["France--History", "Music"]
    assert "Z" in out["predictions"]["r1"]      # untouched candidate retained at tail
