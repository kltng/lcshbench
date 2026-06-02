from lcsh_benchmark.chat_backend import FakeChat
from lcsh_benchmark.retrieval import select


def test_select_emits_generation_submission_subset_of_candidates():
    fc = FakeChat(reply='["Music"]')      # model picks the final set to assign
    records = [{"id": "r1", "title": "t", "authors": [], "language_code": "eng"}]
    l1 = {"system": "ret", "predictions": {"r1": ["Music", "France--History", "Z"]}}
    out = select.run(records, l1, fc, top_n=3, system="llm-sel")
    assert out["task"] == "generation"               # a set, scored as Task A
    assert out["predictions"]["r1"] == ["Music"]
    # invented headings are rejected (must come from candidates)
    fc2 = FakeChat(reply='["Invented Heading"]', name="fake2")
    out2 = select.run(records, l1, fc2, top_n=3, system="x")
    assert out2["predictions"]["r1"] == []
