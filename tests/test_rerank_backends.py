from lcsh_benchmark.retrieval.rerank_backends import FakeReranker


def test_fake_reranker_orders_by_lexical_overlap():
    # FakeReranker scores by shared-token overlap with the query (deterministic).
    rr = FakeReranker()
    ranked = rr.rerank("music history of france",
                       ["Cooking", "Music--History", "France--History"])
    assert ranked[0] in ("Music--History", "France--History")
    assert ranked[-1] == "Cooking"            # no overlap -> last
    assert set(ranked) == {"Cooking", "Music--History", "France--History"}
    assert rr.name == "fake-reranker"
