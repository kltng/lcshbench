from lcsh_benchmark.retrieval import score_retrieval as sr


def _rec(id, merged, lang="eng"):
    return {"id": id, "language_code": lang, "ground_truth_lcsh_merged": merged,
            "heading_types": {h: {"type": "topical"} for h in merged}}


VOCAB = sr.VocabKeys(exact={"music", "france"}, root={"music", "france"})


def test_reachable_exact_keeps_only_vocab_labels():
    rec = _rec("r1", ["Music", "Music--France--History", "Quux"])
    out, kept = sr.reachable([rec], VOCAB, "exact")
    assert out[0]["ground_truth_lcsh_merged"] == ["Music"]
    assert kept == 1


def test_reachable_root_keeps_subdivided_whose_base_is_in_vocab():
    rec = _rec("r1", ["Music--France--History", "Quux--Stuff"])
    out, kept = sr.reachable([rec], VOCAB, "root")
    assert out[0]["ground_truth_lcsh_merged"] == ["Music--France--History"]
    assert kept == 1


def test_score_reports_per_mode_reachable_and_total():
    recs = [_rec("r1", ["Music", "Music--France"])]
    preds = {"r1": ["Music", "France"]}
    out = sr.score(recs, preds, ks=[5], vocab=VOCAB)
    assert out["total_gt"] == 2
    assert out["exact"]["reachable_gt"] == 1
    assert out["root"]["reachable_gt"] == 2
    assert out["exact"]["recall@5"] == 1.0
