from lcsh_benchmark.leaderboard import leaderboard_row, render_leaderboard


def _rec(rid, merged):
    return {"id": rid, "language_code": "eng",
            "ground_truth_lcsh_merged": merged,
            "heading_types": {h: {"authority": "lcsh"} for h in merged}}


def test_leaderboard_row_headline_numbers():
    records = [_rec("r1", ["a", "b"])]
    sub = {"system": "perfect", "task": "selection", "predictions": {"r1": ["a", "b"]}}
    row = leaderboard_row(records, sub, gen_topk=10)
    assert row["system"] == "perfect"
    assert row["gen_f1_exact"] == 1.0
    assert row["core_recall_exact"] == 1.0
    assert row["recall@200_exact"] == 1.0
    assert row["mrr_exact"] == 1.0


def test_render_leaderboard_is_markdown_table():
    rows = [{"system": "s1", "gen_f1_exact": 0.5, "gen_f1_root": 0.6,
             "core_recall_exact": 0.4, "recall@10_exact": 0.3,
             "recall@200_exact": 0.7, "mrr_exact": 0.2}]
    md = render_leaderboard(rows)
    assert md.startswith("|") and "s1" in md and "gen_f1_exact" in md


from lcsh_benchmark.leaderboard import render_l1_board


def test_render_l1_board_sorts_by_recall_at_10_desc():
    # sentinel names that cannot collide with header words like "recall"
    rows = [
        {"system": "ZZA", "results": {"exact": {"recall@10": 0.40, "mrr": 0.3,
                                                "recall@200": 0.8}}},
        {"system": "ZZB", "results": {"exact": {"recall@10": 0.55, "mrr": 0.4,
                                                "recall@200": 0.9}}},
    ]
    md = render_l1_board(rows)
    assert md.index("ZZB") < md.index("ZZA")     # higher recall@10 first
    assert "recall@10" in md and "recall@200" in md


from lcsh_benchmark.leaderboard import render_l2_board


def test_render_l2_board_shows_delta_vs_baseline():
    baseline = {"system": "te3", "results": {"exact": {"recall@10": 0.10, "mrr": 0.12}}}
    reranked = {"system": "te3+rerank-jina",
                "results": {"exact": {"recall@10": 0.18, "mrr": 0.20}}}
    md = render_l2_board(baseline, [reranked])
    assert "+0.08" in md          # recall@10 delta shown
    assert "te3+rerank-jina" in md


from lcsh_benchmark.leaderboard import render_generation_board


def test_render_generation_board_sorts_by_f1():
    rows = [
        {"system": "m1", "results": {"exact": {"micro": [0.4, 0.3, 0.34]},
                                     "root": {"micro": [0.5, 0.4, 0.44]}}},
        {"system": "m2", "results": {"exact": {"micro": [0.6, 0.5, 0.55]},
                                     "root": {"micro": [0.7, 0.6, 0.65]}}},
    ]
    md = render_generation_board(rows)
    assert md.index("m2") < md.index("m1")    # higher exact F1 first
    assert "F1" in md


import json as _json
from lcsh_benchmark.leaderboard import assemble_leaderboard


def test_assemble_leaderboard_includes_present_boards(tmp_path):
    runs = tmp_path / "runs"; runs.mkdir()
    (runs / "retrieval_x_dev2k.score.json").write_text(_json.dumps(
        {"system": "ret-x", "results": {"exact": {"recall@10": 0.1, "recall@50": 0.2,
                                                  "recall@200": 0.3, "mrr": 0.1}}}))
    (runs / "gen_y_dev2k.score.json").write_text(_json.dumps(
        {"system": "gen-y", "results": {"exact": {"micro": [0.4, 0.3, 0.34]},
                                        "root": {"micro": [0.5, 0.4, 0.44]}}}))
    md = assemble_leaderboard(str(runs))
    assert "L1 — Embedding retrieval" in md and "ret-x" in md
    assert "Task A — Generation" in md and "gen-y" in md
    assert "L2 — Cross-encoder rerank" not in md   # omitted when absent


from lcsh_benchmark.leaderboard import fmt_ci


def test_fmt_ci():
    assert fmt_ci((0.218, 0.201, 0.235)) == "0.218 [0.201, 0.235]"
    assert fmt_ci((0.5, 0.5, 0.5)) == "0.500 [0.500, 0.500]"


def test_render_l1_board_uses_cis_when_present():
    rows = [{"system": "x", "results": {
        "exact": {"recall@10": 0.10, "recall@50": 0.2, "recall@200": 0.3, "mrr": 0.12,
                  "cis": {"recall@10": [0.10, 0.08, 0.12], "mrr": [0.12, 0.10, 0.14]}}}}]
    md = render_l1_board(rows)
    assert "[0.080, 0.120]" in md


def test_render_generation_board_uses_cis_when_present():
    rows = [{"system": "g", "results": {
        "exact": {"micro": [0.4, 0.3, 0.34], "cis": {"f1": [0.34, 0.31, 0.37]}},
        "root": {"micro": [0.5, 0.4, 0.44]}}}]
    md = render_generation_board(rows)
    assert "[0.310, 0.370]" in md
