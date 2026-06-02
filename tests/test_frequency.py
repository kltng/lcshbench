import json

from lcsh_benchmark.baselines.frequency import (
    ranked_by_frequency, frequency_submission, run)


def _rec(rid, merged):
    return {"id": rid, "ground_truth_lcsh_merged": merged}


def test_ranked_by_frequency_orders_by_count_then_alpha():
    recs = [_rec("r1", ["Music", "Art"]), _rec("r2", ["Music", "Botany"])]
    assert ranked_by_frequency(recs) == ["Music", "Art", "Botany"]


def test_frequency_submission_predicts_topk_for_every_record():
    recs = [_rec("r1", ["Music"]), _rec("r2", ["Art"])]
    sub = frequency_submission(recs, k=1)
    assert sub["task"] == "selection"
    assert sub["predictions"] == {"r1": ["Art"], "r2": ["Art"]}
    assert sub["system"].startswith("frequency-floor")


def test_frequency_submission_can_rank_from_separate_set():
    """Held-out test has no GT: rank from dev, predict for the GT-less targets."""
    train = [_rec("t1", ["Music", "Art"]), _rec("t2", ["Music"])]
    targets = [{"id": "x1"}, {"id": "x2"}]            # no ground_truth_lcsh_merged
    sub = frequency_submission(targets, k=2, rank_from=train)
    assert sub["predictions"] == {"x1": ["Music", "Art"], "x2": ["Music", "Art"]}


def test_run_reads_dataset_and_writes_submission(tmp_path):
    ds = tmp_path / "dev.json"
    ds.write_text(json.dumps([_rec("r1", ["Music", "Art"]), _rec("r2", ["Music"])]))
    out = tmp_path / "preds.json"
    sub = run(str(ds), str(out), k=2)
    written = json.loads(out.read_text())
    assert written == sub
    assert written["predictions"]["r1"] == ["Music", "Art"]
    assert written["system"] == "frequency-floor-top2"
