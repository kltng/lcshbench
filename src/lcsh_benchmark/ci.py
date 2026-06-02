"""Confidence intervals for a submission's metrics, reusing metrics.py for the
per-record values and stats.py for the bootstrap. Selection metrics use vocab
reachability filtering (Task B), matching score_retrieval."""
from .metrics import keys, mrr, precision_at_k, r_precision, recall_at_k
from .stats import bootstrap_mean_ci, bootstrap_micro_ci, paired_randomization_pvalue
from .retrieval.score_retrieval import reachable, load_vocab_keys, VocabKeys, DEFAULT_VOCAB

SELECTION_METRICS = ["recall@10", "recall@50", "recall@200", "mrr"]
GENERATION_METRICS = ["f1"]


def _sel_one(gt, ranked, metric):
    if metric.startswith("recall@"):
        return recall_at_k(gt, ranked, int(metric.split("@")[1]))
    if metric.startswith("p@"):
        return precision_at_k(gt, ranked, int(metric.split("@")[1]))
    if metric == "mrr":
        return mrr(gt, ranked)
    if metric == "r_precision":
        return r_precision(gt, ranked)
    raise ValueError(metric)


def selection_values(records, predictions, mode, metric):
    out = []
    for r in records:
        gt = set(keys(r["ground_truth_lcsh_merged"], mode))
        ranked = keys(predictions.get(r["id"], []), mode)
        out.append(_sel_one(gt, ranked, metric))
    return out


def generation_micro_counts(records, predictions, mode):
    out = []
    for r in records:
        gt = set(keys(r["ground_truth_lcsh_merged"], mode))
        pred = set(keys(predictions.get(r["id"], []), mode))
        out.append((len(gt & pred), len(pred), len(gt)))
    return out


def compare_systems(records, preds_a, preds_b, task, metric, mode="exact",
                    n_iter=10000, seed=0, vocab=None, vocab_path=DEFAULT_VOCAB):
    """Paired significance of (A - B) on a per-record metric. selection only
    for now (per-record values); generation comparison can be added later."""
    if task != "selection":
        raise NotImplementedError(
            "compare_systems currently supports task='selection' only")
    import numpy as np
    vocab = vocab or load_vocab_keys(vocab_path)
    recs, _ = reachable(records, vocab, mode)
    a = selection_values(recs, preds_a, mode, metric)
    b = selection_values(recs, preds_b, mode, metric)
    delta = float(np.mean(a) - np.mean(b))
    p = paired_randomization_pvalue(a, b, n_iter=n_iter, seed=seed)
    return {"metric": metric, "mode": mode, "delta": delta, "p_value": p}


def main() -> None:
    import argparse
    import json
    from .leaderboard import fmt_ci
    from .score import load_submission
    ap = argparse.ArgumentParser(description="CIs for a submission; optional A/B significance.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--vs", default=None, help="second submission for significance")
    ap.add_argument("--metric", default="recall@10")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    with open(a.dataset, encoding="utf-8") as f:
        records = json.load(f)
    sub = load_submission(a.submission)
    task = sub.get("task", "generation")
    cis = metric_cis(records, sub["predictions"], task, n_boot=a.n_boot)
    print(f"# CIs — {sub.get('system','?')} ({task})")
    for m, modes in cis.items():
        for mode, triple in modes.items():
            print(f"  {m:12s} {mode:5s} {fmt_ci(triple)}")
    if a.vs:
        other = load_submission(a.vs)
        res = compare_systems(records, sub["predictions"], other["predictions"],
                              task, a.metric)
        print(f"\n# {sub.get('system','A')} vs {other.get('system','B')} on {a.metric}:")
        print(f"  delta={res['delta']:+.3f}  p={res['p_value']:.4f}")


def metric_cis(records, predictions, task, n_boot=1000, seed=0,
               vocab=None, vocab_path=DEFAULT_VOCAB):
    """{metric: {mode: (point, lo, hi)}} for the headline metrics of `task`."""
    out = {}
    if task == "selection":
        vocab = vocab or load_vocab_keys(vocab_path)
        for m in SELECTION_METRICS:
            out[m] = {}
            for mode in ("exact", "root"):
                recs, _ = reachable(records, vocab, mode)
                out[m][mode] = bootstrap_mean_ci(
                    selection_values(recs, predictions, mode, m),
                    n_boot=n_boot, seed=seed)
    else:
        for m in GENERATION_METRICS:
            out[m] = {mode: bootstrap_micro_ci(
                generation_micro_counts(records, predictions, mode), m,
                n_boot=n_boot, seed=seed)
                for mode in ("exact", "root")}
    return out
