# src/lcsh_benchmark/score.py
"""Standalone benchmark scorer: submission + dataset -> metric panel."""
import argparse
import hashlib
import json
from collections import defaultdict

from .metrics import (keys, micro_prf, mrr, precision_at_k, prf, r_precision,
                      recall_at_k, root_key, norm_key)
from .normalize import normalize_label


def _key_fn(mode):
    return root_key if mode == "root" else norm_key


def score_generation(records: list[dict], predictions: dict[str, list[str]],
                     mode: str) -> dict:
    kf = _key_fn(mode)
    per_record: list[tuple[set, set]] = []
    macro: list[tuple[float, float, float]] = []
    by_lang: dict[str, list[tuple[set, set]]] = defaultdict(list)
    type_gt: dict[str, set] = defaultdict(set)
    type_hit: dict[str, set] = defaultdict(set)

    for r in records:
        gt = set(keys(r["ground_truth_lcsh_merged"], mode))
        pred = set(keys(predictions.get(r["id"], []), mode))
        per_record.append((gt, pred))
        macro.append(prf(gt, pred))
        by_lang[r["language_code"]].append((gt, pred))
        for h in r["ground_truth_lcsh_merged"]:
            ht = r["heading_types"][h]
            # v2 types by MARC tag ("topical"/"geographic"/"name"/"genre");
            # v1 typed by authority file ("lcsh"/"lcnaf"). Accept either.
            a = ht.get("type") or ht.get("authority")
            k = kf(h)
            type_gt[a].add(k)
            if k in pred:
                type_hit[a].add(k)

    macro_avg = tuple(sum(c) / len(macro) for c in zip(*macro)) if macro else (0.0, 0.0, 0.0)
    return {
        "n": len(records),
        "micro": micro_prf(per_record),
        "macro": macro_avg,
        "per_language": {l: micro_prf(v) for l, v in sorted(by_lang.items())},
        "per_type_recall": {
            a: (len(type_hit[a]) / len(type_gt[a]) if type_gt[a] else 0.0, len(type_gt[a]))
            for a in sorted(type_gt)
        },
    }


def score_selection(records: list[dict], predictions: dict[str, list[str]],
                    mode: str, ks: list[int]) -> dict:
    n = max(1, len(records))
    rsum = defaultdict(float)
    psum = defaultdict(float)
    rprec = mrr_sum = 0.0
    by_lang_recall = defaultdict(list)
    kmax = max(ks)

    for r in records:
        gt = set(keys(r["ground_truth_lcsh_merged"], mode))
        ranked = keys(predictions.get(r["id"], []), mode)
        for k in ks:
            rsum[k] += recall_at_k(gt, ranked, k)
            psum[k] += precision_at_k(gt, ranked, k)
        rprec += r_precision(gt, ranked)
        mrr_sum += mrr(gt, ranked)
        by_lang_recall[r["language_code"]].append(recall_at_k(gt, ranked, kmax))

    out = {"n": len(records), "mrr": mrr_sum / n, "r_precision": rprec / n}
    for k in ks:
        out[f"recall@{k}"] = rsum[k] / n
        out[f"p@{k}"] = psum[k] / n
    out["per_language_recall"] = {
        l: sum(v) / len(v) for l, v in sorted(by_lang_recall.items())
    }
    return out


def load_submission(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        sub = json.load(f)
    if "predictions" not in sub or not isinstance(sub["predictions"], dict):
        raise ValueError("submission must have a 'predictions' object")
    return sub


def hash_preds(headings: list[str]) -> list[str]:
    return [hashlib.sha256(normalize_label(h).encode()).hexdigest() for h in headings]


def score_generation_hashed(test_records: list[dict],
                            predictions: dict[str, list[str]],
                            hashed_gt: dict[str, list[str]]) -> dict:
    per_record: list[tuple[set, set]] = []
    by_lang: dict[str, list[tuple[set, set]]] = defaultdict(list)
    for r in test_records:
        gt = set(hashed_gt.get(r["id"], []))
        pred = set(hash_preds(predictions.get(r["id"], [])))
        per_record.append((gt, pred))
        by_lang[r["language_code"]].append((gt, pred))
    return {
        "n": len(test_records),
        "micro": micro_prf(per_record),
        "macro": tuple(sum(c) / len(per_record) for c in zip(*[prf(g, p) for g, p in per_record]))
        if per_record else (0.0, 0.0, 0.0),
        "per_language": {l: micro_prf(v) for l, v in sorted(by_lang.items())},
    }


def score_selection_hashed(test_records: list[dict],
                           predictions: dict[str, list[str]],
                           hashed_gt: dict[str, list[str]],
                           ks: list[int]) -> dict:
    """Rank metrics for the held-out test: hash the ranked predictions and score
    against the hashed GT set. Membership + order are all the rank metrics need,
    so this equals plaintext exact scoring. Exact-only (hashes carry no root)."""
    n = max(1, len(test_records))
    rsum: dict[int, float] = defaultdict(float)
    psum: dict[int, float] = defaultdict(float)
    rprec = mrr_sum = 0.0
    by_lang_recall: dict[str, list[float]] = defaultdict(list)
    kmax = max(ks)

    for r in test_records:
        gt = set(hashed_gt.get(r["id"], []))
        ranked = hash_preds(predictions.get(r["id"], []))
        for k in ks:
            rsum[k] += recall_at_k(gt, ranked, k)
            psum[k] += precision_at_k(gt, ranked, k)
        rprec += r_precision(gt, ranked)
        mrr_sum += mrr(gt, ranked)
        by_lang_recall[r["language_code"]].append(recall_at_k(gt, ranked, kmax))

    out = {"n": len(test_records), "mrr": mrr_sum / n, "r_precision": rprec / n}
    for k in ks:
        out[f"recall@{k}"] = rsum[k] / n
        out[f"p@{k}"] = psum[k] / n
    out["per_language_recall"] = {
        l: sum(v) / len(v) for l, v in sorted(by_lang_recall.items())
    }
    return out


def render_report(task: str, results: dict, exact_only: bool = False) -> str:
    """Format a results dict as a text report.

    exact_only suppresses the root column + exact-vs-root gap (used for the
    held-out test, where hashed GT has no root/type info).
    """
    lines = [f"# Scoring report — task: {task}", ""]
    ex, ro = results.get("exact", {}), results.get("root", {})
    if task == "generation":
        if exact_only:
            lines.append(f"{'metric':24s}{'exact':>10s}")
            for label, key, idx in (("micro P", "micro", 0), ("micro R", "micro", 1),
                                    ("micro F1", "micro", 2), ("macro P", "macro", 0),
                                    ("macro R", "macro", 1), ("macro F1", "macro", 2)):
                lines.append(f"{label:24s}{ex.get(key, (0, 0, 0))[idx]:>10.3f}")
        else:
            lines.append(f"{'metric':24s}{'exact':>10s}{'root':>10s}{'gap':>10s}")
            for label, key, idx in (("micro P", "micro", 0), ("micro R", "micro", 1),
                                    ("micro F1", "micro", 2), ("macro P", "macro", 0),
                                    ("macro R", "macro", 1), ("macro F1", "macro", 2)):
                e = ex.get(key, (0, 0, 0))[idx]
                r = ro.get(key, (0, 0, 0))[idx]
                lines.append(f"{label:24s}{e:>10.3f}{r:>10.3f}{r - e:>+10.3f}")
        ptr = ex.get("per_type_recall", {})
        if ptr:
            lines += ["", "## Per-type recall (exact) — 'lcsh' = topical core", ""]
            for a, (rec, n) in ptr.items():
                lines.append(f"  {a:10s} recall={rec:.3f}  (n={n})")
        lines += ["", "## Per-language recall (exact)", ""]
        for l, prf3 in ex.get("per_language", {}).items():
            lines.append(f"  {l:5s} R={prf3[1]:.3f}")
    else:  # selection
        ks = sorted(int(k.split("@")[1]) for k in ex if k.startswith("recall@"))
        if exact_only:
            lines.append(f"{'metric':16s}{'exact':>10s}")
            for label, mkey in (("MRR", "mrr"), ("R-Precision", "r_precision")):
                lines.append(f"{label:16s}{ex.get(mkey, 0):>10.3f}")
            for k in ks:
                for prefix in (f"recall@{k}", f"p@{k}"):
                    lines.append(f"{prefix:16s}{ex.get(prefix, 0):>10.3f}")
        else:
            lines.append(f"{'metric':16s}{'exact':>10s}{'root':>10s}{'gap':>10s}")
            for label, mkey in (("MRR", "mrr"), ("R-Precision", "r_precision")):
                e, r = ex.get(mkey, 0), ro.get(mkey, 0)
                lines.append(f"{label:16s}{e:>10.3f}{r:>10.3f}{r - e:>+10.3f}")
            for k in ks:
                for prefix in (f"recall@{k}", f"p@{k}"):
                    e, r = ex.get(prefix, 0), ro.get(prefix, 0)
                    lines.append(f"{prefix:16s}{e:>10.3f}{r:>10.3f}{r - e:>+10.3f}")
    return "\n".join(lines) + "\n"


KS = [5, 10, 50, 200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--hashed-gt", default=None, help="for held-out test scoring")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        records = json.load(f)
    sub = load_submission(args.submission)
    task = sub.get("task", "generation")
    preds = sub["predictions"]

    if args.hashed_gt:  # held-out test: exact mode only
        with open(args.hashed_gt, encoding="utf-8") as f:
            hashed = json.load(f)
        if task == "selection":
            results = {"exact": score_selection_hashed(records, preds, hashed, KS)}
        else:
            results = {"exact": score_generation_hashed(records, preds, hashed)}
    elif task == "selection":
        results = {m: score_selection(records, preds, m, KS) for m in ("exact", "root")}
    else:
        results = {m: score_generation(records, preds, m) for m in ("exact", "root")}

    report = render_report(task, results, exact_only=bool(args.hashed_gt))
    print(report)
    if args.out:
        if task == "generation" and not args.hashed_gt:
            from .ci import metric_cis
            gcis = metric_cis(records, preds, "generation")
            for mode in ("exact", "root"):
                results[mode]["cis"] = {"f1": list(gcis["f1"][mode])}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"system": sub.get("system", "?"), "task": task, "results": results},
                      f, ensure_ascii=False, indent=1)
