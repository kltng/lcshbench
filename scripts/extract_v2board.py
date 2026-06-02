"""Derive every v2-leaderboard.md cell from data, for v2.1 re-score.
Usage: uv run python scripts/extract_v2board.py
Prints a JSON blob with all summary/selection/per-type/per-language cells.
"""
import json
import sys

from lcsh_benchmark.score import score_generation, score_selection

KS = [5, 10, 50, 200]
DS = "data/v2/dev/dataset_dev_subset2k.json"
SUBS = {
    "frequency": "results/runs/frequency_v2_dev_subset2k.json",
    "onnx": "results/runs/lcsh_onnx_v2_dev_subset2k.json",
    "vern": "results/runs/lcsh_onnx_vern_v2_dev_subset2k.json",
    "distill": "results/runs/lcsh_onnx_distill_v2_dev_subset2k.json",
}
records = json.load(open(DS, encoding="utf-8"))


def trunc(preds, k):
    return {rid: p[:k] for rid, p in preds.items()}


out = {"n_records": len(records)}
# per-type n only needs one pass (GT is shared); collect from first available
for name, path in SUBS.items():
    try:
        sub = json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        out[name] = {"_missing": True}
        continue
    preds = sub["predictions"]
    gen10_ex = score_generation(records, trunc(preds, 10), "exact")
    gen10_ro = score_generation(records, trunc(preds, 10), "root")
    sel_ex = score_selection(records, preds, "exact", KS)
    sel_ro = score_selection(records, preds, "root", KS)
    ptr = gen10_ex["per_type_recall"]
    out[name] = {
        "system": sub.get("system"),
        "gen_f1_exact": round(gen10_ex["micro"][2], 3),
        "gen_f1_root": round(gen10_ro["micro"][2], 3),
        "core_topical": round((ptr.get("topical") or (0, 0))[0], 3),
        "summary_recall@10": round(sel_ex["recall@10"], 3),
        "summary_recall@200": round(sel_ex["recall@200"], 3),
        "summary_mrr": round(sel_ex["mrr"], 3),
        "sel_exact": {m: round(sel_ex[m], 3) for m in
                      ["mrr", "r_precision", "recall@10", "recall@50", "recall@200"]},
        "sel_root": {m: round(sel_ro[m], 3) for m in
                     ["mrr", "r_precision", "recall@10", "recall@50", "recall@200"]},
        "per_type_exact_top10": {t: (round(v[0], 3), v[1]) for t, v in ptr.items()},
        "per_language_recall@200_exact": {l: round(v, 3) for l, v in sel_ex["per_language_recall"].items()},
        "per_language_recall@200_root": {l: round(v, 3) for l, v in sel_ro["per_language_recall"].items()},
    }

json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
print()
