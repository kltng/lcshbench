# src/lcsh_benchmark/leaderboard.py
"""Score a set of submissions and render a leaderboard comparison table."""
import argparse
import glob as _glob
import json

from .score import load_submission, score_generation, score_selection

KS = [5, 10, 50, 200]


def fmt_ci(point_lo_hi) -> str:
    """Format a (point, lo, hi) triple as 'point [lo, hi]'."""
    p, lo, hi = point_lo_hi
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


COLUMNS = ["system", "gen_f1_exact", "gen_f1_root", "core_recall_exact",
           "recall@10_exact", "recall@200_exact", "mrr_exact"]


def _truncate(predictions: dict[str, list[str]], k: int) -> dict[str, list[str]]:
    return {rid: preds[:k] for rid, preds in predictions.items()}


def leaderboard_row(records: list[dict], submission: dict, gen_topk: int) -> dict:
    preds = submission["predictions"]
    gen_ex = score_generation(records, _truncate(preds, gen_topk), "exact")
    gen_ro = score_generation(records, _truncate(preds, gen_topk), "root")
    sel_ex = score_selection(records, preds, "exact", KS)
    # Topical-core recall key: v2 = "topical" (MARC 650), v1 = "lcsh" (authority).
    ptr = gen_ex["per_type_recall"]
    core = ptr.get("topical") or ptr.get("lcsh") or (0.0, 0)
    return {
        "system": submission.get("system", "?"),
        "gen_f1_exact": round(gen_ex["micro"][2], 3),
        "gen_f1_root": round(gen_ro["micro"][2], 3),
        "core_recall_exact": round(core[0], 3),
        "recall@10_exact": round(sel_ex["recall@10"], 3),
        "recall@200_exact": round(sel_ex["recall@200"], 3),
        "mrr_exact": round(sel_ex["mrr"], 3),
    }


def _load(paths: str) -> list[dict]:
    return [json.load(open(p, encoding="utf-8")) for p in sorted(_glob.glob(paths))]


def assemble_leaderboard(runs_dir: str) -> str:
    """One page from the committed *.score.json artifacts. Omits empty boards."""
    sections = []
    gen = _load(f"{runs_dir}/gen_*dev2k.score.json")
    if gen:
        sections.append("## Task A — Generation\n\n" + render_generation_board(gen))
    l1 = (_load(f"{runs_dir}/retrieval_*dev2k.score.json")
          + _load(f"{runs_dir}/*retrieval-score.json"))
    if l1:
        rows = [{"system": d["system"], "results": d["results"]} for d in l1]
        sections.append("## L1 — Embedding retrieval\n\n" + render_l1_board(rows))
    rr = _load(f"{runs_dir}/rerank_*dev2k.score.json")
    if rr:
        base = _load(f"{runs_dir}/retrieval_te3-small_dev2k.score.json")
        if base:
            sections.append("## L2 — Cross-encoder rerank\n\n"
                            + render_l2_board(base[0], rr))
    l3 = _load(f"{runs_dir}/llmrr_*dev2k.score.json")
    if l3:
        rows = [{"system": d["system"], "results": d["results"]} for d in l3]
        sections.append("## L3 — LLM rerank\n\n" + render_l1_board(rows))
    sel = _load(f"{runs_dir}/sel_*dev2k.score.json")
    if sel:
        sections.append("## L4 — Final-cut selection\n\n"
                        + render_generation_board(sel))
    header = ("# LCSH benchmark leaderboard (dev-2K)\n\n"
              "Task B layers (L1–L4) are scored on the in-vocab GT subset "
              "(topical+geographic+genre). Generation/L4 use set P/R/F1; "
              "L1–L3 use rank metrics.\n\n")
    return header + "\n".join(sections) + "\n"


def render_generation_board(rows: list[dict]) -> str:
    """Task A / L4 board: micro P/R/F1 (exact + root F1), sorted by exact F1."""
    def f1(r):
        return r["results"]["exact"]["micro"][2]
    ordered = sorted(rows, key=f1, reverse=True)
    lines = ["| system | P | R | F1 (exact) | F1 (root) |", "|---|---|---|---|---|"]
    for r in ordered:
        ex = r["results"]["exact"]["micro"]
        ro = r["results"]["root"]["micro"]
        cis = r["results"]["exact"].get("cis", {})
        f1cell = fmt_ci(cis["f1"]) if "f1" in cis else f"{ex[2]:.3f}"
        lines.append(f"| {r['system']} | {ex[0]:.3f} | {ex[1]:.3f} | {f1cell} "
                     f"| {ro[2]:.3f} |")
    return "\n".join(lines) + "\n"


def render_l2_board(baseline: dict, reranked: list[dict]) -> str:
    """L2 board: each reranker's top-focused metrics + delta vs the L1 baseline."""
    b = baseline["results"]["exact"]
    lines = [f"Baseline (L1): {baseline['system']} — "
             f"recall@10={b.get('recall@10',0):.3f} MRR={b.get('mrr',0):.3f}", "",
             "| reranker | recall@10 | Δrecall@10 | MRR | ΔMRR |",
             "|---|---|---|---|---|"]
    for r in sorted(reranked, key=lambda x: -x["results"]["exact"].get("recall@10", 0)):
        e = r["results"]["exact"]
        dr = e.get("recall@10", 0) - b.get("recall@10", 0)
        dm = e.get("mrr", 0) - b.get("mrr", 0)
        lines.append(f"| {r['system']} | {e.get('recall@10',0):.3f} | {dr:+.2f} | "
                     f"{e.get('mrr',0):.3f} | {dm:+.2f} |")
    return "\n".join(lines) + "\n"


def render_l1_board(rows: list[dict]) -> str:
    """Markdown table for the L1 embedding-retrieval board, sorted by recall@10."""
    def rec(r, k):
        return r["results"]["exact"].get(k, 0.0)
    ordered = sorted(rows, key=lambda r: rec(r, "recall@10"), reverse=True)
    lines = ["| system | recall@10 | recall@50 | recall@200 | MRR |",
             "|---|---|---|---|---|"]
    for r in ordered:
        e = r["results"]["exact"]
        cis = e.get("cis", {})
        def cell(m):
            return fmt_ci(cis[m]) if m in cis else f"{e.get(m,0):.3f}"
        lines.append(f"| {r['system']} | {cell('recall@10')} | "
                     f"{cell('recall@50')} | {cell('recall@200')} | {cell('mrr')} |")
    return "\n".join(lines) + "\n"


def render_leaderboard(rows: list[dict]) -> str:
    head = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body = [
        "| " + " | ".join(str(r.get(c, "")) for c in COLUMNS) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body]) + "\n"


def main() -> None:
    import sys
    if "--assemble" in sys.argv:
        out = "results/leaderboard.md"  # lowercase: case-insensitive-FS safe
        md = assemble_leaderboard("results/runs")
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"wrote {out}")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--submissions", nargs="+", required=True)
    ap.add_argument("--gen-topk", type=int, default=10)
    ap.add_argument("--out", default="results/leaderboard.md")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        records = json.load(f)
    rows = [leaderboard_row(records, load_submission(p), args.gen_topk)
            for p in args.submissions]

    md = ("# LCSH benchmark — dev leaderboard\n\n"
          f"Generation columns use top-{args.gen_topk} predictions as the set. "
          "Retrieval columns (recall@k, MRR) use the full ranked list.\n\n"
          + render_leaderboard(rows))
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
