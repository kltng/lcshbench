"""L4 — an LLM selects the FINAL set of headings to assign (the few a cataloger
would actually use) from the top-N candidates. Output is a set, scored with the
Task A generation metrics. Only candidates may be selected."""
import argparse
import json

from ..chat_backend import parse_heading_list, run_concurrent
from .rerank import _rerank_query

SYSTEM = (
    "You are an expert LCSH cataloger. Given a record and candidate LCSH "
    "headings, select ONLY the headings you would actually assign to this work "
    "(typically 2-6). Choose only from the candidates, verbatim. Respond ONLY "
    "with a JSON array of the selected headings."
)


def build_prompt(record: dict, candidates: list[str]) -> str:
    cand = "\n".join(f"- {c}" for c in candidates)
    return f"{_rerank_query(record)}\n\nCandidates:\n{cand}"


def run(records: list[dict], l1_submission: dict, chat, top_n: int,
        system: str, max_tokens: int = 1024, workers: int = 8) -> dict:
    preds_in = l1_submission["predictions"]

    def one(r):
        cands = preds_in.get(r["id"], [])[:top_n]
        if cands:
            reply = chat.complete(SYSTEM, build_prompt(r, cands), max_tokens)
            return r["id"], [h for h in parse_heading_list(reply) if h in cands]
        return r["id"], []

    out = run_concurrent(records, one, workers, "select")
    return {"system": system, "task": "generation", "predictions": out}


def main() -> None:
    from ..chat_backend import OpenRouterChat, DeepSeekNativeChat
    from ..ledger import Ledger
    ap = argparse.ArgumentParser(description="L4 — LLM final-cut selection.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--l1-submission", required=True)
    ap.add_argument("--provider", required=True, choices=["openrouter", "deepseek"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default="data/chat_cache/ledger.json")
    ap.add_argument("--max-cost", type=float, default=25.0)
    a = ap.parse_args()
    led = Ledger(a.ledger, a.max_cost)
    chat = (DeepSeekNativeChat if a.provider == "deepseek" else OpenRouterChat)(
        a.model, ledger=led)
    records = json.load(open(a.dataset, encoding="utf-8"))
    l1 = json.load(open(a.l1_submission, encoding="utf-8"))
    sub = run(records, l1, chat, a.top_n, system=f"sel-{chat.name}")
    json.dump(sub, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(sub['predictions'])} -> {a.out}; spent ${led.spent:.4f}")
