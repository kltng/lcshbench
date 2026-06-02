"""L3 — an LLM reorders the top-N candidates of an L1 submission. The model may
only reorder the given candidates; invented headings are dropped and any
candidate it omits is appended in original order (so recall@N is preserved)."""
import argparse
import json

from ..chat_backend import parse_heading_list, run_concurrent
from .rerank import _rerank_query

SYSTEM = (
    "You are an expert LCSH cataloger. You are given a record and a numbered list "
    "of candidate LCSH headings. Reorder the candidates from most to least "
    "relevant to this record. Only use headings from the list, verbatim. "
    "Respond ONLY with a JSON array of the headings in your preferred order."
)


def build_prompt(record: dict, candidates: list[str]) -> str:
    cand = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    return f"{_rerank_query(record)}\n\nCandidates:\n{cand}"


def run(records: list[dict], l1_submission: dict, chat, top_n: int,
        system: str, max_tokens: int = 2048, workers: int = 8) -> dict:
    # larger max_tokens: reordering up to top_n headings + the model's reasoning.
    preds_in = l1_submission["predictions"]

    def one(r):
        ranked = preds_in.get(r["id"], [])
        head, tail = ranked[:top_n], ranked[top_n:]
        if head:
            reply = chat.complete(SYSTEM, build_prompt(r, head), max_tokens)
            picked = [h for h in parse_heading_list(reply) if h in head]
            picked += [h for h in head if h not in picked]   # omitted -> appended
        else:
            picked = []
        return r["id"], picked + tail

    out = run_concurrent(records, one, workers, "llm-rerank")
    return {"system": system, "task": "selection", "predictions": out}


def main() -> None:
    from ..chat_backend import OpenRouterChat, DeepSeekNativeChat
    from ..ledger import Ledger
    ap = argparse.ArgumentParser(description="L3 — LLM rerank of an L1 submission.")
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
    sub = run(records, l1, chat, a.top_n,
              system=f"{l1.get('system','l1')}+llmrr-{chat.name}")
    json.dump(sub, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(sub['predictions'])} -> {a.out}; spent ${led.spent:.4f}")
