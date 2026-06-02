"""Task A — open-vocabulary LCSH generation from bibliographic input."""
import argparse
import json

from ..chat_backend import parse_heading_list, run_concurrent

SYSTEM = (
    "You are an expert library cataloger assigning Library of Congress Subject "
    "Headings (LCSH). Given a bibliographic record, output the LCSH headings you "
    "would assign. Use valid LCSH form including subdivisions with ' -- '. "
    "Respond ONLY with a JSON array of heading strings, most important first."
)


def build_prompt(record: dict) -> str:
    f = []
    f.append(f"Title: {record.get('title','')}")
    if record.get("authors"):
        f.append("Authors: " + "; ".join(record["authors"]))
    if record.get("language"):
        f.append(f"Language: {record['language']}")
    if record.get("date"):
        f.append(f"Date: {record['date']}")
    if record.get("abstract"):
        f.append(f"Abstract: {record['abstract']}")
    if record.get("toc"):
        f.append(f"Table of contents: {record['toc'][:2000]}")
    return "\n".join(f)


def run(records: list[dict], chat, system: str, max_tokens: int = 1024,
        workers: int = 8) -> dict:
    # max_tokens generous: deepseek v4 models reason before answering, so a
    # small budget can leave `content` empty (reasoning eats it).
    def one(r):
        reply = chat.complete(SYSTEM, build_prompt(r), max_tokens)
        return r["id"], parse_heading_list(reply)
    preds = run_concurrent(records, one, workers, "generate")
    return {"system": system, "task": "generation", "predictions": preds}


def main() -> None:
    from ..chat_backend import OpenRouterChat, DeepSeekNativeChat
    from ..ledger import Ledger
    ap = argparse.ArgumentParser(description="Task A — LLM generation.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--provider", required=True, choices=["openrouter", "deepseek"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default="data/chat_cache/ledger.json")
    ap.add_argument("--max-cost", type=float, default=25.0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    led = Ledger(a.ledger, a.max_cost)
    chat = (DeepSeekNativeChat if a.provider == "deepseek" else OpenRouterChat)(
        a.model, ledger=led)
    records = json.load(open(a.dataset, encoding="utf-8"))
    if a.limit:
        records = records[: a.limit]
    sub = run(records, chat, system=f"gen-{chat.name}")
    json.dump(sub, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(sub['predictions'])} -> {a.out}; spent ${led.spent:.4f}")
