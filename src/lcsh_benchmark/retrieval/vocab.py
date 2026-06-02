"""Extract the LCSH + LCGFT label vocabulary (the Task B retrieval target).

Reads the lcsh-onnx authority DB (default ../lcsh-onnx/db-builder/dist/lcsh.db),
keeps non-deprecated lcsh + lcgft labels, drops the 12.2M LCNAF names (out of
scope), and writes one JSON object per line to vocab.jsonl.
"""
import argparse
import json
import sqlite3
from pathlib import Path

from ..normalize import normalize_label

DEFAULT_DB = "../lcsh-onnx/db-builder/dist/lcsh.db"
VOCAB_AUTHORITIES = ("lcsh", "lcgft")


def extract(db_path: str, out_path: str) -> int:
    """Write the vocab to out_path (JSONL). Returns the number of rows written."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(VOCAB_AUTHORITIES))
    cur = conn.execute(
        f"SELECT uri, label, authority FROM auth "
        f"WHERE authority IN ({placeholders}) AND COALESCE(deprecated, 0) = 0 "
        f"ORDER BY authority, label",
        VOCAB_AUTHORITIES,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in cur:
            rec = {
                "uri": row["uri"],
                "label": row["label"],
                "authority": row["authority"],
                "normalized": normalize_label(row["label"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    conn.close()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract LCSH+LCGFT vocab to JSONL.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default="data/vocab/vocab.jsonl")
    a = ap.parse_args()
    n = extract(a.db, a.out)
    print(f"wrote {n} vocab labels -> {a.out}")
