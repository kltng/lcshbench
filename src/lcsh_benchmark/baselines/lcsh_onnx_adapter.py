# src/lcsh_benchmark/baselines/lcsh_onnx_adapter.py
"""lcsh-onnx retrieve+rerank baseline -> benchmark submission.

Run under the lcsh-onnx db-builder venv (it provides onnxruntime + the
pipeline). From `lcsh-onnx/db-builder`:

    uv run python <abs path to this file> \
        --dataset <abs path>/data/dev/dataset_dev.json \
        --out <abs path>/results/runs/lcsh_onnx_dev.json [--limit N]

Emits {"system","task":"selection","predictions":{id:[ranked headings]}}.
"""
import argparse
import json
import sys
from pathlib import Path

from lcsh_db_builder.lib.distill_cache import get_distilled, load_cache
from lcsh_db_builder.lib.embedders import make_embedder
from lcsh_db_builder.lib.pool import (build_query_text, build_rerank_query,
                                      connect, retrieve, search_top_k)
from lcsh_db_builder.lib.rerankers import OnnxCrossEncoder


def augment_vernacular(rec: dict) -> dict:
    """Fold v2 native-script fields into the romanized title/authors the
    lcsh-onnx query builders read, so the multilingual embedder/reranker see
    the original script (CJK/Cyrillic/Arabic/Hebrew) alongside the romanization.
    Latin-script records carry no vernacular and pass through unchanged.
    """
    r = dict(rec)
    tv = rec.get("title_vernacular")
    if tv:
        r["title"] = f"{rec.get('title', '')}\n{tv}".strip()
    av = rec.get("authors_vernacular") or []
    if av:
        r["authors"] = list(rec.get("authors") or []) + list(av)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="dist/lcsh.db")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--rerank-file", default="onnx/model_quantized.onnx")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--vernacular", action="store_true",
                    help="fold native-script title/authors into the query (v2)")
    ap.add_argument("--strategy", default="raw",
                    choices=["raw", "replace", "augment", "multi"],
                    help="route-A query strategy: raw text, or distilled-English "
                         "(replace/augment/multi). Non-raw needs --distill-cache.")
    ap.add_argument("--distill-cache", default=None,
                    help="prebuilt distilled-query cache (id -> English query); "
                         "build with build_distill_cache.py")
    args = ap.parse_args()

    if args.strategy != "raw" and not args.distill_cache:
        sys.exit(f"--strategy {args.strategy} requires --distill-cache")
    cache = load_cache(Path(args.distill_cache)) if args.distill_cache else {}

    conn = connect(Path(args.db))
    embedder = make_embedder("onnx", "onnx-community/embeddinggemma-300m-ONNX", 256)
    reranker = OnnxCrossEncoder("jinaai/jina-reranker-v2-base-multilingual",
                                onnx_file=args.rerank_file)

    with open(args.dataset, encoding="utf-8") as f:
        records = json.load(f)
    if args.limit:
        records = records[:args.limit]

    preds: dict[str, list[str]] = {}
    for i, rec in enumerate(records, 1):
        rq = augment_vernacular(rec) if args.vernacular else rec
        if args.strategy == "raw":
            q = embedder.encode([build_query_text(rq)])[0]
            hits = search_top_k(conn, q, k=args.pool)
        else:
            distilled = get_distilled(rec, None, cache)
            hits = retrieve(conn, embedder, build_query_text(rq), distilled,
                            strategy=args.strategy, k=args.pool)
        labels = [h[1] for h in hits]
        if labels:
            order = reranker.rerank(build_rerank_query(rq), labels)
            labels = [labels[idx] for idx, _ in order]
        preds[rec["id"]] = labels
        print(f"[adapter] {i}/{len(records)} {rec['id']}", file=sys.stderr)

    base = "lcsh-onnx-retrieve-rerank" if args.strategy == "raw" \
        else f"lcsh-onnx-distill-{args.strategy}-rerank"
    system = base + ("-vernacular" if args.vernacular else "")
    out = {"system": system, "task": "selection", "predictions": preds}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[adapter] wrote {len(preds)} records -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
