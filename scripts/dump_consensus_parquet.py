"""Dump the consensus store -> Parquet for HF (queryable via the Dataset Viewer
/ DuckDB). Identifiers are HASHED (opaque key_hash) so we share the agreement
structure + LCSH headings without redistributing OCLC/LCCN identifiers."""
import sqlite3, json, hashlib
import pyarrow as pa, pyarrow.parquet as pq

DB = "data/raw/v2/v2.2.db"
OUT = "data/lcsh_consensus_matched.parquet"
cx = sqlite3.connect(DB)
rows = []
for mk, cc, nag, lang, lcc, ism, gt_json in cx.execute(
        "SELECT match_key,catalog_count,n_agencies,lang,lc_class,is_monograph,gt_json FROM consensus"):
    gt = json.loads(gt_json)
    rows.append({
        "key_hash": hashlib.sha256(mk.encode()).hexdigest()[:16],   # opaque; no raw OCLC/LCCN
        "catalog_count": cc, "n_agencies": nag, "lang": lang, "lc_class": lcc,
        "is_monograph": bool(ism),
        "gt_columbia": gt.get("columbia", []),
        "gt_harvard": gt.get("harvard", []),
        "gt_princeton": gt.get("princeton", []),
    })
pq.write_table(pa.Table.from_pylist(rows), OUT, compression="zstd")
import os
print(f"wrote {len(rows):,} rows -> {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB)")
