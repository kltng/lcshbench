# src/lcsh_benchmark/scaleup/stats.py
"""Descriptive statistics for the paper — the questions every reviewer asks.

Three blocks:
  1. catalog_funnel   — per-catalog totals + overlap (exactly-2 vs all-3), from
                        the per-source index shards (the raw population, on VPS).
  2. consensus_structure — heading-level inter-catalog agreement: tiers,
                        pairwise Jaccard, % corroborated, headings/book. Needs
                        per-catalog GT (dataset dev, or the matched population).
  3. composition      — language / LC-class / heading-type mix, subdivision
                        structure, vocab reachability, input coverage. From the
                        released dataset + vocab.

All heading comparisons normalize first (normalize_label) — comparing raw
surface strings miscounts punctuation/case variants as disagreements.

CLI:
  lcsh-benchmark-stats --indices data/raw/v2/corpus/*/*.jsonl \\
                       --dataset data/v2.2/dev/dataset_dev.json \\
                       --vocab data/vocab/vocab.jsonl --out results/dataset-stats.md
Any block whose inputs are absent is skipped, so it runs incrementally.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import statistics
from collections import Counter, defaultdict

from ..normalize import normalize_label

CATALOGS = ("harvard", "columbia", "princeton")


def _key(r: dict) -> str:
    o = (r.get("oclc") or "").strip()
    if o.isdigit() and int(o) > 0:
        return f"o:{o}"
    l = (r.get("lccn") or "").strip()
    return f"l:{l}" if l else ""


# ---------- 1. catalog overlap funnel (raw population) ----------

def catalog_funnel(index_paths: list[str]) -> dict:
    """Per-catalog totals and cross-catalog overlap from the index shards.

    Counts a book once per (source, match_key). 'with_lcsh' = has >=1 kept 6xx.
    Overlap is computed on the set of sources holding each match_key."""
    per_source_keys: dict[str, set] = defaultdict(set)
    per_source_records = Counter()
    per_source_with_lcsh = Counter()
    key_sources: dict[str, set] = defaultdict(set)

    for path in index_paths:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            src = r.get("source", "?")
            per_source_records[src] += 1
            if r.get("headings"):
                per_source_with_lcsh[src] += 1
            k = _key(r)
            if not k:
                continue
            per_source_keys[src].add(k)
            key_sources[k].add(src)

    # overlap by exact source-set
    combo = Counter(frozenset(s) for s in key_sources.values())
    n2 = sum(v for s, v in combo.items() if len(s) == 2)
    n3 = sum(v for s, v in combo.items() if len(s) == 3)
    return {
        "per_source_records": dict(per_source_records),
        "per_source_with_lcsh": dict(per_source_with_lcsh),
        "per_source_unique_keys": {s: len(k) for s, k in per_source_keys.items()},
        "distinct_keys": len(key_sources),
        "held_by_1": sum(v for s, v in combo.items() if len(s) == 1),
        "held_by_2": n2,
        "held_by_3": n3,
        "matched_ge2": n2 + n3,
        "combinations": {"+".join(sorted(s)): v for s, v in
                         sorted(combo.items(), key=lambda kv: -kv[1])},
    }


# ---------- 2. inter-catalog agreement (consensus structure) ----------

def consensus_structure(records: list[dict]) -> dict:
    """records must carry ground_truth_lcsh = {catalog: [headings]}.
    Works on dataset dev or the full matched population."""
    tiers_by_cc: dict = {2: Counter(), 3: Counter(), "all": Counter()}
    pair_jaccard: dict = defaultdict(list)
    per_book_headings: list[int] = []
    corrob = total = 0

    for r in records:
        per = {c: set(normalize_label(h) for h in hs)
               for c, hs in (r.get("ground_truth_lcsh") or {}).items() if hs}
        if not per:
            continue
        ncat = len(per)
        cnt = Counter(h for hs in per.values() for h in hs)
        per_book_headings.append(len(cnt))
        cc = r.get("catalog_count") or ncat
        for h, v in cnt.items():
            tier = "unanimous" if (v == ncat and ncat > 1) else ("single" if v == 1 else "majority")
            tiers_by_cc["all"][tier] += 1
            if cc in (2, 3):
                tiers_by_cc[cc][tier] += 1
            total += 1
            if v >= 2:
                corrob += 1
        for a, b in itertools.combinations(sorted(per), 2):
            u = per[a] | per[b]
            if u:
                pair_jaccard[(a, b)].append(len(per[a] & per[b]) / len(u))

    def pct(counter):
        t = sum(counter.values()) or 1
        return {k: round(100 * counter[k] / t, 1) for k in ("single", "majority", "unanimous")} | {"n": sum(counter.values())}

    return {
        "tiers_all": pct(tiers_by_cc["all"]),
        "tiers_2cat": pct(tiers_by_cc[2]),
        "tiers_3cat": pct(tiers_by_cc[3]),
        "pairwise_jaccard": {f"{a}+{b}": {"mean": round(statistics.mean(v), 3), "n_books": len(v)}
                             for (a, b), v in sorted(pair_jaccard.items())},
        "assertions_corroborated_ge2_pct": round(100 * corrob / max(1, total), 1),
        "distinct_assertions": total,
        "headings_per_book": {
            "mean": round(statistics.mean(per_book_headings), 2),
            "median": int(statistics.median(per_book_headings)),
            "p95": int(sorted(per_book_headings)[int(0.95 * len(per_book_headings))]),
            "max": max(per_book_headings),
        },
    }


# ---------- 2b. three-catalog concordance (per-book, for the paper) ----------

def _nset(hs):  return {normalize_label(h) for h in hs if normalize_label(h)}
def _rset(hs):  return {normalize_label(h).split("--", 1)[0].strip() for h in hs if normalize_label(h)}
def _jacc(a, b): return len(a & b) / len(a | b) if (a | b) else 1.0


def concordance(records: list[dict], catalogs=CATALOGS) -> dict:
    """Per-book agreement among the three catalogs, for books all three cataloged.

    'exact' = normalized heading; 'root' (= "similar") = main concept before the
    first '--'. Independence control buckets by n_agencies (>=3 distinct 040
    agencies = copy-cataloging control). Also: single-source rate, per-library
    granularity, human-cataloger baseline (each library vs the consensus of the
    other two), and the subdivision-depth disagreement axis.
    """
    three = [r for r in records if (r.get("catalog_count")
             or len([c for c in catalogs if (r.get("ground_truth_lcsh") or {}).get(c)])) == 3]
    out = {"n_3cat": len(three)}
    if not three:
        return out

    def per_book(keyfn):
        ident = ge1 = nothing = 0
        j3, jp = [], []
        for r in three:
            sets = [keyfn((r["ground_truth_lcsh"]).get(c, [])) for c in catalogs]
            union = set().union(*sets); inter3 = set.intersection(*sets)
            if not union:
                continue
            j3.append(len(inter3) / len(union))
            jp.append(statistics.mean(_jacc(a, b) for a, b in itertools.combinations(sets, 2)))
            if sets[0] == sets[1] == sets[2]: ident += 1
            if inter3: ge1 += 1
            elif not any((a & b) for a, b in itertools.combinations(sets, 2)): nothing += 1
        n = len(j3); p = lambda x: round(100 * x / n, 1)
        return {"identical_pct": p(ident), "ge1_shared_pct": p(ge1),
                "share_nothing_pct": p(nothing),
                "jaccard3_mean": round(statistics.mean(j3), 3),
                "jaccard_pairwise_mean": round(statistics.mean(jp), 3)}

    out["exact"] = per_book(_nset)
    out["root"] = per_book(_rset)

    # single-source rate (share of all (book,heading) assertions by 1 / 2 / 3 libraries)
    def single_source(keyfn):
        tier = Counter()
        for r in three:
            vote = Counter()
            for c in catalogs:
                for h in keyfn((r["ground_truth_lcsh"]).get(c, [])): vote[h] += 1
            for v in vote.values(): tier[v] += 1
        tot = sum(tier.values()) or 1
        return {"single_pct": round(100 * tier[1] / tot, 1),
                "majority_pct": round(100 * tier[2] / tot, 1),
                "unanimous_pct": round(100 * tier[3] / tot, 1), "n": sum(tier.values())}
    out["single_source"] = {"exact": single_source(_nset), "root": single_source(_rset)}

    # independence control: bucket by n_agencies (>=3 = three distinct cataloging agencies)
    def bucket(pred):
        recs = [r for r in three if pred(r.get("n_agencies") or 0)]
        if not recs: return {"n": 0}
        ident = ge1 = 0; j3 = []
        for r in recs:
            sets = [_nset((r["ground_truth_lcsh"]).get(c, [])) for c in catalogs]
            u = set().union(*sets); i3 = set.intersection(*sets)
            if not u: continue
            j3.append(len(i3) / len(u))
            if sets[0] == sets[1] == sets[2]: ident += 1
            if i3: ge1 += 1
        n = len(j3) or 1
        return {"n": len(j3), "identical_pct": round(100 * ident / n, 1),
                "ge1_shared_pct": round(100 * ge1 / n, 1), "jaccard3_mean": round(statistics.mean(j3), 3)}
    out["independence"] = {"indep3": bucket(lambda a: a >= 3), "shared": bucket(lambda a: a == 2)}

    # per-library granularity (headings per book) and subdivision depth
    hc = {c: [] for c in catalogs}; depth = {c: [] for c in catalogs}
    for r in three:
        for c in catalogs:
            s = _nset((r["ground_truth_lcsh"]).get(c, []))
            hc[c].append(len(s))
            depth[c].extend(h.count("--") for h in s)
    out["granularity"] = {c: {"mean_headings": round(statistics.mean(hc[c]), 2),
                              "mean_subdiv_depth": round(statistics.mean(depth[c] or [0]), 2)} for c in catalogs}

    # (g) human-cataloger baseline: each library vs the consensus (intersection) of the other two
    def human(keyfn):
        per_cat = {}
        for left in catalogs:
            others = [c for c in catalogs if c != left]
            recs = []
            for r in three:
                cons = keyfn((r["ground_truth_lcsh"]).get(others[0], [])) & keyfn((r["ground_truth_lcsh"]).get(others[1], []))
                if cons:
                    recs.append(len(keyfn((r["ground_truth_lcsh"]).get(left, [])) & cons) / len(cons))
            per_cat[left] = round(statistics.mean(recs), 3) if recs else None
        vals = [v for v in per_cat.values() if v is not None]
        return {"per_catalog_recall": per_cat, "mean": round(statistics.mean(vals), 3) if vals else None}
    out["human_baseline"] = {"exact": human(_nset), "root": human(_rset)}
    return out


# ---------- 3. dataset composition ----------

def composition(records: list[dict], vocab_path: str | None = None) -> dict:
    n = len(records)
    lang = Counter(r.get("language_code") or r.get("lang") or "?" for r in records)
    lcc = Counter((r.get("lc_class") or "<blank>") for r in records)
    cc = Counter(r.get("catalog_count") for r in records)

    headings = [h for r in records for h in (r.get("ground_truth_lcsh_merged") or [])]
    types = Counter()
    for r in records:
        ht = r.get("heading_types") or {}
        for h in (r.get("ground_truth_lcsh_merged") or []):
            t = ht.get(h, {})
            types[(t.get("type") or t.get("authority") or "untyped")] += 1
    subdiv = sum(1 for h in headings if "--" in h)

    out = {
        "n_records": n,
        "catalog_count_pct": {k: round(100 * v / n, 1) for k, v in sorted(cc.items())},
        "languages": {k: round(100 * v / n, 1) for k, v in lang.most_common()},
        "lc_classes": {k: round(100 * v / n, 1) for k, v in lcc.most_common()},
        "heading_type_pct": {k: round(100 * v / max(1, len(headings)), 1) for k, v in types.most_common()},
        "total_headings": len(headings),
        "headings_with_subdivision_pct": round(100 * subdiv / max(1, len(headings)), 1),
        "abstract_coverage_pct": round(100 * sum(1 for r in records if (r.get("abstract") or "").strip()) / n, 1),
        "toc_coverage_pct": round(100 * sum(1 for r in records if (r.get("toc") or "").strip()) / n, 1),
        "input_coverage_by_language": {},
    }
    by_lang_cov: dict = defaultdict(lambda: [0, 0])
    for r in records:
        L = r.get("language_code") or "?"
        by_lang_cov[L][1] += 1
        if (r.get("abstract") or "").strip() or (r.get("toc") or "").strip():
            by_lang_cov[L][0] += 1
    out["input_coverage_by_language"] = {L: round(100 * a / t, 0) for L, (a, t) in sorted(by_lang_cov.items())}

    if vocab_path:
        exact, root = set(), set()
        for line in open(vocab_path, encoding="utf-8"):
            lab = json.loads(line)["label"]
            exact.add(normalize_label(lab))
            root.add(normalize_label(lab).split("--", 1)[0])
        er = sum(1 for h in headings if normalize_label(h) in exact)
        rr = sum(1 for h in headings if normalize_label(h).split("--", 1)[0] in root)
        out["exact_reachable_pct"] = round(100 * er / max(1, len(headings)), 1)
        out["root_reachable_pct"] = round(100 * rr / max(1, len(headings)), 1)
    return out


def render_md(funnel, consensus, comp, concord=None) -> str:
    lines = ["# Dataset statistics\n"]
    if funnel:
        lines.append("## Catalog overlap (raw population)\n")
        lines.append(f"- per-catalog records: {funnel['per_source_records']}")
        lines.append(f"- with LCSH: {funnel['per_source_with_lcsh']}")
        lines.append(f"- held by exactly 2: **{funnel['held_by_2']:,}**, all 3: "
                     f"**{funnel['held_by_3']:,}**, matched (≥2): **{funnel['matched_ge2']:,}**")
        lines.append(f"- combinations: {funnel['combinations']}\n")
    if consensus:
        c = consensus
        lines.append("## Inter-catalog agreement (heading level, normalized)\n")
        lines.append(f"- tiers (all): {c['tiers_all']}")
        lines.append(f"- tiers (2-cat): {c['tiers_2cat']}  | (3-cat): {c['tiers_3cat']}")
        lines.append(f"- pairwise Jaccard: {c['pairwise_jaccard']}")
        lines.append(f"- assertions corroborated by ≥2 catalogs: **{c['assertions_corroborated_ge2_pct']}%**")
        lines.append(f"- headings/book: {c['headings_per_book']}\n")
    if concord and concord.get("n_3cat"):
        c = concord
        lines.append("## Three-catalog concordance (per book, all three cataloged)\n")
        lines.append(f"- n books: **{c['n_3cat']:,}**")
        lines.append(f"- EXACT: identical {c['exact']['identical_pct']}% · "
                     f"≥1 shared {c['exact']['ge1_shared_pct']}% · share-nothing "
                     f"{c['exact']['share_nothing_pct']}% · Jaccard₃ {c['exact']['jaccard3_mean']}")
        lines.append(f"- SIMILAR (root): identical {c['root']['identical_pct']}% · "
                     f"≥1 shared {c['root']['ge1_shared_pct']}% · share-nothing "
                     f"{c['root']['share_nothing_pct']}% · Jaccard₃ {c['root']['jaccard3_mean']}")
        lines.append(f"- single-source assertions: exact {c['single_source']['exact']['single_pct']}% · "
                     f"root {c['single_source']['root']['single_pct']}%")
        lines.append(f"- independence control (n_agencies≥3, n={c['independence']['indep3'].get('n', 0):,}): "
                     f"≥1 shared {c['independence']['indep3'].get('ge1_shared_pct')}% "
                     f"vs shared-agency {c['independence']['shared'].get('ge1_shared_pct')}%")
        lines.append(f"- human baseline (library vs consensus of other two): "
                     f"exact recall {c['human_baseline']['exact']['mean']} · "
                     f"root {c['human_baseline']['root']['mean']}")
        lines.append(f"- granularity / subdivision depth: {c['granularity']}\n")
    if comp:
        lines.append("## Composition\n")
        for k, v in comp.items():
            lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def _load_matched(path: str) -> list[dict]:
    """Load the matched population from Parquet (gt_<catalog> list columns) or JSON.
    Reconstructs ground_truth_lcsh = {catalog: [headings]} for the agreement code."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        tbl = pq.read_table(path).to_pylist()
        out = []
        for row in tbl:
            gt = {c: list(row.get(f"gt_{c}") or []) for c in CATALOGS}
            out.append({"catalog_count": row.get("catalog_count"),
                        "n_agencies": row.get("n_agencies"), "ground_truth_lcsh": gt})
        return out
    return json.load(open(path, encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", nargs="*", default=[], help="per-source index JSONL shards (glob ok)")
    ap.add_argument("--dataset", default=None, help="released dataset json (dev) for agreement + composition")
    ap.add_argument("--matched", default=None, help="full matched population json (optional, for population agreement)")
    ap.add_argument("--vocab", default="data/vocab/vocab.jsonl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    paths = [p for pat in a.indices for p in glob.glob(pat)]
    funnel = catalog_funnel(paths) if paths else None

    agree_src = None
    if a.matched:
        agree_src = _load_matched(a.matched)
    elif a.dataset:
        agree_src = json.load(open(a.dataset, encoding="utf-8"))
    consensus = consensus_structure(agree_src) if agree_src else None
    concord = concordance(agree_src) if agree_src else None

    comp = composition(json.load(open(a.dataset, encoding="utf-8")), a.vocab) if a.dataset else None

    md = render_md(funnel, consensus, comp, concord)
    print(md)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(md)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
