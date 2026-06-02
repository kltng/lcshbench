"""Population-scale three-catalog heading agreement, over the FULL consensus
store (v2.2.db, ~465K books held & subject-cataloged by all three libraries) —
not the sampled benchmark set."""
import sqlite3, json, statistics as st
from itertools import combinations
from lcsh_benchmark.normalize import normalize_label

CATS = ("columbia", "harvard", "princeton")
cx = sqlite3.connect("data/raw/v2/v2.2.db")

def norm_set(hs): return {normalize_label(h) for h in hs if normalize_label(h)}
def root_set(hs): return {normalize_label(h).split("--", 1)[0].strip() for h in hs if normalize_label(h)}
def jacc(a, b): return len(a & b) / len(a | b) if (a | b) else 1.0

# accumulate both views in one pass over the DB
acc = {"exact": dict(n=0, identical=0, some3=0, pairwise_only=0, disjoint=0, j3=[], jp=[]),
       "root":  dict(n=0, identical=0, some3=0, pairwise_only=0, disjoint=0, j3=[], jp=[])}
cur = cx.execute("SELECT gt_json FROM consensus WHERE catalog_count=3")
for (gt_json,) in cur:
    gt = json.loads(gt_json)
    for view, keyfn in (("exact", norm_set), ("root", root_set)):
        sets = [keyfn(gt.get(c, [])) for c in CATS]
        union = set().union(*sets); inter3 = set.intersection(*sets)
        if not union: continue
        a = acc[view]; a["n"] += 1
        a["j3"].append(len(inter3) / len(union))
        a["jp"].append(st.mean(jacc(x, y) for x, y in combinations(sets, 2)))
        if sets[0] == sets[1] == sets[2]: a["identical"] += 1
        if inter3: a["some3"] += 1
        elif any((x & y) for x, y in combinations(sets, 2)): a["pairwise_only"] += 1
        else: a["disjoint"] += 1

for view, label in (("exact", "EXACT headings"),
                    ("root", "SIMILAR headings (root / same main concept)")):
    a = acc[view]; n = a["n"]; pct = lambda x: round(100 * x / n, 1)
    print(f"\n## {label} — POPULATION (n={n:,} books held & cataloged by all three)")
    print(f"  identical heading sets:            {a['identical']:,} ({pct(a['identical'])}%)")
    print(f"  >=1 heading shared by ALL THREE:   {a['some3']:,} ({pct(a['some3'])}%)")
    print(f"  shared only by a PAIR:             {a['pairwise_only']:,} ({pct(a['pairwise_only'])}%)")
    print(f"  share NOTHING in common:           {a['disjoint']:,} ({pct(a['disjoint'])}%)")
    print(f"  3-way Jaccard: mean {st.mean(a['j3']):.3f}  median {st.median(a['j3']):.3f}")
    print(f"  mean pairwise Jaccard: mean {st.mean(a['jp']):.3f}")
