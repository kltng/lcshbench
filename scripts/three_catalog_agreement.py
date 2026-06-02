"""Per-book three-catalog subject-heading agreement (Columbia/Harvard/Princeton).

For every book held by ALL THREE catalogs, compare the three heading sets:
how often they share an exact heading, a *similar* heading (same root, i.e.
same main concept ignoring subdivisions), and how often they share nothing.
"""
import json, statistics as st
from itertools import combinations
from lcsh_benchmark.normalize import normalize_label

dev = json.load(open("data/v2.2/dev/dataset_dev.json", encoding="utf-8"))
three = [r for r in dev if r.get("catalog_count") == 3]
CATS = ("columbia", "harvard", "princeton")

def norm_set(hs):      # exact normalized headings
    return {normalize_label(h) for h in hs if normalize_label(h)}
def root_set(hs):      # main concept only (before first '--') = "similar"
    return {normalize_label(h).split("--", 1)[0].strip() for h in hs if normalize_label(h)}

def jacc(a, b): return len(a & b) / len(a | b) if (a | b) else 1.0

def analyse(keyfn, label):
    n = len(three)
    j3, jpair = [], []                       # 3-way and mean-pairwise Jaccard per book
    identical = some3 = pairwise_only = disjoint = 0
    agreed_frac = []                         # fraction of the union agreed by all 3
    for r in three:
        sets = [keyfn(r["ground_truth_lcsh"].get(c, [])) for c in CATS]
        union = set().union(*sets); inter3 = set.intersection(*sets)
        if not union:                        # no headings at all (shouldn't happen for 3-cat)
            continue
        j3.append(len(inter3) / len(union))
        jpair.append(st.mean(jacc(a, b) for a, b in combinations(sets, 2)))
        agreed_frac.append(len(inter3) / len(union))
        pair_hits = any((a & b) for a, b in combinations(sets, 2))
        if sets[0] == sets[1] == sets[2]:
            identical += 1
        if inter3:
            some3 += 1                        # at least one heading shared by ALL three
        elif pair_hits:
            pairwise_only += 1                # shared by some pair, but not all three
        else:
            disjoint += 1                     # share NOTHING (not even one heading in common)
    pct = lambda x: round(100 * x / n, 1)
    print(f"\n## {label} (n={n} books held by all three catalogs)")
    print(f"  identical heading sets (all 3 the same): {identical} ({pct(identical)}%)")
    print(f"  >=1 heading shared by ALL THREE:         {some3} ({pct(some3)}%)")
    print(f"  shared only by a PAIR (not all 3):       {pairwise_only} ({pct(pairwise_only)}%)")
    print(f"  share NOTHING in common:                 {disjoint} ({pct(disjoint)}%)")
    print(f"  3-way Jaccard (|∩3|/|∪3|): mean {st.mean(j3):.3f}  median {st.median(j3):.3f}")
    print(f"  mean pairwise Jaccard:     mean {st.mean(jpair):.3f}  median {st.median(jpair):.3f}")
    print(f"  fraction of the union agreed by all 3:   mean {st.mean(agreed_frac):.3f}")

analyse(norm_set, "EXACT headings")
analyse(root_set, 'SIMILAR headings (root match — same main concept, subdivisions ignored)')
