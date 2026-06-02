"""(a) independent-agency concordance control + (b) single-source rate &
granularity, over the full 3-catalog population in v2.2.db."""
import sqlite3, json, statistics as st
from itertools import combinations
from collections import Counter
from lcsh_benchmark.normalize import normalize_label

CATS = ("columbia", "harvard", "princeton")
cx = sqlite3.connect("data/raw/v2/v2.2.db")
def norm_set(hs): return {normalize_label(h) for h in hs if normalize_label(h)}
def root_set(hs): return {normalize_label(h).split("--", 1)[0].strip() for h in hs if normalize_label(h)}
def jacc(a, b): return len(a & b) / len(a | b) if (a | b) else 1.0

# (a) concordance bucketed by n_agencies (independence proxy)
def newbucket(): return dict(n=0, identical=0, some3=0, disjoint=0, j3=[], jp=[])
buckets = {}                       # n_agencies bucket -> stats (exact only for (a))
# (b) heading-level tiers (exact + root) and per-library granularity
tiers = {"exact": Counter(), "root": Counter()}     # total (book,heading) assertions by vote-count
hcount = {c: [] for c in CATS}                       # headings-per-book per library

cur = cx.execute("SELECT n_agencies, gt_json FROM consensus WHERE catalog_count=3")
for nag, gt_json in cur:
    gt = json.loads(gt_json)
    # ---- (a) exact concordance by independence bucket ----
    sets = [norm_set(gt.get(c, [])) for c in CATS]
    union = set().union(*sets); inter3 = set.intersection(*sets)
    if not union: continue
    key = "3 (fully independent)" if nag >= 3 else "2 (one shared agency)"
    b = buckets.setdefault(key, newbucket()); b["n"] += 1
    b["j3"].append(len(inter3)/len(union))
    b["jp"].append(st.mean(jacc(x, y) for x, y in combinations(sets, 2)))
    if sets[0]==sets[1]==sets[2]: b["identical"] += 1
    if inter3: b["some3"] += 1
    elif not any((x & y) for x, y in combinations(sets, 2)): b["disjoint"] += 1
    # ---- (b) per-heading vote tiers (exact + root) + granularity ----
    for c in CATS: hcount[c].append(len(norm_set(gt.get(c, []))))
    for view, kf in (("exact", norm_set), ("root", root_set)):
        vote = Counter()
        for c in CATS:
            for h in kf(gt.get(c, [])): vote[h] += 1
        for h, v in vote.items(): tiers[view][v] += 1

print("# (a) Independent-agency concordance control (3-catalog population, EXACT)")
for key in sorted(buckets, reverse=True):
    b = buckets[key]; n = b["n"]; p = lambda x: round(100*x/n, 1)
    print(f"\n  n_agencies = {key}  (n={n:,})")
    print(f"    identical sets:            {p(b['identical'])}%")
    print(f"    >=1 heading shared by all3:{p(b['some3'])}%")
    print(f"    share nothing:             {p(b['disjoint'])}%")
    print(f"    3-way Jaccard mean:        {st.mean(b['j3']):.3f}   pairwise mean: {st.mean(b['jp']):.3f}")

print("\n# (b) Single-source rate & granularity (3-catalog population)")
for view in ("exact", "root"):
    t = tiers[view]; tot = sum(t.values())
    pc = lambda v: round(100*t[v]/tot, 1)
    print(f"\n  {view}: {tot:,} (book,heading) assertions")
    print(f"    assigned by only 1 of 3 (single-source): {pc(1)}%")
    print(f"    by exactly 2 (majority):                 {pc(2)}%")
    print(f"    by all 3 (unanimous):                    {pc(3)}%")
print("\n  headings per book, by library (exact, mean / median):")
for c in CATS:
    print(f"    {c:10s}: {st.mean(hcount[c]):.2f} / {st.median(hcount[c]):.0f}")
