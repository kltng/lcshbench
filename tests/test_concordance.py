"""Per-book three-catalog concordance (stats.concordance).

Fixture (3 books, catalogs c/h/p), hand-computed expectations:
  B1: all identical {A,B}
  B2: c={X--Y} h={X--Z} p={X--W}  -> exact disjoint; root all share {x}
  B3: c={P} h={P} p={Q}           -> P shared by 2 (not all 3); Q single
"""
from lcsh_benchmark.scaleup.stats import concordance

def _rec(c, h, p, n_agencies=3):
    return {"catalog_count": 3, "n_agencies": n_agencies,
            "ground_truth_lcsh": {"columbia": c, "harvard": h, "princeton": p}}

RECS = [
    _rec(["A", "B"], ["A", "B"], ["A", "B"]),
    _rec(["X--Y"], ["X--Z"], ["X--W"]),
    _rec(["P"], ["P"], ["Q"]),
]


def test_exact_per_book_agreement():
    c = concordance(RECS)
    assert c["n_3cat"] == 3
    e = c["exact"]
    assert e["identical_pct"] == 33.3        # B1
    assert e["ge1_shared_pct"] == 33.3       # B1 only (A,B by all 3)
    assert e["share_nothing_pct"] == 33.3    # B2 (no pair shares an exact heading)


def test_root_relaxes_disagreement():
    r = concordance(RECS)["root"]
    assert r["identical_pct"] == 66.7        # B1 {a,b}, B2 {x}
    assert r["ge1_shared_pct"] == 66.7       # B1, B2
    assert r["share_nothing_pct"] == 0.0     # B2 now shares root x; B3 shares p


def test_single_source_rate_exact():
    ss = concordance(RECS)["single_source"]["exact"]
    # assertions: A(3),B(3) | X--Y(1),X--Z(1),X--W(1) | P(2),Q(1)  -> n=7
    assert ss["n"] == 7
    assert ss["single_pct"] == 57.1          # 4/7
    assert ss["unanimous_pct"] == 28.6       # 2/7


def test_independence_control_buckets_by_n_agencies():
    recs = [_rec(["A"], ["A"], ["A"], n_agencies=3),
            _rec(["A"], ["A"], ["B"], n_agencies=2)]
    ind = concordance(recs)["independence"]
    assert ind["indep3"]["n"] == 1 and ind["shared"]["n"] == 1
