from lcsh_benchmark.baselines.subset import stratified_subset


def _recs(counts: dict[str, int]) -> list[dict]:
    out = []
    for lang, n in counts.items():
        out.extend({"id": f"{lang}-{i}", "language_code": lang} for i in range(n))
    return out


def test_deterministic_given_seed():
    recs = _recs({"eng": 500, "rus": 100, "hin": 8})
    a = stratified_subset(recs, target=200, seed=13)
    b = stratified_subset(recs, target=200, seed=13)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_all_languages_present_with_floor():
    recs = _recs({"eng": 500, "rus": 100, "hin": 8})
    sub = stratified_subset(recs, target=200, seed=13, floor=5)
    langs = {r["language_code"] for r in sub}
    assert langs == {"eng", "rus", "hin"}
    # small language gets at least the floor (capped at its availability)
    hin = [r for r in sub if r["language_code"] == "hin"]
    assert 5 <= len(hin) <= 8


def test_proportional_allocation():
    recs = _recs({"eng": 800, "rus": 200})  # 80/20 split, n=1000
    sub = stratified_subset(recs, target=100, seed=7)
    eng = sum(r["language_code"] == "eng" for r in sub)
    rus = sum(r["language_code"] == "rus" for r in sub)
    assert eng == 80 and rus == 20


def test_subset_records_are_from_input():
    recs = _recs({"eng": 50, "rus": 50})
    ids = {r["id"] for r in recs}
    sub = stratified_subset(recs, target=20, seed=1)
    assert all(r["id"] in ids for r in sub)
