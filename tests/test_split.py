from lcsh_benchmark.build import stratified_split, hash_gt


def test_stratified_split_is_deterministic_and_per_language():
    recs = [{"id": f"r{i}", "language_code": "eng"} for i in range(10)] + \
           [{"id": f"c{i}", "language_code": "chi"} for i in range(10)]
    dev1, test1 = stratified_split(recs, test_frac=0.2, seed=13)
    dev2, test2 = stratified_split(recs, test_frac=0.2, seed=13)
    assert [r["id"] for r in test1] == [r["id"] for r in test2]  # deterministic
    assert len(test1) == 4 and len(dev1) == 16                   # 20% of each lang
    assert {r["language_code"] for r in test1} == {"eng", "chi"}


def test_hash_gt_hashes_normalized_headings():
    import hashlib
    h = hashlib.sha256("sociology".encode()).hexdigest()
    assert hash_gt(["Sociology."]) == [h]
