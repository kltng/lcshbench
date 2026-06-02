# tests/test_build_v2.py
import json
from pathlib import Path
from lcsh_benchmark.scaleup import build_v2


def _corpus_line(source, oclc, title, headings, **kw):
    r = {"source": source, "oclc": oclc, "lccn": "", "lang": "eng", "lc_class": "H",
         "title": title, "authors": [], "date": "", "publisher": "",
         "physical_description": "", "abstract": "", "toc": "", "notes": "", "headings": headings}
    r.update(kw)
    return json.dumps(r, ensure_ascii=False)


def _setup(tmp_path):
    def mrec(i):
        return {"key": f"o:{i}", "lang": "eng", "lc_class": "H", "n_agencies": 2,
                "ground_truth_lcsh": {"columbia": ["Sociology"], "harvard": ["Sociology"]},
                "consensus_exact": {"sociology": {"surface": "Sociology", "votes": 2,
                                    "sources": ["columbia", "harvard"], "tier": "unanimous"}},
                "consensus_root": {}}
    manifest = {"core": [mrec(1)], "breadth": [mrec(2)], "counts": {}}
    (tmp_path / "v2_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    corpus = tmp_path / "corpus"
    (corpus / "columbia").mkdir(parents=True); (corpus / "harvard").mkdir(parents=True)
    for i in (1, 2):
        (corpus / "columbia" / f"{i}.jsonl").write_text(
            _corpus_line("columbia", str(i), "T", {"650": ["Sociology"]}) + "\n", encoding="utf-8")
        (corpus / "harvard" / f"{i}.jsonl").write_text(
            _corpus_line("harvard", str(i), "T longer", {"650": ["Sociology"]}) + "\n", encoding="utf-8")
    return str(tmp_path / "v2_manifest.json"), str(corpus)


def test_build_v2_writes_split_with_withheld_hashed_test_gt(tmp_path):
    manifest, corpus = _setup(tmp_path)
    out = tmp_path / "v2"
    counts = build_v2.build(manifest, corpus, str(out), test_frac=0.5, seed=13)
    assert counts["records"] == 2 and counts["dropped"] == 0
    assert counts["corpus_miss"] == 0 and counts["empty_gt"] == 0
    assert counts["dev"] + counts["test"] == 2

    dev = json.loads((out / "dev/dataset_dev.json").read_text())
    test_pub = json.loads((out / "test/dataset_test.json").read_text())
    hashed = json.loads((out / "test/gt_test.hashed.json").read_text())

    assert "ground_truth_lcsh_merged" in dev[0]
    assert all("ground_truth_lcsh_merged" not in r for r in test_pub)
    assert all("ground_truth_lcsh" not in r for r in test_pub)
    assert all("heading_types" not in r for r in test_pub)
    assert all("ground_truth_lcsh_unanimous" not in r for r in test_pub)
    assert len(hashed) == len(test_pub) and all(len(v) == 1 for v in hashed.values())
