from lcsh_benchmark.chat_backend import FakeChat
from lcsh_benchmark.generation import generate


def test_generate_emits_generation_submission():
    fc = FakeChat(reply='["Sociology", "Sociology--Research"]')
    records = [{"id": "r1", "title": "A study", "authors": ["Doe, J"],
                "language": "English", "language_code": "eng",
                "abstract": "x", "toc": "", "date": "2020"}]
    sub = generate.run(records, fc, system="fake-gen")
    assert sub["task"] == "generation"
    assert sub["predictions"]["r1"] == ["Sociology", "Sociology--Research"]
    assert sub["system"] == "fake-gen"


def test_build_prompt_includes_signal_fields():
    rec = {"title": "T", "authors": ["A"], "abstract": "ABS", "toc": "TOC",
           "language": "French", "date": "1999"}
    p = generate.build_prompt(rec)
    for s in ("T", "A", "ABS", "TOC", "French"):
        assert s in p
