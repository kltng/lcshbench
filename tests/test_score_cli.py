from lcsh_benchmark.score import render_report


def test_render_report_contains_headline_numbers():
    gen = {"exact": {"n": 2, "micro": (0.5, 0.4, 0.444), "macro": (0.5, 0.4, 0.44),
                     "per_language": {"eng": (0.5, 0.4, 0.44)},
                     "per_type_recall": {"lcsh": (0.6, 10), "lcnaf": (0.2, 5)}},
           "root": {"n": 2, "micro": (0.7, 0.6, 0.64), "macro": (0.7, 0.6, 0.64),
                    "per_language": {"eng": (0.7, 0.6, 0.64)},
                    "per_type_recall": {"lcsh": (0.8, 10)}}}
    text = render_report("generation", gen)
    assert "generation" in text
    assert "lcsh" in text and "exact" in text and "root" in text
    assert "gap" in text  # exact-vs-root gap column shown by default


def test_render_report_exact_only_hides_root_and_empty_per_type():
    # held-out test shape: only 'exact', no per_type_recall
    res = {"exact": {"n": 2, "micro": (0.5, 0.4, 0.44), "macro": (0.5, 0.4, 0.44),
                     "per_language": {"eng": (0.5, 0.4, 0.44)}, "per_type_recall": {}}}
    text = render_report("generation", res, exact_only=True)
    assert "exact" in text
    assert "root" not in text and "gap" not in text   # no phantom root/gap columns
    assert "Per-type recall" not in text              # suppressed when empty
