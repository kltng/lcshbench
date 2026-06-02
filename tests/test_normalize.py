from lcsh_benchmark.normalize import normalize_label


def test_lowercases_and_strips_trailing_period():
    assert normalize_label("Sociology.") == "sociology"


def test_canonicalizes_em_and_en_dash_separators():
    assert normalize_label("Music — History") == "music--history"
    assert normalize_label("Music – History") == "music--history"
    assert normalize_label("Music -- History") == "music--history"


def test_collapses_whitespace_and_nfc():
    assert normalize_label("World  War,  1939") == "world war, 1939"
    assert normalize_label("Café") == normalize_label("Café")


def test_keeps_parenthetical_qualifier():
    assert normalize_label("Mercury (Planet)") == "mercury (planet)"
