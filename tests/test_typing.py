import sqlite3
from lcsh_benchmark.typing import base_of, type_heading, make_db_lookup


def test_base_of_takes_first_segment():
    assert base_of("Sociology--Research--History") == "Sociology"
    assert base_of("Sociology") == "Sociology"


def test_type_heading_uses_lookup_and_subdivision_flags():
    lookup = {"sociology": "lcsh", "france": "lcnaf"}.get
    t = type_heading("Sociology--Research", lookup)
    assert t == {
        "base": "sociology", "authority": "lcsh",
        "has_subdivision": True, "subdivision_depth": 1,
    }
    assert type_heading("France", lookup)["authority"] == "lcnaf"
    assert type_heading("Nonexistent Term", lookup)["authority"] == "unmatched"


def test_make_db_lookup_reads_auth_table():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE auth (label_normalized TEXT, authority TEXT)")
    conn.execute("INSERT INTO auth VALUES ('sociology', 'lcsh')")
    conn.commit()
    lookup = make_db_lookup(conn)
    assert lookup("sociology") == "lcsh"
    assert lookup("missing") is None
