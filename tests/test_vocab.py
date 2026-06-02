import sqlite3
import json
from lcsh_benchmark.retrieval import vocab


def _fixture_db(path):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE auth (id INTEGER PRIMARY KEY, uri TEXT, authority TEXT, "
              "label TEXT, label_normalized TEXT, scope_note TEXT, deprecated INTEGER)")
    rows = [
        (1, "u/1", "lcsh", "Sociology", "sociology", None, 0),
        (2, "u/2", "lcsh", "Music", "music", None, 0),
        (3, "u/3", "lcgft", "Fiction", "fiction", None, 0),
        (4, "u/4", "lcnaf", "Twain, Mark", "twain, mark", None, 0),   # excluded
        (5, "u/5", "lcsh", "Obsolete", "obsolete", None, 1),          # deprecated, excluded
    ]
    c.executemany("INSERT INTO auth VALUES (?,?,?,?,?,?,?)", rows)
    c.commit(); c.close()


def test_extract_vocab_lcsh_and_lcgft_only(tmp_path):
    db = tmp_path / "lcsh.db"
    _fixture_db(db)
    out = tmp_path / "vocab.jsonl"
    n = vocab.extract(str(db), str(out))
    assert n == 3  # 2 lcsh + 1 lcgft, names + deprecated excluded
    recs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    auths = {r["authority"] for r in recs}
    assert auths == {"lcsh", "lcgft"}
    r0 = recs[0]
    assert set(r0) == {"uri", "label", "authority", "normalized"}
    assert r0["normalized"] == r0["normalized"].strip().lower()
