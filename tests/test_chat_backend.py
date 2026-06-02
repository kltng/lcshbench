import pytest
from lcsh_benchmark.chat_backend import FakeChat, parse_heading_list
from lcsh_benchmark.ledger import Ledger, BudgetExceeded


def test_fake_chat_caches_and_meters(tmp_path):
    led = Ledger(str(tmp_path / "l.json"), cap_usd=1.0)
    fc = FakeChat(reply='["Music", "France--History"]', price_in=1.0, price_out=1.0,
                  ledger=led, cache_dir=str(tmp_path / "cache"))
    out1 = fc.complete("sys", "user A", max_tokens=50)
    assert out1 == '["Music", "France--History"]'
    spent_after_first = led.spent
    assert spent_after_first > 0
    fc.complete("sys", "user A", max_tokens=50)        # identical -> cache hit
    assert led.spent == spent_after_first              # no new charge
    assert fc.calls == 1                                # only one real call


def test_parse_heading_list_tolerates_fences_and_prose():
    assert parse_heading_list('```json\n["A", "B"]\n```') == ["A", "B"]
    assert parse_heading_list('Here:\n- Music\n- France') == ["Music", "France"]
    assert parse_heading_list('["A","A","B"]') == ["A", "B"]   # dedup, order-preserving


def test_over_cap_raises_before_calling(tmp_path):
    led = Ledger(str(tmp_path / "l.json"), cap_usd=0.0)
    fc = FakeChat(reply="x", price_in=1.0, price_out=1.0, ledger=led,
                  cache_dir=str(tmp_path / "c"))
    with pytest.raises(BudgetExceeded):
        fc.complete("sys", "u", max_tokens=1000)
    assert fc.calls == 0                                # never hit the API
