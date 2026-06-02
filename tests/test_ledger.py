import json
import pytest
from lcsh_benchmark.ledger import Ledger, BudgetExceeded


def test_charge_accumulates_and_persists(tmp_path):
    p = tmp_path / "ledger.json"
    led = Ledger(str(p), cap_usd=1.0)
    led.charge(0.30, {"step": "a"})
    led.charge(0.20, {"step": "b"})
    assert led.spent == pytest.approx(0.50)
    assert led.remaining == pytest.approx(0.50)
    # reload from disk preserves state
    led2 = Ledger(str(p), cap_usd=1.0)
    assert led2.spent == pytest.approx(0.50)


def test_charge_over_cap_raises_and_does_not_record(tmp_path):
    led = Ledger(str(tmp_path / "l.json"), cap_usd=1.0)
    led.charge(0.90, {})
    with pytest.raises(BudgetExceeded):
        led.charge(0.20, {})
    assert led.spent == pytest.approx(0.90)  # rejected charge not recorded


def test_would_exceed(tmp_path):
    led = Ledger(str(tmp_path / "l.json"), cap_usd=1.0)
    led.charge(0.80, {})
    assert led.would_exceed(0.25) is True
    assert led.would_exceed(0.10) is False
