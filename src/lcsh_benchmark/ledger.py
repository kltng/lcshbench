"""Shared spend ledger with a hard USD cap, persisted to disk.

Every paid backend call routes its cost through charge(); the ledger refuses
any charge that would push cumulative spend over the cap, so a run aborts
cleanly (cached partial results survive) rather than overspending.
"""
import json
import threading
from pathlib import Path


class BudgetExceeded(Exception):
    """Raised when a charge would exceed the configured cap."""


class Ledger:
    def __init__(self, path: str, cap_usd: float):
        self.path = Path(path)
        self.cap_usd = float(cap_usd)
        self._lock = threading.Lock()      # charge() is called from worker threads
        self._entries: list[dict] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = data.get("entries", [])

    @property
    def spent(self) -> float:
        return round(sum(e["usd"] for e in self._entries), 6)

    @property
    def remaining(self) -> float:
        return round(self.cap_usd - self.spent, 6)

    def would_exceed(self, usd: float) -> bool:
        return self.spent + usd > self.cap_usd + 1e-9

    def charge(self, usd: float, meta: dict) -> None:
        with self._lock:
            if self.would_exceed(usd):
                raise BudgetExceeded(
                    f"charge ${usd:.4f} would exceed cap ${self.cap_usd:.2f} "
                    f"(spent ${self.spent:.4f})"
                )
            self._entries.append({"usd": round(float(usd), 6), **meta})
            self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"cap_usd": self.cap_usd, "spent": self.spent,
                        "entries": self._entries}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
