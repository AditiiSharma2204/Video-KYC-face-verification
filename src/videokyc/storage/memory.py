from __future__ import annotations

import copy
from typing import Any


class MemoryStore:
    """Non-persistent store for development and tests."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def save(self, record: dict[str, Any]) -> None:
        self._records[record["session_id"]] = copy.deepcopy(record)

    def get(self, session_id: str) -> dict[str, Any] | None:
        rec = self._records.get(session_id)
        return copy.deepcopy(rec) if rec else None

    def count_approved_by_hash(self, id_hash: str) -> int:
        return sum(
            1 for r in self._records.values() if r.get("id_hash") == id_hash and r.get("outcome") == "approved"
        )
