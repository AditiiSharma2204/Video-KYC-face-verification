from __future__ import annotations

from typing import Any, Protocol


class KycStore(Protocol):
    def save(self, record: dict[str, Any]) -> None:
        """Insert or replace the record keyed by ``record['session_id']``."""
        ...

    def get(self, session_id: str) -> dict[str, Any] | None: ...

    def count_approved_by_hash(self, id_hash: str) -> int:
        """How many previously *approved* sessions used this (hashed) ID - duplicate detection."""
        ...
