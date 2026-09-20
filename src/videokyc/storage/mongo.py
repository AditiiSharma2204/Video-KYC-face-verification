"""MongoDB persistence for KYC outcomes.

Only the minimised record from ``KycSession.to_record()`` is written: verdicts, scores,
masked ID, keyed hash and an audit trail. No images, names, DOBs or transcripts.
"""

from __future__ import annotations

from typing import Any


class MongoStore:
    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db: str = "videokyc",
        collection: str = "sessions",
        ttl_days: int | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            from pymongo import MongoClient

            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self._col = client[db][collection]
        self._col.create_index("session_id", unique=True)
        self._col.create_index([("id_hash", 1), ("outcome", 1)])
        if ttl_days:  # auto-expire records after the retention period
            self._col.create_index("created_at", expireAfterSeconds=ttl_days * 86400, name="retention_ttl")

    def save(self, record: dict[str, Any]) -> None:
        self._col.replace_one({"session_id": record["session_id"]}, record, upsert=True)

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self._col.find_one({"session_id": session_id}, {"_id": 0})

    def count_approved_by_hash(self, id_hash: str) -> int:
        return self._col.count_documents({"id_hash": id_hash, "outcome": "approved"})
