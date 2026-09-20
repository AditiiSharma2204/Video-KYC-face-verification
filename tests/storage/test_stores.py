import json
from datetime import datetime, timezone

import pytest

from tests.session.conftest import Harness
from videokyc.storage.memory import MemoryStore
from videokyc.storage.mongo import MongoStore

mongomock = pytest.importorskip("mongomock")


def record(sid="s1", outcome="approved", id_hash="h1"):
    return {"session_id": sid, "created_at": datetime(2026, 9, 20, tzinfo=timezone.utc), "outcome": outcome,
            "id_hash": id_hash, "voice": [{"question": "age", "verdict": "pass"}]}


@pytest.fixture(params=["memory", "mongo"])
def store(request):
    if request.param == "memory":
        return MemoryStore()
    return MongoStore(client=mongomock.MongoClient(), db="test", collection="sessions")


def test_save_get_roundtrip_and_upsert(store):
    assert store.get("s1") is None
    store.save(record())
    assert store.get("s1")["voice"][0]["verdict"] == "pass"
    store.save({**record(), "outcome": "rejected"})  # same session_id replaces, never duplicates
    assert store.get("s1")["outcome"] == "rejected"


def test_get_does_not_expose_mongo_internal_id(store):
    store.save(record())
    assert "_id" not in store.get("s1")


def test_duplicate_detection_counts_only_approved(store):
    store.save(record("a", "approved", "same"))
    store.save(record("b", "rejected", "same"))
    store.save(record("c", "approved", "other"))
    assert store.count_approved_by_hash("same") == 1
    assert store.count_approved_by_hash("nobody") == 0


def test_memory_store_returns_copies():
    s = MemoryStore()
    s.save(record())
    s.get("s1")["voice"].clear()
    assert s.get("s1")["voice"]


def test_mongo_creates_indexes_and_optional_ttl():
    client = mongomock.MongoClient()
    MongoStore(client=client, db="d", collection="c", ttl_days=30)
    idx = client["d"]["c"].index_information()
    assert any("session_id" in str(v["key"]) and v.get("unique") for v in idx.values())
    assert idx["retention_ttl"]["expireAfterSeconds"] == 30 * 86400


def test_engine_persists_to_mongo_end_to_end():
    client = mongomock.MongoClient()
    h = Harness()
    h.store = h.engine.store = MongoStore(client=client, db="kyc", collection="sessions")
    s = h.to_voice()
    h.answer_all(s.id)
    stored = client["kyc"]["sessions"].find_one({"session_id": s.id})
    assert stored["outcome"] == "approved" and stored["id_masked"] == "XXXXXX234F"
    blob = json.dumps(stored, default=str)
    assert "Rahul" not in blob and "ABCPE1234F" not in blob
