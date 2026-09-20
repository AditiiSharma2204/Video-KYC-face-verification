import pytest

from videokyc import factory
from videokyc.storage.memory import MemoryStore


def test_refuses_persistent_store_without_a_hash_key(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    monkeypatch.delenv("KYC_HASH_KEY", raising=False)
    with pytest.raises(RuntimeError, match="KYC_HASH_KEY"):
        factory.build_engine()


def test_store_defaults_to_memory_without_mongo_uri(monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    assert isinstance(factory.build_store(), MemoryStore)
