"""PII helpers: masking for display/storage and keyed hashing for de-duplication.

We never persist a raw government ID number. Stored records carry a masked form
(last 4 characters) plus an HMAC so the same ID can be recognised across
sessions without being recoverable from the database.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re


def normalize_id(value: str) -> str:
    """Upper-case and strip everything that is not a letter or digit."""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def mask_id(value: str, visible: int = 4) -> str:
    """'ABCDE1234F' -> 'XXXXXX234F' style masking (keeps the last ``visible`` chars)."""
    v = normalize_id(value)
    if len(v) <= visible:
        return "X" * len(v)
    return "X" * (len(v) - visible) + v[-visible:]


def hash_id(value: str, key: bytes | None = None) -> str:
    """Keyed hash of a normalised ID. Set ``KYC_HASH_KEY`` in production."""
    key = key or os.environ.get("KYC_HASH_KEY", "dev-only-insecure-key").encode()
    return hmac.new(key, normalize_id(value).encode(), hashlib.sha256).hexdigest()
