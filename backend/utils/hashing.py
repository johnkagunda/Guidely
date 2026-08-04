"""Content hashing helpers used to detect whether a document has changed."""
from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    """Return the SHA256 hex digest of a text string (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()
