"""Tests for utils.hashing — pure Python, no external dependencies."""
from utils.hashing import sha256_text, sha256_bytes


def test_same_text_produces_same_hash():
    assert sha256_text("hello world") == sha256_text("hello world")


def test_different_text_produces_different_hash():
    assert sha256_text("hello") != sha256_text("world")


def test_hash_is_64_char_hex():
    h = sha256_text("Guidely")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_bytes_and_text_hash_agree_for_utf8():
    text = "Employees receive 21 days of leave."
    assert sha256_text(text) == sha256_bytes(text.encode("utf-8"))


def test_whitespace_change_alters_hash():
    assert sha256_text("hello world") != sha256_text("hello  world")
