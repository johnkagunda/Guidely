"""Tests for services.chunker — pure Python, no external dependencies."""
from services.chunker import chunk_document, _slug


def test_chunk_ids_are_sequential_and_slugged():
    text = "Intro line.\n\nSection One\nSome body text here about policy details."
    chunks = chunk_document(text, "My Doc.txt")
    assert all(c.chunk_id.startswith("my-doc-") for c in chunks)
    numbers = [int(c.chunk_id.split("-")[-1]) for c in chunks]
    assert numbers == sorted(numbers)
    assert numbers[0] == 1


def test_section_headers_are_detected():
    text = (
        "Annual Leave\n"
        "Employees are entitled to 21 working days of paid annual leave.\n\n"
        "Sick Leave\n"
        "Employees receive 10 paid sick days per year."
    )
    chunks = chunk_document(text, "policy.txt")
    sections = {c.section for c in chunks}
    assert "Annual Leave" in sections
    assert "Sick Leave" in sections


def test_markdown_headers_are_detected():
    text = "# Getting Started\nWelcome to the team.\n\n## Setup\nInstall the tools."
    chunks = chunk_document(text, "guide.md")
    sections = {c.section for c in chunks}
    assert "Getting Started" in sections
    assert "Setup" in sections


def test_long_section_is_split_into_multiple_chunks():
    long_body = "word " * 1600  # well beyond the ~750-word max chunk size
    text = f"Big Section\n{long_body}"
    chunks = chunk_document(text, "big.txt", target_words=550, max_words=750)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 750 + 50  # allow small merge slack


def test_short_trailing_chunk_gets_merged():
    # A section just over the target should not leave a tiny orphan chunk.
    words = ["word"] * 560  # 10 words over target of 550
    text = "Section\n" + " ".join(words)
    chunks = chunk_document(text, "doc.txt", target_words=550, max_words=750)
    assert len(chunks) == 1  # small remainder merged back in


def test_empty_document_produces_no_chunks():
    assert chunk_document("   \n\n  ", "empty.txt") == []


def test_chunk_metadata_shape():
    text = "Policy\nSome content about the policy."
    chunks = chunk_document(text, "policy.txt")
    d = chunks[0].to_dict()
    assert set(d.keys()) == {"chunk_id", "document", "section", "text"}
    assert d["document"] == "policy.txt"


def test_slug_strips_extension_and_special_chars():
    assert _slug("Company Policy v2.docx") == "company-policy-v2"
    assert _slug("faq.txt") == "faq"
