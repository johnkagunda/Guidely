"""Document chunking service.

Token counting via a real tokenizer is unnecessary complexity for this
project, so chunk size is approximated using a word count.

Approximation used: 1 token ~= 0.75 English words (i.e. ~1.33 tokens per
word), which is the commonly cited rule of thumb for English text with
GPT/LLaMA-style BPE tokenizers. That means a target of 500-1000 tokens
per chunk maps to roughly 375-750 words. We target the midpoint (~550
words) per chunk with a soft cap of 750 words, and merge short trailing
chunks so sections aren't left with tiny orphan chunks.

Chunking is "section aware": documents are first split on headings
(markdown '#' headers, or short standalone lines that look like a
title/heading) so each chunk carries an accurate `section` label. Long
sections are further split by word count; short sections are kept whole.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

MIN_WORDS_PER_CHUNK_TARGET = 375  # ~500 tokens
MAX_WORDS_PER_CHUNK_TARGET = 750  # ~1000 tokens
TARGET_WORDS_PER_CHUNK = 550  # ~730 tokens, comfortable midpoint

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")  # markdown headings
# A short line (<=80 chars), no trailing sentence punctuation, likely a title
_LOOSE_HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9 ,&/'()\-]{2,79}$")


@dataclass
class Chunk:
    chunk_id: str
    document: str
    section: Optional[str]
    text: str
    order: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "section": self.section,
            "text": self.text,
        }


@dataclass
class _Section:
    title: Optional[str]
    lines: List[str] = field(default_factory=list)


def _split_into_sections(text: str) -> List[_Section]:
    """Split raw text into (heading, body) sections."""
    lines = text.splitlines()
    sections: List[_Section] = []
    current = _Section(title=None)

    for raw_line in lines:
        line = raw_line.rstrip()
        heading_match = _HEADING_RE.match(line)
        is_loose_heading = (
            not heading_match
            and line.strip() != ""
            and len(line.strip()) <= 80
            and not line.strip().endswith((".", ",", ";", ":"))
            and _LOOSE_HEADING_RE.match(line.strip())
            and (not current.lines or current.lines[-1].strip() == "")
        )

        if heading_match:
            if current.lines:
                sections.append(current)
            current = _Section(title=heading_match.group(1).strip())
        elif is_loose_heading:
            if current.lines:
                sections.append(current)
            current = _Section(title=line.strip())
        else:
            current.lines.append(line)

    if current.lines:
        sections.append(current)

    if not sections:
        sections = [_Section(title=None, lines=lines)]

    return sections


def _word_chunks(words: List[str], target: int, max_words: int) -> List[List[str]]:
    """Greedily group words into chunks close to `target`, never exceeding max."""
    if not words:
        return []
    chunks: List[List[str]] = []
    current: List[str] = []
    for word in words:
        current.append(word)
        if len(current) >= target:
            chunks.append(current)
            current = []
    if current:
        # merge a small trailing remainder into the previous chunk if possible
        if chunks and len(current) < MIN_WORDS_PER_CHUNK_TARGET // 2 and \
                len(chunks[-1]) + len(current) <= max_words:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return chunks


def chunk_document(
    text: str,
    document_name: str,
    target_words: int = TARGET_WORDS_PER_CHUNK,
    max_words: int = MAX_WORDS_PER_CHUNK_TARGET,
) -> List[Chunk]:
    """Split a document's raw text into metadata-rich chunks."""
    sections = _split_into_sections(text)
    chunks: List[Chunk] = []
    order = 1

    for section in sections:
        body = "\n".join(section.lines).strip()
        if not body:
            continue
        words = body.split()
        for word_group in _word_chunks(words, target_words, max_words):
            chunk_text = " ".join(word_group).strip()
            if not chunk_text:
                continue
            chunk_id = f"{_slug(document_name)}-{order:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document=document_name,
                    section=section.title,
                    text=chunk_text,
                    order=order,
                )
            )
            order += 1

    return chunks


def _slug(name: str) -> str:
    base = re.sub(r"\.[a-zA-Z0-9]+$", "", name)  # strip extension
    base = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()
    return base or "document"
