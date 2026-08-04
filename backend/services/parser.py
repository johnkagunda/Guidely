"""Document parsing service.

Converts supported uploaded file types (.txt, .md, .pdf, .docx) into
clean raw text. Unsupported file types raise a ParsingError which the
route layer converts into an HTTP 400 response.
"""
from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


class ParsingError(Exception):
    """Raised when a file cannot be parsed (unsupported type or corrupted)."""


class UnsupportedFileTypeError(ParsingError):
    pass


class CorruptedDocumentError(ParsingError):
    pass


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def extract_text(file_path: Path, filename: str) -> str:
    """Extract raw text from a file on disk based on its extension.

    Raises UnsupportedFileTypeError or CorruptedDocumentError on failure.
    """
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Supported types: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if ext in (".txt", ".md"):
            return _extract_plain_text(file_path)
        if ext == ".pdf":
            return _extract_pdf(file_path)
        if ext == ".docx":
            return _extract_docx(file_path)
    except ParsingError:
        raise
    except Exception as exc:  # noqa: BLE001 - convert any parser failure
        raise CorruptedDocumentError(
            f"Failed to parse '{filename}': the file appears to be corrupted "
            f"or unreadable ({exc})"
        ) from exc

    # unreachable, but keeps type checkers happy
    raise UnsupportedFileTypeError(f"Unsupported file type '{ext}'")


def _extract_plain_text(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="strict")
    if not text.strip():
        raise CorruptedDocumentError("Document is empty")
    return text


def _extract_pdf(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ParsingError(
            "PDF support requires the 'pypdf' package to be installed"
        ) from exc

    reader = PdfReader(str(file_path))
    if reader.is_encrypted:
        raise CorruptedDocumentError("Encrypted PDFs are not supported")

    pages_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages_text).strip()
    if not text:
        raise CorruptedDocumentError(
            "No extractable text found in PDF (it may be a scanned image)"
        )
    return text


def _extract_docx(file_path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ParsingError(
            "DOCX support requires the 'python-docx' package to be installed"
        ) from exc

    document = docx.Document(str(file_path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs).strip()
    if not text:
        raise CorruptedDocumentError("Document contains no extractable text")
    return text
