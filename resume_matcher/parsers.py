"""Extract plain text from PDF and Word resume files."""

from __future__ import annotations

import io
import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}

# Fancy layouts often yield very little text — treat below this as a failed extract
MIN_USEFUL_CHARS = 80


class UnsupportedFileTypeError(ValueError):
    """Raised when a file type cannot be parsed."""


class EmptyExtractError(ValueError):
    """Raised when a file opens but yields no usable text."""


def _normalize_ext(filename: str) -> str:
    return Path(filename).suffix.lower()


def _clean_extracted(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    # Normalize whitespace while keeping paragraph breaks
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    # Drop near-empty lines of only bullets/ornaments
    lines = [ln for ln in lines if ln and not re.fullmatch(r"[|•●▪◦\-_/\\]+", ln)]
    return "\n".join(lines).strip()


def _extract_with_pypdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — try next page / fallback extractor
            text = ""
        if text.strip():
            pages.append(text)
    return _clean_extracted("\n".join(pages))


def _extract_with_pdfplumber(data: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""
    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            # layout=True preserves reading order better on multi-column resumes
            try:
                text = page.extract_text(layout=True) or page.extract_text() or ""
            except TypeError:
                text = page.extract_text() or ""
            if text and text.strip():
                parts.append(text)
            # Also pull table cell text (common in designed resumes)
            try:
                tables = page.extract_tables() or []
            except Exception:  # noqa: BLE001
                tables = []
            for table in tables:
                for row in table:
                    cells = [c.strip() for c in row if c and str(c).strip()]
                    if cells:
                        parts.append(" | ".join(cells))
    return _clean_extracted("\n".join(parts))


def extract_text_from_pdf(data: bytes) -> str:
    """
    Extract text from a PDF using multiple engines and keep the richest result.

    Designed resumes (columns, timelines, icons) often confuse a single extractor.
    """
    candidates = []
    for extractor in (_extract_with_pdfplumber, _extract_with_pypdf):
        try:
            text = extractor(data)
        except Exception:  # noqa: BLE001
            text = ""
        if text:
            candidates.append(text)

    if not candidates:
        return ""

    # Prefer the longest extract that still looks like a resume (has letters)
    def _quality(t: str) -> tuple[int, int]:
        letters = sum(ch.isalpha() for ch in t)
        return (letters, len(t))

    return max(candidates, key=_quality)


def extract_text_from_docx(data: bytes) -> str:
    """Extract text from a DOCX byte stream."""
    document = Document(io.BytesIO(data))
    parts: list[str] = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return _clean_extracted("\n".join(parts))


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """
    Extract text from uploaded file bytes based on filename extension.

    Supports ``.pdf`` and ``.docx``. Legacy ``.doc`` is not supported;
    convert those to DOCX or PDF first.
    """
    ext = _normalize_ext(filename)
    if ext == ".pdf":
        text = extract_text_from_pdf(data)
    elif ext == ".docx":
        text = extract_text_from_docx(data)
    elif ext == ".doc":
        raise UnsupportedFileTypeError(
            f"'{filename}' is a legacy .doc file. Please convert it to .docx or .pdf."
        )
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}' for '{filename}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS - {'.doc'}))}."
        )

    if len(text) < MIN_USEFUL_CHARS:
        raise EmptyExtractError(
            f"Could not extract usable text from '{filename}' "
            f"({len(text)} characters). Designed/column PDFs and scanned resumes "
            "often fail text extraction — export a text-based PDF, upload .docx, "
            "or paste the resume text."
        )
    return text


def extract_text_from_path(path: str | Path) -> str:
    """Extract text from a file on disk."""
    path = Path(path)
    return extract_text_from_bytes(path.read_bytes(), path.name)
