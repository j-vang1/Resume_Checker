"""Unit tests for resume matching and language detection."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from docx import Document

from resume_matcher.language import detect_language
from resume_matcher.matching import extract_keywords, score_resume, score_resumes
from resume_matcher.parsers import extract_text_from_bytes, extract_text_from_docx


JOB_EN = """
We are hiring a Senior Python Software Engineer with experience in FastAPI,
Django, PostgreSQL, Docker, Kubernetes, and AWS. Candidates should know
REST APIs, microservices, CI/CD, pytest, and distributed systems.
"""

RESUME_STRONG = """
Jane Doe — Senior Software Engineer
Experienced Python developer specializing in FastAPI and Django.
Built microservices on AWS with Docker and Kubernetes.
Strong with PostgreSQL, REST APIs, CI/CD pipelines, and pytest.
"""

RESUME_WEAK = """
John Smith — Marketing Coordinator
Managed social media campaigns, wrote newsletters, organized events,
and coordinated brand partnerships for retail clients.
"""

JOB_ES = """
Buscamos un ingeniero de software con experiencia en Python, Django,
bases de datos PostgreSQL y despliegue en la nube con Docker.
"""


def test_detect_english_job_description():
    result = detect_language(JOB_EN)
    assert result["code"] == "en"
    assert result["name"] == "English"
    assert result["confidence"] > 0.5


def test_detect_spanish_job_description():
    result = detect_language(JOB_ES)
    assert result["code"] == "es"
    assert result["confidence"] > 0.5


def test_detect_too_short():
    result = detect_language("hi")
    assert result["code"] == "unknown"


def test_strong_resume_greenlit_at_50():
    result = score_resume(JOB_EN, RESUME_STRONG, filename="strong.pdf", threshold=50)
    assert result.match_percent >= 50
    assert result.greenlit is True
    assert "python" in result.matched_keywords


def test_weak_resume_rejected_at_50():
    result = score_resume(JOB_EN, RESUME_WEAK, filename="weak.pdf", threshold=50)
    assert result.match_percent < 50
    assert result.greenlit is False


def test_threshold_is_tunable():
    result = score_resume(JOB_EN, RESUME_STRONG, filename="strong.pdf", threshold=99)
    # Even a strong resume may not hit 99%; assert threshold logic works
    assert result.greenlit == (result.match_percent >= 99)


def test_score_resumes_sorted_and_filtered():
    results = score_resumes(
        JOB_EN,
        [("strong.pdf", RESUME_STRONG), ("weak.pdf", RESUME_WEAK)],
        threshold=50,
    )
    assert len(results) == 2
    assert results[0].filename == "strong.pdf"
    assert results[0].greenlit is True
    assert results[1].filename == "weak.pdf"
    assert results[1].greenlit is False


def test_empty_inputs():
    empty_job = score_resume("", RESUME_STRONG, filename="a.pdf")
    assert empty_job.error
    assert empty_job.greenlit is False

    empty_resume = score_resume(JOB_EN, "", filename="b.pdf")
    assert empty_resume.error
    assert empty_resume.greenlit is False


def test_extract_keywords_finds_skills():
    keywords = extract_keywords(JOB_EN)
    joined = " ".join(keywords)
    assert "python" in joined
    assert "fastapi" in joined or "django" in joined


def test_docx_parser_roundtrip():
    document = Document()
    document.add_paragraph("Alice Example")
    document.add_paragraph("Skills: Python, FastAPI, PostgreSQL")
    buffer = io.BytesIO()
    document.save(buffer)
    text = extract_text_from_docx(buffer.getvalue())
    assert "Alice Example" in text
    assert "Python" in text


def test_extract_text_from_bytes_docx():
    document = Document()
    document.add_paragraph("Resume text for parsing.")
    buffer = io.BytesIO()
    document.save(buffer)
    text = extract_text_from_bytes(buffer.getvalue(), "resume.docx")
    assert "Resume text" in text


def test_unsupported_extension():
    with pytest.raises(Exception):
        extract_text_from_bytes(b"not a resume", "resume.txt")
