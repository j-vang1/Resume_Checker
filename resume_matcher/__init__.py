"""Resume matcher: evidence-based job-fit analysis (not keyword counting)."""

from .analyzer import analyze_resume, analyze_resumes
from .language import detect_language
from .parsers import extract_text_from_bytes, extract_text_from_path
from .report import MatchReport

# Back-compat aliases used by earlier keyword matcher API / tests
from .matching_legacy import MatchResult, score_resume, score_resumes  # noqa: F401

__all__ = [
    "analyze_resume",
    "analyze_resumes",
    "detect_language",
    "extract_text_from_bytes",
    "extract_text_from_path",
    "MatchReport",
    "MatchResult",
    "score_resume",
    "score_resumes",
]
