"""Resume matcher: score resumes against a job description."""

from .language import detect_language
from .matching import MatchResult, score_resume, score_resumes
from .parsers import extract_text_from_bytes, extract_text_from_path

__all__ = [
    "detect_language",
    "extract_text_from_bytes",
    "extract_text_from_path",
    "MatchResult",
    "score_resume",
    "score_resumes",
]