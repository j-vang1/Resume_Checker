"""Detect the language of job-description or resume text."""

from __future__ import annotations

from langdetect import DetectorFactory, detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException

# Deterministic results across runs
DetectorFactory.seed = 0

# Common ISO 639-1 codes → display names
LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}


def language_display_name(code: str) -> str:
    """Return a human-readable language name for an ISO code."""
    return LANGUAGE_NAMES.get(code.lower(), code.upper())


def detect_language(text: str) -> dict[str, object]:
    """
    Detect the primary language of ``text``.

    Returns a dict with keys:
      - code: ISO language code (or "unknown")
      - name: display name
      - confidence: probability 0–1 (best guess)
      - alternatives: list of {code, name, confidence}
    """
    cleaned = (text or "").strip()
    if len(cleaned) < 20:
        return {
            "code": "unknown",
            "name": "Unknown",
            "confidence": 0.0,
            "alternatives": [],
            "error": "Need at least ~20 characters to detect language.",
        }

    try:
        primary = detect(cleaned)
        langs = detect_langs(cleaned)
        alternatives = [
            {
                "code": lang.lang,
                "name": language_display_name(lang.lang),
                "confidence": round(float(lang.prob), 3),
            }
            for lang in langs
        ]
        confidence = alternatives[0]["confidence"] if alternatives else 0.0
        return {
            "code": primary,
            "name": language_display_name(primary),
            "confidence": confidence,
            "alternatives": alternatives,
            "error": None,
        }
    except LangDetectException as exc:
        return {
            "code": "unknown",
            "name": "Unknown",
            "confidence": 0.0,
            "alternatives": [],
            "error": str(exc),
        }