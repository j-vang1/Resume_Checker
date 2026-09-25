"""Parse a job description into weighted, structured requirements."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from .concepts import (
    CORE_HINTS,
    GENERIC_PHRASES,
    IMPORTANT_HINTS,
    PREFERRED_HINTS,
    related_concepts_for_requirement,
)

_BULLET_RE = re.compile(r"^\s*(?:[-*•●▪◦]|\d+[.)])\s+")
_SECTION_RE = re.compile(
    r"(?im)^(minimum qualifications|required qualifications|requirements|"
    r"responsibilities|what you.?ll do|what you will do|about the role|"
    r"preferred qualifications|nice to haves?|bonus|qualifications|"
    r"skills|experience|who you are|must[- ]haves?)\s*:?\s*$"
)


class Importance(str, Enum):
    CORE = "Core"
    IMPORTANT = "Important"
    PREFERRED = "Preferred"
    SUPPORTING = "Supporting"
    GENERIC = "Generic"


IMPORTANCE_WEIGHT = {
    Importance.CORE: 1.0,
    Importance.IMPORTANT: 0.75,
    Importance.PREFERRED: 0.45,
    Importance.SUPPORTING: 0.25,
    Importance.GENERIC: 0.05,
}


@dataclass
class Requirement:
    text: str
    importance: Importance
    related_concepts: list[str] = field(default_factory=list)
    source_section: str = ""
    weight: float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["importance"] = self.importance.value
        return d


def _is_generic(line: str) -> bool:
    lower = line.lower()
    return any(p.lower() in lower for p in GENERIC_PHRASES) or len(line.split()) < 3


def _classify_line(line: str, section: str) -> Importance:
    if _is_generic(line):
        return Importance.GENERIC

    lower = line.lower()
    section_l = (section or "").lower()

    if any(h in lower for h in PREFERRED_HINTS) or "preferred" in section_l or "nice" in section_l or "bonus" in section_l:
        return Importance.PREFERRED
    if any(h in lower for h in CORE_HINTS) or "required" in section_l or "minimum" in section_l or "responsibilities" in section_l or "what you" in section_l:
        return Importance.CORE
    if any(h in lower for h in IMPORTANT_HINTS) or "qualification" in section_l or "skills" in section_l:
        return Importance.IMPORTANT
    if "experience" in section_l:
        return Importance.IMPORTANT
    # Default: substantive skill lines are Important; short soft skills Supporting
    if any(
        kw in lower
        for kw in (
            "python",
            "hardware",
            "validation",
            "debug",
            "analysis",
            "design",
            "aws",
            "sql",
            "test",
            "doe",
            "failure",
            "engineer",
        )
    ):
        return Importance.IMPORTANT
    return Importance.SUPPORTING


def _normalize_line(line: str) -> str:
    line = _BULLET_RE.sub("", line).strip()
    line = re.sub(r"\s+", " ", line)
    return line.strip(" ;,")


def _split_lines(text: str) -> list[str]:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for block in raw.split("\n"):
        block = block.strip()
        if not block:
            continue
        # Also split long semicolon-separated skill lists
        if block.count(";") >= 2 and len(block) < 400:
            parts = [p.strip() for p in block.split(";") if p.strip()]
            lines.extend(parts)
        else:
            lines.append(block)
    return lines


def parse_job_description(text: str) -> list[Requirement]:
    """
    Break a job description into structured, weighted requirements.

    Skips boilerplate / generic posting language and elevates core duties.
    """
    lines = _split_lines(text)
    section = "general"
    requirements: list[Requirement] = []
    seen: set[str] = set()

    for line in lines:
        section_match = _SECTION_RE.match(line)
        if section_match:
            section = section_match.group(1).lower()
            continue

        cleaned = _normalize_line(line)
        if len(cleaned) < 8:
            continue

        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)

        importance = _classify_line(cleaned, section)
        if importance == Importance.GENERIC:
            continue

        related = related_concepts_for_requirement(cleaned)
        req = Requirement(
            text=cleaned,
            importance=importance,
            related_concepts=related,
            source_section=section,
            weight=IMPORTANCE_WEIGHT[importance],
        )
        requirements.append(req)

    # If the JD was a single paragraph with few newlines, extract noun phrases-ish chunks
    if len(requirements) < 3 and text and len(text) > 80:
        requirements = _fallback_chunk_parse(text)

    # Cap at a reasonable number; keep highest weight first
    requirements.sort(key=lambda r: (r.weight, len(r.related_concepts)), reverse=True)
    return requirements[:40]


def _fallback_chunk_parse(text: str) -> list[Requirement]:
    """Split dense paragraphs on periods / commas when bullets are missing."""
    sentences = re.split(r"(?<=[.:;])\s+", text.strip())
    reqs: list[Requirement] = []
    for sentence in sentences:
        cleaned = _normalize_line(sentence)
        if len(cleaned) < 12 or _is_generic(cleaned):
            continue
        importance = _classify_line(cleaned, "general")
        if importance == Importance.GENERIC:
            continue
        reqs.append(
            Requirement(
                text=cleaned,
                importance=importance,
                related_concepts=related_concepts_for_requirement(cleaned),
                source_section="general",
                weight=IMPORTANCE_WEIGHT[importance],
            )
        )
    return reqs
