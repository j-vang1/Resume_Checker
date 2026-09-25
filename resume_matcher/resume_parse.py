"""Parse resume text into structured sections, bullets, skills, and chronology."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

_SECTION_HEADERS = {
    "experience": re.compile(
        r"(?i)^(professional\s+)?(experience|work\s+history|employment|"
        r"work\s+experience|career\s+history)\s*:?\s*$"
    ),
    "skills": re.compile(
        r"(?i)^(skills|technical\s+skills|core\s+competencies|technologies|"
        r"tools|proficiencies)\s*:?\s*$"
    ),
    "projects": re.compile(r"(?i)^(projects|selected\s+projects|key\s+projects)\s*:?\s*$"),
    "education": re.compile(r"(?i)^(education|academic|degrees?)\s*:?\s*$"),
    "summary": re.compile(
        r"(?i)^(summary|profile|objective|about|professional\s+summary)\s*:?\s*$"
    ),
    "certifications": re.compile(r"(?i)^(certifications?|licenses?)\s*:?\s*$"),
}

_BULLET_RE = re.compile(r"^\s*(?:[-*•●▪◦◦]|\d+[.)])\s+(.*)$")
_DATE_RANGE_RE = re.compile(
    r"(?i)("
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}"
    r"|\d{1,2}/\d{4}"
    r"|\d{4})"
    r"\s*[-–—to]+\s*"
    r"("
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}"
    r"|\d{1,2}/\d{4}"
    r"|\d{4}|present|current|now)"
)
_MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


@dataclass
class Bullet:
    text: str
    section: str = "experience"
    role_context: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DateSpan:
    start: datetime | None
    end: datetime | None
    raw: str

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "raw": self.raw,
        }


@dataclass
class ParsedResume:
    raw_text: str
    bullets: list[Bullet] = field(default_factory=list)
    skills_listed: list[str] = field(default_factory=list)
    sections_found: list[str] = field(default_factory=list)
    date_spans: list[DateSpan] = field(default_factory=list)
    contact_hints: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "bullets": [b.to_dict() for b in self.bullets],
            "skills_listed": self.skills_listed,
            "sections_found": self.sections_found,
            "date_spans": [d.to_dict() for d in self.date_spans],
            "contact_hints": self.contact_hints,
        }


def _parse_date_token(token: str) -> datetime | None:
    token = token.strip().lower()
    if token in {"present", "current", "now"}:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    m = re.match(r"([a-z]+)\.?\s+(\d{4})", token)
    if m:
        month = _MONTH_MAP.get(m.group(1)[:3], 1)
        return datetime(int(m.group(2)), month, 1)
    m = re.match(r"(\d{1,2})/(\d{4})", token)
    if m:
        return datetime(int(m.group(2)), int(m.group(1)), 1)
    if re.fullmatch(r"\d{4}", token):
        return datetime(int(token), 1, 1)
    return None


def _extract_skills_from_line(line: str) -> list[str]:
    # Split on common delimiters
    parts = re.split(r"[,|/•;]", line)
    skills: list[str] = []
    for part in parts:
        skill = part.strip(" .-:\t")
        if 1 < len(skill) < 40 and not skill.lower().startswith("skills"):
            skills.append(skill)
    return skills


def parse_resume(text: str) -> ParsedResume:
    """Structure a resume into bullets, listed skills, dates, and sections."""
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")

    current_section = "summary"
    sections_found: list[str] = []
    bullets: list[Bullet] = []
    skills_listed: list[str] = []
    date_spans: list[DateSpan] = []
    role_context = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        matched_section = None
        for name, pattern in _SECTION_HEADERS.items():
            if pattern.match(stripped):
                matched_section = name
                break
        if matched_section:
            current_section = matched_section
            if matched_section not in sections_found:
                sections_found.append(matched_section)
            continue

        # Date ranges often appear on role header lines
        for match in _DATE_RANGE_RE.finditer(stripped):
            start = _parse_date_token(match.group(1))
            end = _parse_date_token(match.group(2))
            date_spans.append(DateSpan(start=start, end=end, raw=match.group(0)))
            role_context = stripped

        bullet_match = _BULLET_RE.match(stripped)
        if bullet_match:
            bullet_text = bullet_match.group(1).strip()
            if len(bullet_text) > 8:
                bullets.append(
                    Bullet(text=bullet_text, section=current_section, role_context=role_context)
                )
            continue

        if current_section == "skills":
            skills_listed.extend(_extract_skills_from_line(stripped))
            # Keep the full skills line as evidence (e.g. "thermal plunger/pedestal design")
            if len(stripped) > 12:
                bullets.append(
                    Bullet(text=stripped, section="skills", role_context="Skills")
                )
            continue

        # Inline "Skills: Python, SQL, ..." without a dedicated section header
        inline_skills = re.match(r"(?i)^(?:technical\s+)?skills\s*:\s*(.+)$", stripped)
        if inline_skills:
            if "skills" not in sections_found:
                sections_found.append("skills")
            skills_listed.extend(_extract_skills_from_line(inline_skills.group(1)))
            bullets.append(
                Bullet(text=inline_skills.group(1).strip(), section="skills", role_context="Skills")
            )
            continue

        # Summary / profile paragraphs are strong evidence
        if current_section == "summary" and len(stripped) > 40:
            bullets.append(
                Bullet(text=stripped, section="summary", role_context="Summary")
            )
            continue

        # Non-bulleted experience lines that look like achievements
        if current_section in {"experience", "projects"} and len(stripped) > 40:
            if stripped[0].isupper() or stripped[0].isdigit():
                # Skip pure title/company lines with dates only
                if _DATE_RANGE_RE.search(stripped) and len(stripped) < 100:
                    role_context = stripped
                    continue
                bullets.append(
                    Bullet(text=stripped, section=current_section, role_context=role_context)
                )

    # Deduplicate skills (case-insensitive)
    deduped_skills: list[str] = []
    seen_skills: set[str] = set()
    for skill in skills_listed:
        key = skill.lower()
        if key not in seen_skills:
            seen_skills.add(key)
            deduped_skills.append(skill)

    # If no bullets found, split paragraphs into pseudo-bullets
    if not bullets:
        for para in re.split(r"\n{2,}|(?<=\.)\s+", raw):
            para = para.strip()
            if len(para) > 30:
                bullets.append(Bullet(text=para, section="experience"))

    contact_hints = {
        "email": bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", raw)),
        "phone": bool(re.search(r"\+?\d[\d\s().-]{7,}\d", raw)),
        "linkedin": bool(re.search(r"(?i)linkedin\.com", raw)),
    }

    return ParsedResume(
        raw_text=raw,
        bullets=bullets,
        skills_listed=deduped_skills,
        sections_found=sections_found,
        date_spans=date_spans,
        contact_hints=contact_hints,
    )
