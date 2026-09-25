"""ATS compatibility assessment — kept separate from job-fit quality."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .resume_parse import ParsedResume

_STANDARD_SECTIONS = ("experience", "education", "skills")


@dataclass
class ATSAssessment:
    rating: str  # Strong / Moderate / Weak
    score: float  # 0-100 informational only
    findings: list[str] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)
    notes: str = (
        "ATS compatibility is separate from role-fit quality. "
        "Formatting issues should not dominate the candidate-fit assessment."
    )

    def to_dict(self) -> dict:
        return asdict(self)


def assess_ats(resume: ParsedResume, original_filename: str = "") -> ATSAssessment:
    findings: list[str] = []
    positives: list[str] = []
    score = 70.0

    sections = set(resume.sections_found)
    for required in _STANDARD_SECTIONS:
        if required in sections:
            positives.append(f"Standard section present: {required.title()}")
            score += 5
        else:
            findings.append(f"Missing or unrecognized standard section: {required.title()}")
            score -= 10

    contact = resume.contact_hints
    if contact.get("email"):
        positives.append("Email address detected")
        score += 5
    else:
        findings.append("No email address detected in parsed text")
        score -= 8
    if contact.get("phone"):
        positives.append("Phone number detected")
        score += 3
    else:
        findings.append("No phone number detected in parsed text")
        score -= 4

    if resume.date_spans:
        positives.append("Employment date ranges detected")
        score += 4
    else:
        findings.append("No clear employment date ranges found — ATS parsers often expect dates")
        score -= 6

    text = resume.raw_text
    # Unusual symbols / glyph noise
    weird = len(re.findall(r"[■▪▫◆◇★☆✓✔✕✖→←⇒]|[^\x00-\x7F\u00C0-\u024F]", text))
    if weird > 15:
        findings.append("Unusual symbols detected that may confuse ATS parsers")
        score -= 8

    # Table-like dense pipes
    if text.count("|") > 12:
        findings.append("Pipe-heavy / table-like layout detected — may parse poorly in ATS")
        score -= 10

    # Very short extractable text suggests image-heavy/scanned PDF
    if len(text.strip()) < 200:
        findings.append(
            "Very little extractable text — resume may be image-heavy or scanned (OCR needed)"
        )
        score -= 25

    if original_filename.lower().endswith(".doc"):
        findings.append("Legacy .doc format can be less reliable than .docx/.pdf for ATS")
        score -= 5

    if resume.bullets:
        positives.append(f"{len(resume.bullets)} parseable experience/project bullets")
        score += 4

    score = max(0.0, min(100.0, score))
    if score >= 75:
        rating = "Strong"
    elif score >= 50:
        rating = "Moderate"
    else:
        rating = "Weak"

    return ATSAssessment(rating=rating, score=round(score, 1), findings=findings, positives=positives)
