"""Signal detectors: seniority, impact, ownership style, unsupported claims, consistency."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .evidence import EvidenceStrength, RequirementEvidence
from .jd_parser import Importance, Requirement
from .resume_parse import ParsedResume

_JUNIOR = ("assisted", "supported", "helped", "shadowed", "followed", "under supervision", "intern")
_MID = ("independently", "owned", "designed", "debugged", "implemented", "developed", "built", "delivered")
_SENIOR = ("led", "defined", "architected", "drove", "established", "mentored", "coached", "cross-functional decisions")
_STAFF = (
    "technical direction",
    "organization-wide",
    "company-wide",
    "across teams",
    "standards",
    "platform strategy",
    "ambiguous",
    "principal",
    "staff engineer",
)

_JD_SENIORITY = (
    (r"(?i)\b(staff|principal|distinguished)\b", "Staff / Principal"),
    (r"(?i)\b(senior|sr\.?|lead)\b", "Senior"),
    (r"(?i)\b(mid[- ]?level|intermediate)\b", "Mid-level"),
    (r"(?i)\b(junior|jr\.?|entry[- ]?level|intern)\b", "Junior"),
)


@dataclass
class SeniorityAssessment:
    target_level: str
    demonstrated_level: str
    alignment: str  # Strong / Moderate / Weak
    signals: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UnsupportedClaim:
    skill: str
    note: str = "Skill listed but not demonstrated."

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConsistencyIssue:
    kind: str
    detail: str
    severity: str = "review"  # review | warning

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BulletAnalysis:
    bullet: str
    relevance: str
    technical_depth: str
    ownership: str
    impact: str
    clarity: str
    specificity: str
    feedback: str

    def to_dict(self) -> dict:
        return asdict(self)


def detect_target_seniority(job_text: str) -> str:
    for pattern, label in _JD_SENIORITY:
        if re.search(pattern, job_text or ""):
            return label
    return "Mid-level"


def assess_seniority(resume: ParsedResume, job_text: str) -> SeniorityAssessment:
    text = resume.raw_text.lower()
    signals: list[str] = []
    staff = sum(1 for s in _STAFF if s in text)
    senior = sum(1 for s in _SENIOR if s in text)
    mid = sum(1 for s in _MID if s in text)
    junior = sum(1 for s in _JUNIOR if s in text)

    if staff >= 2:
        demonstrated = "Staff / Principal"
        signals.append("Multiple org/architecture ownership signals")
    elif senior >= 3 or (senior >= 1 and mid >= 3):
        demonstrated = "Senior"
        signals.append("Leadership / ownership language across multiple bullets")
    elif mid >= 2:
        demonstrated = "Mid-level"
        signals.append("Independent delivery and design language")
    elif junior >= 1 and mid == 0:
        demonstrated = "Junior"
        signals.append("Support / assist language dominates")
    else:
        demonstrated = "Mid-level"
        signals.append("Mixed ownership signals; defaulting to mid-level")

    target = detect_target_seniority(job_text)
    order = ["Junior", "Mid-level", "Senior", "Staff / Principal"]
    try:
        gap = abs(order.index(demonstrated) - order.index(target))
    except ValueError:
        gap = 1
    alignment = "Strong" if gap == 0 else ("Moderate" if gap == 1 else "Weak")
    notes = (
        f"Resume demonstrates approximately {demonstrated} ownership; "
        f"target role signals {target}. Assessment uses action language, not titles alone."
    )
    return SeniorityAssessment(
        target_level=target,
        demonstrated_level=demonstrated,
        alignment=alignment,
        signals=signals,
        notes=notes,
    )


def find_unsupported_claims(resume: ParsedResume) -> list[UnsupportedClaim]:
    """Skills listed without supporting evidence in experience/project bullets."""
    experience_text = " ".join(
        b.text for b in resume.bullets if b.section in {"experience", "projects", "summary"}
    ).lower()
    claims: list[UnsupportedClaim] = []
    for skill in resume.skills_listed:
        token = skill.lower().strip()
        if len(token) < 2:
            continue
        if token in {"communication", "teamwork", "leadership", "problem solving"}:
            continue
        # Require the skill (or a distinctive alias) to appear in experience bullets
        aliases = {token}
        if token in {"machine learning", "ml"}:
            aliases |= {"machine learning", "ml", "neural", "model training", "scikit", "pytorch", "tensorflow"}
        if token == "python":
            aliases |= {"python", "pandas", "numpy", "pytest", "fastapi", "django"}
        if token == "sql":
            aliases |= {"sql", "postgres", "mysql", "query"}
        demonstrated = any(alias in experience_text for alias in aliases if len(alias) > 1)
        if not demonstrated:
            claims.append(
                UnsupportedClaim(
                    skill=skill,
                    note="Skill listed but not demonstrated in experience or project bullets.",
                )
            )
    return claims[:15]


def check_consistency(resume: ParsedResume) -> list[ConsistencyIssue]:
    issues: list[ConsistencyIssue] = []

    # Duplicate bullets
    texts = [b.text.strip().lower() for b in resume.bullets]
    counts = Counter(texts)
    for text, count in counts.items():
        if count > 1 and len(text) > 20:
            issues.append(
                ConsistencyIssue(
                    kind="duplicate_bullets",
                    detail=f"Repeated bullet ({count}×): “{text[:90]}…”",
                    severity="warning",
                )
            )

    # Date ordering / gaps
    spans = [d for d in resume.date_spans if d.start and d.end]
    spans.sort(key=lambda d: d.start or datetime.min)
    for i in range(1, len(spans)):
        prev_end = spans[i - 1].end
        curr_start = spans[i].start
        if prev_end and curr_start and curr_start < prev_end:
            # Overlap is ok; check inverted ranges inside a span
            pass
        if prev_end and curr_start:
            gap_days = (curr_start - prev_end).days
            if gap_days > 180:
                issues.append(
                    ConsistencyIssue(
                        kind="employment_gap",
                        detail=(
                            f"Possible employment gap of ~{gap_days // 30} months "
                            f"between “{spans[i - 1].raw}” and “{spans[i].raw}”."
                        ),
                        severity="review",
                    )
                )
    for span in resume.date_spans:
        if span.start and span.end and span.start > span.end:
            issues.append(
                ConsistencyIssue(
                    kind="conflicting_dates",
                    detail=f"Date range appears inverted: “{span.raw}”.",
                    severity="warning",
                )
            )

    # Suspicious metrics without context
    for bullet in resume.bullets:
        if re.search(r"\d+\s*%", bullet.text) and not re.search(
            r"(?i)(reduc|improv|increas|decreas|save|yield|uptime|downtime|growth|faster)",
            bullet.text,
        ):
            issues.append(
                ConsistencyIssue(
                    kind="metric_without_context",
                    detail=f"Metric present without clear outcome context: “{bullet.text[:100]}”",
                    severity="review",
                )
            )

    # Generic language density
    generic = ("responsible for", "various", "several", "duties included", "worked on", "helped with")
    generic_hits = [b for b in resume.bullets if any(g in b.text.lower() for g in generic)]
    if len(generic_hits) >= 3:
        issues.append(
            ConsistencyIssue(
                kind="generic_language",
                detail=f"{len(generic_hits)} bullets use generic responsibility language.",
                severity="review",
            )
        )

    return issues[:20]


def classify_bullet_type(text: str) -> str:
    lower = text.lower()
    if re.search(r"\d+\s*%|\$\d+|reduced|improved|increased|saved", lower):
        return "measurable impact"
    if any(w in lower for w in ("led", "owned", "drove", "established", "mentored")):
        return "ownership"
    if any(w in lower for w in ("designed", "built", "implemented", "developed", "architected")):
        return "technical contribution"
    if any(w in lower for w in ("responsible for", "duties", "supported", "assisted")):
        return "responsibility"
    return "task"


_LEVELS = ("Weak", "Moderate", "Strong")


def _level_from_score(score: float) -> str:
    if score >= 0.66:
        return "Strong"
    if score >= 0.33:
        return "Moderate"
    return "Weak"


def analyze_bullets(
    resume: ParsedResume,
    evidence_graph: list[RequirementEvidence],
    limit: int = 8,
) -> list[BulletAnalysis]:
    """Analyze the most relevant resume bullets individually."""
    # Rank bullets by how often they appear as evidence
    relevance_map: dict[str, float] = {}
    for node in evidence_graph:
        for item in node.evidence:
            relevance_map[item.bullet] = max(relevance_map.get(item.bullet, 0.0), item.relevance)

    ranked = sorted(relevance_map.items(), key=lambda x: x[1], reverse=True)[:limit]
    if not ranked:
        ranked = [(b.text, 0.3) for b in resume.bullets[:limit]]

    analyses: list[BulletAnalysis] = []
    from .evidence import _depth_label, _impact_label, _ownership_label

    for text, rel in ranked:
        ownership = _ownership_label(text)
        depth = _depth_label(text)
        impact = _impact_label(text)
        specificity = "Strong" if len(re.findall(r"[A-Za-z]{4,}", text)) >= 8 and (
            any(c.isupper() for c in text[1:]) or re.search(r"\d", text)
        ) else ("Moderate" if len(text) > 60 else "Weak")
        clarity = "Strong" if 40 < len(text) < 220 else ("Moderate" if len(text) <= 280 else "Weak")
        feedback_parts = []
        if impact == "Weak":
            feedback_parts.append(
                "Does not explain performance, scale, reliability, cost, or business impact."
            )
        if ownership == "Weak":
            feedback_parts.append("Reads as support/responsibility rather than personal ownership.")
        if depth == "Weak":
            feedback_parts.append("Limited technical method (debug, DOE, design, analysis).")
        if not feedback_parts:
            feedback_parts.append("Solid evidence; consider adding scale or outcome if known.")
        analyses.append(
            BulletAnalysis(
                bullet=text,
                relevance=_level_from_score(rel),
                technical_depth=depth,
                ownership=ownership,
                impact=impact,
                clarity=clarity,
                specificity=specificity,
                feedback=" ".join(feedback_parts),
            )
        )
    return analyses


def category_from_strengths(strengths: list[EvidenceStrength], weights: list[float] | None = None) -> str:
    if not strengths:
        return "Weak"
    weights = weights or [1.0] * len(strengths)
    from .evidence import STRENGTH_SCORE

    total_w = sum(weights) or 1.0
    avg = sum(STRENGTH_SCORE[s] * w for s, w in zip(strengths, weights)) / total_w
    if avg >= 0.72:
        return "Strong"
    if avg >= 0.4:
        return "Moderate"
    return "Weak"


def overall_alignment(evidence_graph: list[RequirementEvidence], requirements: list[Requirement]) -> dict[str, str]:
    core = [e for e in evidence_graph if e.importance == Importance.CORE.value]
    important = [e for e in evidence_graph if e.importance in {Importance.CORE.value, Importance.IMPORTANT.value}]
    all_nodes = evidence_graph

    core_cov = category_from_strengths([e.strength for e in core]) if core else category_from_strengths([e.strength for e in important])
    tech_strengths = []
    for e in all_nodes:
        if any(i.technical_depth == "Strong" for i in e.evidence):
            tech_strengths.append(EvidenceStrength.STRONG)
        elif e.evidence:
            tech_strengths.append(EvidenceStrength.MODERATE)
        else:
            tech_strengths.append(EvidenceStrength.NONE)

    impact_strengths = []
    for e in all_nodes:
        if any(i.impact == "Strong" for i in e.evidence):
            impact_strengths.append(EvidenceStrength.STRONG)
        elif any(i.impact == "Moderate" for i in e.evidence):
            impact_strengths.append(EvidenceStrength.MODERATE)
        else:
            impact_strengths.append(EvidenceStrength.WEAK if e.evidence else EvidenceStrength.NONE)

    relevance = category_from_strengths([e.strength for e in important or all_nodes])
    overall = category_from_strengths(
        [e.strength for e in (core or important or all_nodes)],
        weights=[1.0] * len(core or important or all_nodes),
    )
    return {
        "overall_role_alignment": overall,
        "core_requirement_coverage": core_cov,
        "technical_depth": category_from_strengths(tech_strengths) if tech_strengths else "Weak",
        "experience_relevance": relevance,
        "demonstrated_impact": category_from_strengths(impact_strengths) if impact_strengths else "Weak",
    }
