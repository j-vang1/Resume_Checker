"""Evidence graph: link job requirements to resume evidence and score strength."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .concepts import expand_query, find_concepts_in_text
from .jd_parser import Importance, Requirement
from .resume_parse import Bullet, ParsedResume


class EvidenceStrength(str, Enum):
    NONE = "No Evidence"
    WEAK = "Weak Evidence"
    MODERATE = "Moderate Evidence"
    STRONG = "Strong Evidence"
    VERY_STRONG = "Very Strong Evidence"


STRENGTH_SCORE = {
    EvidenceStrength.NONE: 0.0,
    EvidenceStrength.WEAK: 0.25,
    EvidenceStrength.MODERATE: 0.55,
    EvidenceStrength.STRONG: 0.8,
    EvidenceStrength.VERY_STRONG: 1.0,
}

_OWNERSHIP_STRONG = (
    "led",
    "owned",
    "designed",
    "built",
    "developed",
    "architected",
    "drove",
    "established",
    "created",
    "implemented",
    "diagnosed",
    "resolved",
    "delivered",
    "launched",
    "defined",
)
_OWNERSHIP_WEAK = (
    "assisted",
    "supported",
    "helped",
    "participated",
    "exposed to",
    "familiar with",
    "responsible for supporting",
    "involved in",
)
_IMPACT_MARKERS = (
    r"\d+\s*%",
    r"\$[\d,]+",
    r"\d+x\b",
    r"reduced",
    r"improved",
    r"increased",
    r"decreased",
    r"saved",
    r"cut\b",
    r"yield",
    r"uptime",
    r"downtime",
    r"throughput",
    r"defect",
)
_DEPTH_MARKERS = (
    "debug",
    "doe",
    "experiment",
    "root cause",
    "investigat",
    "model",
    "characteriz",
    "architect",
    "tradeoff",
    "trade-off",
    "optimize",
    "analy",
    "troubleshoot",
    "isolat",
    "validat",
    "qualif",
    "implement",
)


@dataclass
class EvidenceItem:
    bullet: str
    relevance: float
    ownership: str
    impact: str
    technical_depth: str
    matched_concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RequirementEvidence:
    requirement: str
    importance: str
    related_concepts: list[str]
    evidence: list[EvidenceItem]
    strength: EvidenceStrength
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "requirement": self.requirement,
            "importance": self.importance,
            "related_concepts": self.related_concepts,
            "evidence": [e.to_dict() for e in self.evidence],
            "strength": self.strength.value,
            "notes": self.notes,
            "strength_score": STRENGTH_SCORE[self.strength],
        }


def _semantic_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    # Expand both sides with concept terms so related activities align
    a_exp = a + " " + " ".join(sorted(expand_query(a))[:40])
    b_exp = b + " " + " ".join(sorted(expand_query(b))[:40])
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b",
        )
        matrix = vec.fit_transform([a_exp, b_exp])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        return 0.0


def _ownership_label(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in _OWNERSHIP_STRONG):
        return "Strong"
    if any(w in lower for w in _OWNERSHIP_WEAK):
        return "Weak"
    return "Moderate"


def _impact_label(text: str) -> str:
    lower = text.lower()
    hits = sum(1 for p in _IMPACT_MARKERS if re.search(p, lower))
    has_number = bool(re.search(r"\d", text))
    if hits >= 2 or (hits >= 1 and has_number and any(w in lower for w in ("reduced", "improved", "increased", "saved"))):
        return "Strong"
    if hits >= 1 or (has_number and any(w in lower for w in ("across", "testers", "systems", "users", "customers"))):
        return "Moderate"
    return "Weak"


def _depth_label(text: str) -> str:
    lower = text.lower()
    hits = sum(1 for m in _DEPTH_MARKERS if m in lower)
    # Chain signals: observed → investigated → performed
    chain = sum(
        1
        for w in ("observed", "investigat", "performed", "established", "resulting", "leading to")
        if w in lower
    )
    if hits >= 3 or (hits >= 2 and chain >= 1):
        return "Strong"
    if hits >= 1:
        return "Moderate"
    return "Weak"


def _classify_strength(
    best_sim: float,
    evidence_count: int,
    ownership_hits: int,
    depth_hits: int,
    impact_hits: int,
) -> EvidenceStrength:
    if evidence_count == 0 or best_sim < 0.08:
        return EvidenceStrength.NONE
    score = best_sim
    score += min(0.2, 0.06 * evidence_count)
    score += 0.08 * ownership_hits
    score += 0.08 * depth_hits
    score += 0.05 * impact_hits
    if score >= 0.85 and evidence_count >= 2 and depth_hits:
        return EvidenceStrength.VERY_STRONG
    if score >= 0.55:
        return EvidenceStrength.STRONG
    if score >= 0.32:
        return EvidenceStrength.MODERATE
    return EvidenceStrength.WEAK


def _notes_for(strength: EvidenceStrength, items: list[EvidenceItem], req: Requirement) -> str:
    if strength == EvidenceStrength.NONE:
        return (
            f"No clear resume evidence found for '{req.text}'. "
            "If you have related experience, make the connection explicit."
        )
    if strength == EvidenceStrength.WEAK:
        return "Limited or indirect evidence; clarify ownership and technical method."
    if any(i.impact == "Weak" for i in items) and strength in {
        EvidenceStrength.STRONG,
        EvidenceStrength.MODERATE,
    }:
        return "Technical relevance is present, but measurable outcomes are thin."
    if strength == EvidenceStrength.VERY_STRONG:
        return "Multiple concrete bullets with ownership and technical depth."
    return "Clear supporting evidence present in experience bullets."


def build_evidence_graph(
    requirements: list[Requirement],
    resume: ParsedResume,
    max_evidence_per_req: int = 4,
) -> list[RequirementEvidence]:
    """For each important requirement, find supporting resume evidence."""
    results: list[RequirementEvidence] = []
    bullets = resume.bullets or [Bullet(text=resume.raw_text[:500], section="experience")]

    scored_reqs = [r for r in requirements if r.importance != Importance.GENERIC]
    for req in scored_reqs:
        query = req.text + " " + " ".join(req.related_concepts)
        ranked: list[tuple[float, Bullet, list[str]]] = []
        for bullet in bullets:
            sim = _semantic_similarity(query, bullet.text)
            concepts = find_concepts_in_text(bullet.text)
            concept_boost = 0.0
            matched_labels: list[str] = []
            req_lower = (req.text + " " + " ".join(req.related_concepts)).lower()
            for c in concepts:
                if c.label.lower() in req_lower or any(t in req_lower for t in c.matched_terms):
                    concept_boost += 0.12
                    matched_labels.append(c.label)
                elif any(t in bullet.text.lower() for t in req.related_concepts if len(t) > 3):
                    concept_boost += 0.05
            # Direct substring / related-term hits
            for term in [req.text.lower(), *req.related_concepts]:
                if len(term) > 3 and term.lower() in bullet.text.lower():
                    concept_boost += 0.1
                    break
            total = min(1.0, sim + concept_boost)
            if total >= 0.12:
                ranked.append((total, bullet, matched_labels[:5]))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top = ranked[:max_evidence_per_req]

        items: list[EvidenceItem] = []
        ownership_hits = depth_hits = impact_hits = 0
        for sim, bullet, matched in top:
            own = _ownership_label(bullet.text)
            impact = _impact_label(bullet.text)
            depth = _depth_label(bullet.text)
            if own == "Strong":
                ownership_hits += 1
            if depth == "Strong":
                depth_hits += 1
            if impact == "Strong":
                impact_hits += 1
            items.append(
                EvidenceItem(
                    bullet=bullet.text,
                    relevance=round(sim, 3),
                    ownership=own,
                    impact=impact,
                    technical_depth=depth,
                    matched_concepts=matched,
                )
            )

        best_sim = top[0][0] if top else 0.0
        strength = _classify_strength(
            best_sim, len(items), ownership_hits, depth_hits, impact_hits
        )
        results.append(
            RequirementEvidence(
                requirement=req.text,
                importance=req.importance.value,
                related_concepts=req.related_concepts,
                evidence=items,
                strength=strength,
                notes=_notes_for(strength, items, req),
            )
        )

    return results
