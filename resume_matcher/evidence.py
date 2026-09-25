"""Evidence graph: link job requirements to resume evidence using semantic embeddings."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from .concepts import concept_label, find_concepts_in_text
from .jd_parser import Importance, Requirement
from .resume_parse import Bullet, ParsedResume
from .semantic import similarity_matrix

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

# Still used to slightly boost / explain matches — embeddings do the heavy lifting
_DOMAIN_HINTS = (
    "ate",
    "slt",
    "socket",
    "thermal",
    "plunger",
    "doe",
    "jtag",
    "pcb",
    "handler",
    "advantest",
    "validation",
    "qualification",
    "debug",
    "failure",
    "semiconductor",
    "fixture",
    "load board",
    "signal integrity",
    "root cause",
    "python",
    "fastapi",
    "kubernetes",
    "aws",
)

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
    "qualified",
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
    "signal integrity",
    "thermal",
    "socket",
)


@dataclass
class EvidenceItem:
    bullet: str
    relevance: float
    ownership: str
    impact: str
    technical_depth: str
    matched_concepts: list[str] = field(default_factory=list)
    semantic_similarity: float = 0.0

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
    if hits >= 2 or (
        hits >= 1
        and has_number
        and any(w in lower for w in ("reduced", "improved", "increased", "saved"))
    ):
        return "Strong"
    if hits >= 1 or (
        has_number
        and any(w in lower for w in ("across", "testers", "systems", "users", "customers"))
    ):
        return "Moderate"
    return "Weak"


def _depth_label(text: str) -> str:
    lower = text.lower()
    hits = sum(1 for m in _DEPTH_MARKERS if m in lower)
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


def _domain_overlap(req_text: str, bullet_text: str, related: list[str]) -> list[str]:
    blob = (bullet_text or "").lower()
    req = (req_text + " " + " ".join(related)).lower()
    hits = []
    for hint in _DOMAIN_HINTS:
        if hint in req and hint in blob:
            hits.append(hint)
    for phrase in related:
        p = phrase.lower().strip()
        if len(p) >= 4 and p in blob:
            hits.append(p)
    # de-dupe preserve order
    seen: set[str] = set()
    out = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out[:8]


def _classify_strength(
    best_sim: float,
    evidence_count: int,
    ownership_hits: int,
    depth_hits: int,
    impact_hits: int,
    domain_hits: int,
) -> EvidenceStrength:
    """Map embedding similarity (+ light quality signals) to evidence strength."""
    if evidence_count == 0 or best_sim < 0.22:
        return EvidenceStrength.NONE

    # Embedding floors (calibrated on MiniLM)
    if best_sim >= 0.58 and (domain_hits or depth_hits):
        base = EvidenceStrength.VERY_STRONG
    elif best_sim >= 0.45:
        base = EvidenceStrength.STRONG
    elif best_sim >= 0.34:
        base = EvidenceStrength.MODERATE
    elif best_sim >= 0.22:
        base = EvidenceStrength.WEAK
    else:
        return EvidenceStrength.NONE

    # Quality can promote one step when semantic already solid
    quality = ownership_hits + depth_hits + impact_hits
    if base == EvidenceStrength.STRONG and quality >= 2 and evidence_count >= 2 and best_sim >= 0.50:
        return EvidenceStrength.VERY_STRONG
    if base == EvidenceStrength.MODERATE and quality >= 2 and best_sim >= 0.38:
        return EvidenceStrength.STRONG
    if base == EvidenceStrength.WEAK and domain_hits and best_sim >= 0.28:
        return EvidenceStrength.MODERATE
    return base


def _notes_for(strength: EvidenceStrength, items: list[EvidenceItem], req: Requirement) -> str:
    if strength == EvidenceStrength.NONE:
        return (
            f"No clear resume evidence found for '{req.text}'. "
            "If you have related experience, make the connection explicit."
        )
    best = max((i.semantic_similarity for i in items), default=0.0)
    if strength == EvidenceStrength.WEAK:
        return (
            f"Only weak semantic overlap (similarity {best:.2f}); "
            "clarify ownership and technical method."
        )
    if any(i.impact == "Weak" for i in items) and strength in {
        EvidenceStrength.STRONG,
        EvidenceStrength.MODERATE,
    }:
        return (
            f"Relevant experience found (similarity {best:.2f}), "
            "but measurable outcomes are thin."
        )
    if strength == EvidenceStrength.VERY_STRONG:
        return f"Strong semantic match (similarity {best:.2f}) with ownership and technical depth."
    return f"Clear supporting evidence (similarity {best:.2f})."


def _candidate_bullets(resume: ParsedResume) -> list[Bullet]:
    bullets = [
        b
        for b in (resume.bullets or [])
        if b.section in {"experience", "projects", "summary", "skills"}
        and not re.match(r"(?i)^skills?\s*$", b.text.strip())
        and len(b.text.strip()) > 20
    ]
    if not bullets:
        bullets = [
            b
            for b in (resume.bullets or [])
            if b.section != "skills" and len(b.text) > 40
        ]
    if not bullets and resume.raw_text.strip():
        # Chunk raw text as last resort
        chunks = re.split(r"\n+", resume.raw_text)
        bullets = [
            Bullet(text=c.strip(), section="experience")
            for c in chunks
            if len(c.strip()) > 40
        ][:30]
    return bullets


def build_evidence_graph(
    requirements: list[Requirement],
    resume: ParsedResume,
    max_evidence_per_req: int = 4,
) -> list[RequirementEvidence]:
    """For each important requirement, find supporting resume evidence via embeddings."""
    results: list[RequirementEvidence] = []
    bullets = _candidate_bullets(resume)
    scored_reqs = [r for r in requirements if r.importance != Importance.GENERIC]
    if not scored_reqs or not bullets:
        for req in scored_reqs:
            results.append(
                RequirementEvidence(
                    requirement=req.text,
                    importance=req.importance.value,
                    related_concepts=req.related_concepts,
                    evidence=[],
                    strength=EvidenceStrength.NONE,
                    notes=_notes_for(EvidenceStrength.NONE, [], req),
                )
            )
        return results

    # Batch embed all requirements and bullets once (fast for many resumes' bullets)
    queries = [
        req.text
        + (
            ". Related: " + ", ".join(req.related_concepts[:8])
            if req.related_concepts
            else ""
        )
        for req in scored_reqs
    ]
    docs = [b.text for b in bullets]
    sim = similarity_matrix(queries, docs)

    for req_idx, req in enumerate(scored_reqs):
        ranked: list[tuple[float, float, Bullet, list[str], list[str]]] = []
        for bullet_idx, bullet in enumerate(bullets):
            semantic = float(sim[req_idx, bullet_idx])
            domain = _domain_overlap(req.text, bullet.text, req.related_concepts)
            concepts = [
                c.label
                for c in find_concepts_in_text(bullet.text)
                if c.concept_id
                in {x.concept_id for x in find_concepts_in_text(req.text)}
                or any(t in req.text.lower() for t in c.matched_terms[:3])
            ]
            # Soft boost for domain overlap; cannot invent relevance from zero
            boost = min(0.12, 0.03 * len(domain) + 0.02 * len(concepts))
            relevance = min(1.0, semantic + boost)

            # Reject unrelated: low semantic AND no domain overlap
            if semantic < 0.22 and not domain:
                continue
            if relevance < 0.24:
                continue

            # Cap skills-list bullets
            if bullet.section == "skills":
                relevance = min(relevance, 0.42)
                semantic = min(semantic, 0.42)

            ranked.append((relevance, semantic, bullet, concepts[:5], domain))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top = ranked[:max_evidence_per_req]

        items: list[EvidenceItem] = []
        ownership_hits = depth_hits = impact_hits = 0
        domain_hit_count = 0
        for relevance, semantic, bullet, matched, domain in top:
            own = _ownership_label(bullet.text)
            impact = _impact_label(bullet.text)
            depth = _depth_label(bullet.text)
            domain_hit_count += len(domain)
            if bullet.section != "skills":
                if own == "Strong":
                    ownership_hits += 1
                if depth == "Strong":
                    depth_hits += 1
                if impact == "Strong":
                    impact_hits += 1
            items.append(
                EvidenceItem(
                    bullet=bullet.text,
                    relevance=round(relevance, 3),
                    ownership=own if bullet.section != "skills" else "Listed",
                    impact=impact if bullet.section != "skills" else "Listed",
                    technical_depth=depth if bullet.section != "skills" else "Listed",
                    matched_concepts=matched or [concept_label(d) for d in domain[:3]],
                    semantic_similarity=round(semantic, 3),
                )
            )

        best_sim = max((i.semantic_similarity for i in items), default=0.0)
        exp_count = sum(1 for _, _, b, _, _ in top if b.section != "skills")
        strength = _classify_strength(
            best_sim,
            exp_count if exp_count else len(items),
            ownership_hits,
            depth_hits,
            impact_hits,
            domain_hit_count,
        )
        if items and exp_count == 0 and strength in {
            EvidenceStrength.STRONG,
            EvidenceStrength.VERY_STRONG,
        }:
            strength = EvidenceStrength.MODERATE

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
