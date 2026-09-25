"""Evidence graph: link job requirements to resume evidence and score strength.

Relevance requires domain/concept overlap — shared generic verbs like
"review", "investigate", or "perform" are NOT enough.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .concepts import CONCEPT_GRAPH, concept_label, find_concepts_in_text
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

# Verbs/adjectives that appear in almost every JD and resume — never treat as domain proof
_GENERIC_TOKENS = frozenset(
    """
    a an the and or but if then else when at by for with about against between into
    through during before after above below to from up down in out on off over under
    again further once here there all any both each few more most other some such no
    nor not only own same so than too very can will just don should now is are was
    were be been being have has had do does did of this that these those it its as we
    you your he she they them their our i me my
    review reviews reviewed approve approves approved provide provides provided
    apply applies applied able willingness willing possess possesses experience
    experiences strong good excellent work works working team teams using use used
    including include related relatedness support supports supported help helps
    helped make makes made ensure ensures ensured conduct conducts conducted
    perform performs performed manage manages managed responsible ability skills
    knowledge understanding demonstrate demonstrates demonstrated drive drives
    driven gather gathers gathering analyze analyzes analyzing analysis
    decision decisions recommendation recommendations improvement improvements
    process processes project projects multiple various several duties role roles
    candidate candidates resume resumes screen screens screening evaluate
    evaluates evaluating qualifications hire hiring recruit recruiting
    agricultural agriculture pesticide pesticides surveillance complaint complaints
    marketing social media newsletter brand campaign campaigns sales customer
    customers retail promotional event events
    """.split()
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

# Single-token concept triggers that are too ambiguous alone (need a domain partner)
_AMBIGUOUS_CONCEPT_TRIGGERS = frozenset(
    {
        "investigated",
        "investigate",
        "debugged",
        "debug",
        "debugging",
        "isolated",
        "isolate",
        "diagnosed",
        "diagnose",
        "analyzed",
        "analysis",
        "review",
        "reviewed",
        "coordinated",
        "coordinate",
        "collaborated",
        "collaborate",
        "worked with",
        "supported",
        "support",
        "improved",
        "reduced",
        "designed",
        "design",
        "tested",
        "testing",
        "test",
        "hardware",
        "system",
        "systems",
        "data",
        "customer",
        "client",
        "automation",
        "automated",
        "qualification",
        "validation",
        "verification",
        "failure",
        "failures",
        "defect",
        "defects",
    }
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


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}", (text or "").lower())
        if t not in _GENERIC_TOKENS and len(t) > 2
    }


def _domain_tokens(text: str) -> set[str]:
    """Content tokens that can justify domain relevance."""
    return _tokens(text)


def _shared_domain_terms(req_text: str, bullet_text: str, related: list[str]) -> set[str]:
    req_terms = _domain_tokens(req_text) | {
        t.lower() for t in related if t.lower() not in _GENERIC_TOKENS and len(t) > 3
    }
    # Prefer multi-word related phrases present in the bullet
    bullet_lower = bullet_text.lower()
    hits: set[str] = set()
    for phrase in related:
        p = phrase.lower().strip()
        if len(p) < 4 or p in _GENERIC_TOKENS or p in _AMBIGUOUS_CONCEPT_TRIGGERS:
            continue
        if p in bullet_lower:
            hits.add(p)
    # Token overlap on distinctive terms
    bullet_terms = _domain_tokens(bullet_text)
    for term in req_terms & bullet_terms:
        if term not in _AMBIGUOUS_CONCEPT_TRIGGERS:
            hits.add(term)
    return hits


def _concept_ids_in_text(text: str) -> set[str]:
    """Concepts evidenced in text, ignoring ambiguous single-verb-only hits."""
    lower = (text or "").lower()
    found: set[str] = set()
    for concept_id, terms in CONCEPT_GRAPH.items():
        # Strong hits: multi-word terms or distinctive tokens (>= 5 chars, not ambiguous)
        strong = [
            t
            for t in terms
            if t in lower
            and (
                (" " in t or "-" in t or "/" in t)
                or (len(t) >= 5 and t not in _AMBIGUOUS_CONCEPT_TRIGGERS)
            )
        ]
        if strong:
            found.add(concept_id)
            continue
        # Weak single-token hits only count if paired with another domain token from same concept
        weak = [t for t in terms if t in lower and t in _AMBIGUOUS_CONCEPT_TRIGGERS]
        if len(weak) >= 2:
            found.add(concept_id)
    return found


def _shared_concepts(req: Requirement, bullet_text: str) -> list[str]:
    req_concepts = _concept_ids_in_text(req.text + " " + " ".join(req.related_concepts))
    # Also map requirement text onto graph via find_concepts with filtering
    for c in find_concepts_in_text(req.text):
        # Keep only if not solely ambiguous
        if c.concept_id in _concept_ids_in_text(req.text) or c.concept_id in req_concepts:
            req_concepts.add(c.concept_id)
    bullet_concepts = _concept_ids_in_text(bullet_text)
    shared = sorted(req_concepts & bullet_concepts)
    return [concept_label(c) for c in shared]


def _tfidf_domain_similarity(a: str, b: str) -> float:
    """TF-IDF similarity on domain tokens only (generics stripped)."""
    a_toks = sorted(_domain_tokens(a))
    b_toks = sorted(_domain_tokens(b))
    if len(a_toks) < 2 or len(b_toks) < 2:
        return 0.0
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vec.fit_transform([" ".join(a_toks), " ".join(b_toks)])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        return 0.0


def relevance_score(
    req: Requirement, bullet_text: str
) -> tuple[float, list[str], set[str]]:
    """
    Return (score 0-1, shared concept labels, shared domain terms).

    A bullet must share distinctive domain terms and/or the same technical
    concept — overlapping verbs like "review"/"investigate" alone score ~0.
    """
    shared_terms = _shared_domain_terms(req.text, bullet_text, req.related_concepts)
    shared_concepts = _shared_concepts(req, bullet_text)
    sim = _tfidf_domain_similarity(
        req.text + " " + " ".join(req.related_concepts), bullet_text
    )

    if not shared_terms and not shared_concepts:
        # Pure TF-IDF on leftover tokens is still suspicious — require a floor
        return (0.0 if sim < 0.35 else round(sim * 0.4, 3), [], set())

    score = 0.0
    # Distinctive term hits are the primary signal
    score += min(0.55, 0.18 * len(shared_terms))
    # Shared technical concepts
    score += min(0.35, 0.15 * len(shared_concepts))
    # Mild TF-IDF assist only after domain proof exists
    score += 0.25 * sim

    # Multi-word technical phrase in both sides is strong
    bullet_l = bullet_text.lower()
    req_l = req.text.lower()
    for phrase in (
        "root cause",
        "signal integrity",
        "thermal plunger",
        "test socket",
        "hardware validation",
        "failure analysis",
        "design of experiments",
        "ate",
        "slt",
        "doe",
    ):
        if phrase in req_l and phrase in bullet_l:
            score += 0.2

    return min(1.0, score), shared_concepts[:6], shared_terms


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


def _classify_strength(
    best_relevance: float,
    evidence_count: int,
    ownership_hits: int,
    depth_hits: int,
    impact_hits: int,
) -> EvidenceStrength:
    # Relevance gate: ownership/depth cannot inflate unrelated bullets
    if evidence_count == 0 or best_relevance < 0.28:
        return EvidenceStrength.NONE
    if best_relevance < 0.40:
        return EvidenceStrength.WEAK

    score = best_relevance
    score += min(0.12, 0.04 * evidence_count)
    score += 0.05 * ownership_hits
    score += 0.05 * depth_hits
    score += 0.04 * impact_hits

    if score >= 0.88 and evidence_count >= 2 and best_relevance >= 0.55:
        return EvidenceStrength.VERY_STRONG
    if score >= 0.62 and best_relevance >= 0.48:
        return EvidenceStrength.STRONG
    if score >= 0.45:
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
    bullets = [
        b
        for b in (resume.bullets or [])
        if b.section in {"experience", "projects", "summary"}
        and not re.match(r"(?i)^skills?\b", b.text.strip())
    ]
    if not bullets:
        bullets = [
            b
            for b in (resume.bullets or [])
            if b.section != "skills" and len(b.text) > 40
        ]
    if not bullets and resume.raw_text.strip():
        bullets = [Bullet(text=resume.raw_text[:500], section="experience")]

    scored_reqs = [r for r in requirements if r.importance != Importance.GENERIC]
    for req in scored_reqs:
        ranked: list[tuple[float, Bullet, list[str]]] = []
        for bullet in bullets:
            score, shared_concepts, shared_terms = relevance_score(req, bullet.text)
            # Hard reject: no distinctive overlap
            if score < 0.28 or (not shared_terms and not shared_concepts):
                continue
            ranked.append((score, bullet, shared_concepts))

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
