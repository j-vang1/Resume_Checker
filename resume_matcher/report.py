"""Assemble the final evidence-based resume match report."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .ats import ATSAssessment
from .evidence import EvidenceStrength, RequirementEvidence, STRENGTH_SCORE
from .jd_parser import Importance, Requirement
from .resume_parse import ParsedResume
from .signals import (
    BulletAnalysis,
    ConsistencyIssue,
    SeniorityAssessment,
    UnsupportedClaim,
)


@dataclass
class RewriteSuggestion:
    original: str
    improved_structure: str
    needs_from_candidate: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchReport:
    filename: str
    language: dict
    overall: dict[str, str]
    greenlit: bool
    composite_score: float
    score_explanation: str
    core_strengths: list[str]
    major_gaps: list[str]
    strength_table: list[dict]
    gap_table: list[dict]
    requirement_coverage: list[dict]
    evidence_graph: list[dict]
    technical_analysis: list[dict]
    seniority: dict
    transferable_skills: list[str]
    impact_analysis: list[str]
    ats: dict
    unsupported_claims: list[dict]
    consistency_issues: list[dict]
    improvements: list[str]
    bullet_analyses: list[dict]
    rewrites: list[dict]
    missing_evidence: list[str]
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# RESUME MATCH REPORT",
            f"**File:** {self.filename}",
            "",
            "## Overall Role Alignment",
            f"- **Overall:** {self.overall.get('overall_role_alignment', 'n/a')}",
            f"- **Core Requirement Coverage:** {self.overall.get('core_requirement_coverage', 'n/a')}",
            f"- **Technical Depth:** {self.overall.get('technical_depth', 'n/a')}",
            f"- **Experience Relevance:** {self.overall.get('experience_relevance', 'n/a')}",
            f"- **Demonstrated Impact:** {self.overall.get('demonstrated_impact', 'n/a')}",
            f"- **Seniority Alignment:** {self.seniority.get('alignment', 'n/a')}",
            f"- **ATS Compatibility:** {self.ats.get('rating', 'n/a')}",
            "",
            f"**Composite score (informational):** {self.composite_score:.0f}/100 — {self.score_explanation}",
            f"**Greenlight (≥ {self.threshold:.0f}):** {'Yes' if self.greenlit else 'No'}",
            "",
            "## Core Strengths",
        ]
        lines.extend(f"- {s}" for s in self.core_strengths or ["None identified"])
        lines += ["", "## Major Gaps"]
        lines.extend(f"- {g}" for g in self.major_gaps or ["None identified"])
        lines += ["", "## Requirement Coverage Table", ""]
        lines.append("| Requirement | Importance | Evidence | Strength | Notes |")
        lines.append("|---|---|---|---|---|")
        for row in self.requirement_coverage:
            ev = row.get("evidence_summary", "").replace("|", "/")
            notes = row.get("notes", "").replace("|", "/")
            lines.append(
                f"| {row['requirement'][:60]} | {row['importance']} | {ev[:50]} | "
                f"{row['strength']} | {notes[:60]} |"
            )
        lines += ["", "## Technical Experience Analysis"]
        for b in self.bullet_analyses[:5]:
            lines += [
                f"- **Bullet:** {b['bullet']}",
                f"  - Relevance: {b['relevance']} · Depth: {b['technical_depth']} · "
                f"Ownership: {b['ownership']} · Impact: {b['impact']}",
                f"  - Feedback: {b['feedback']}",
            ]
        lines += ["", "## Seniority Alignment", self.seniority.get("notes", "")]
        lines += ["", "## Transferable Skills"]
        lines.extend(f"- {s}" for s in self.transferable_skills or ["None highlighted"])
        lines += ["", "## Impact / Accomplishment Analysis"]
        lines.extend(f"- {s}" for s in self.impact_analysis or ["Limited measurable impact language"])
        lines += ["", "## ATS Compatibility", f"Rating: **{self.ats.get('rating')}**"]
        for f in self.ats.get("findings", []):
            lines.append(f"- {f}")
        lines += ["", "## Weak or Unsupported Claims"]
        if self.unsupported_claims:
            for c in self.unsupported_claims:
                lines.append(f"- {c['skill']}: {c['note']}")
        else:
            lines.append("- None flagged")
        lines += ["", "## Consistency Review Items"]
        if self.consistency_issues:
            for i in self.consistency_issues:
                lines.append(f"- ({i['severity']}) {i['detail']}")
        else:
            lines.append("- None flagged")
        lines += ["", "## Top Resume Improvements"]
        for i, item in enumerate(self.improvements, 1):
            lines.append(f"{i}. {item}")
        lines += ["", "## Missing-Evidence Recommendations"]
        lines.extend(f"- {m}" for m in self.missing_evidence or ["None"])
        lines += ["", "## Bullet Rewrite Opportunities"]
        for r in self.rewrites:
            lines += [
                f"- **Original:** {r['original']}",
                f"  - **Improved structure:** {r['improved_structure']}",
            ]
            if r.get("needs_from_candidate"):
                lines.append(
                    "  - **Needs from candidate:** " + "; ".join(r["needs_from_candidate"])
                )
        return "\n".join(lines)


def _coverage_rows(graph: list[RequirementEvidence]) -> list[dict]:
    rows = []
    for node in graph:
        if node.importance == Importance.GENERIC.value:
            continue
        if node.evidence:
            summary = "; ".join(f'“{e.bullet[:70]}”' for e in node.evidence[:2])
        else:
            summary = "—"
        rows.append(
            {
                "requirement": node.requirement,
                "importance": node.importance,
                "evidence_summary": summary,
                "strength": node.strength.value,
                "notes": node.notes,
            }
        )
    return rows


def _short(text: str, limit: int = 90) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _core_strength_rows(graph: list[RequirementEvidence]) -> list[dict]:
    rows = []
    for node in graph:
        if node.strength not in {EvidenceStrength.STRONG, EvidenceStrength.VERY_STRONG}:
            continue
        sample = node.evidence[0].bullet if node.evidence else ""
        rows.append(
            {
                "Requirement": _short(node.requirement, 70),
                "Importance": node.importance,
                "Strength": node.strength.value.replace(" Evidence", ""),
                "Evidence": _short(sample, 100) if sample else "—",
            }
        )
    return rows[:12]


def _major_gap_rows(graph: list[RequirementEvidence]) -> list[dict]:
    rows = []
    for node in graph:
        if node.importance not in {Importance.CORE.value, Importance.IMPORTANT.value}:
            continue
        if node.strength not in {EvidenceStrength.NONE, EvidenceStrength.WEAK}:
            continue
        rows.append(
            {
                "Requirement": _short(node.requirement, 70),
                "Importance": node.importance,
                "Strength": node.strength.value.replace(" Evidence", ""),
                "Gap": _short(node.notes, 110),
            }
        )
    return rows[:12]


def _core_strengths(graph: list[RequirementEvidence]) -> list[str]:
    return [
        f"{r['Requirement']}: {r['Strength']}"
        + (f" — e.g. “{r['Evidence']}”" if r["Evidence"] != "—" else "")
        for r in _core_strength_rows(graph)
    ]


def _major_gaps(graph: list[RequirementEvidence]) -> list[str]:
    return [
        f"{r['Requirement']} ({r['Importance']}): {r['Gap']}"
        for r in _major_gap_rows(graph)
    ]


def _transferable(graph: list[RequirementEvidence]) -> list[str]:
    skills: list[str] = []
    seen: set[str] = set()
    for node in graph:
        for concept in node.related_concepts:
            if concept.lower() in seen:
                continue
            # Only if some evidence touched it
            if any(concept.lower() in e.bullet.lower() for e in node.evidence):
                seen.add(concept.lower())
                skills.append(
                    f"{concept} — demonstrated in support of “{node.requirement[:50]}”"
                )
        for e in node.evidence:
            for c in e.matched_concepts:
                if c.lower() not in seen:
                    seen.add(c.lower())
                    skills.append(f"{c} — evidenced in experience bullets")
    return skills[:12]


def _impact_analysis(graph: list[RequirementEvidence], resume: ParsedResume) -> list[str]:
    notes: list[str] = []
    strong = [
        e.bullet
        for node in graph
        for e in node.evidence
        if e.impact == "Strong"
    ]
    weak_metric = [
        b.text
        for b in resume.bullets
        if re.search(r"\d", b.text)
        and not re.search(r"(?i)(reduc|improv|increas|decreas|save|yield|uptime)", b.text)
    ]
    if strong:
        notes.append(f"{len(strong)} bullet(s) show scale plus outcome language.")
        notes.append(f'Example: “{strong[0][:120]}”')
    else:
        notes.append("Few bullets combine scale with a clear outcome (e.g. reduced downtime X%).")
    if weak_metric:
        notes.append(
            f"{len(weak_metric)} bullet(s) include numbers without clear impact context "
            "(scale alone is weaker than scale + result)."
        )
    return notes


def _missing_evidence(graph: list[RequirementEvidence]) -> list[str]:
    tips: list[str] = []
    for node in graph:
        if node.importance not in {Importance.CORE.value, Importance.IMPORTANT.value}:
            continue
        if node.strength in {EvidenceStrength.NONE, EvidenceStrength.WEAK, EvidenceStrength.MODERATE}:
            related = ", ".join(node.related_concepts[:5]) or "related work"
            if node.evidence:
                tips.append(
                    f"The role emphasizes “{node.requirement}”. Your resume shows related evidence "
                    f"({related}) that may support this, but it does not clearly explain ownership "
                    f"of the end-to-end requirement. If accurate, make that connection more explicit. "
                    "Do not fabricate experience."
                )
            else:
                tips.append(
                    f"No clear evidence for “{node.requirement}”. If you have done related work "
                    f"({related}), describe the problem → method → result. "
                    "Do not invent experience."
                )
    return tips[:6]


def _improvements(
    graph: list[RequirementEvidence],
    claims: list[UnsupportedClaim],
    bullet_analyses: list[BulletAnalysis],
    seniority: SeniorityAssessment,
) -> list[str]:
    items: list[str] = []
    for node in graph:
        if node.importance == Importance.CORE.value and node.strength in {
            EvidenceStrength.NONE,
            EvidenceStrength.WEAK,
            EvidenceStrength.MODERATE,
        }:
            items.append(f"Make “{node.requirement}” ownership more explicit with method and outcome.")
            break
    weak_impact = [b for b in bullet_analyses if b.impact == "Weak" and b.relevance != "Weak"]
    if weak_impact:
        items.append("Add measurable outcomes to strong technical bullets (only if true; use [metric] placeholders).")
    if claims:
        items.append(
            f"Demonstrate listed skill “{claims[0].skill}” via a real project/accomplishment, or remove it."
        )
    generic = [b for b in bullet_analyses if b.ownership == "Weak"]
    if generic:
        items.append("Replace generic support language with specific technical ownership.")
    if seniority.alignment != "Strong":
        items.append(
            f"Clarify scope of ownership to better match {seniority.target_level} expectations "
            "(lead, define, cross-functional decisions) — only where accurate."
        )
    # Deduplicate / cap
    deduped: list[str] = []
    for i in items:
        if i not in deduped:
            deduped.append(i)
    if len(deduped) < 3:
        deduped.append("Clarify scale of cross-functional or vendor responsibility where applicable.")
    return deduped[:5]


def _rewrites(bullet_analyses: list[BulletAnalysis]) -> list[RewriteSuggestion]:
    suggestions: list[RewriteSuggestion] = []
    for b in bullet_analyses:
        if b.impact != "Weak" and b.ownership != "Weak":
            continue
        original = b.bullet
        # Preserve content; suggest structure with placeholders — never invent
        improved = (
            "Describe the problem, the engineering method you used, and the result. "
            f"Starting from your experience (“{original[:110]}”), consider: "
            "Diagnosed/designed/implemented [specific system or failure mode] using "
            "[method: DOE / debug / design / analysis], improving "
            "[reliability / uptime / cycle time / defect rate — only if known] "
            "across [scale: N systems/users — only if known]."
        )
        needs = []
        if b.impact == "Weak":
            needs.append("Optional metric or qualitative outcome (do not invent)")
        if b.ownership == "Weak":
            needs.append("Your actual ownership vs. team support role")
        needs.append("Any tools/methods you truly used")
        suggestions.append(
            RewriteSuggestion(
                original=original,
                improved_structure=improved,
                needs_from_candidate=needs,
            )
        )
        if len(suggestions) >= 3:
            break
    return suggestions


def composite_score(
    graph: list[RequirementEvidence],
    seniority: SeniorityAssessment,
    overall: dict[str, str],
) -> tuple[float, str]:
    """Informational weighted score — not a hiring probability."""
    if not graph:
        return 0.0, "No requirements could be parsed from the job description."

    importance_weight = {
        Importance.CORE.value: 1.0,
        Importance.IMPORTANT.value: 0.75,
        Importance.PREFERRED.value: 0.4,
        Importance.SUPPORTING.value: 0.2,
        Importance.GENERIC.value: 0.0,
    }
    num = den = 0.0
    for node in graph:
        w = importance_weight.get(node.importance, 0.3)
        num += STRENGTH_SCORE[node.strength] * w
        den += w
    base = 100.0 * (num / den if den else 0.0)

    # Small adjustments — never dominate
    adj = 0.0
    if overall.get("demonstrated_impact") == "Strong":
        adj += 4
    elif overall.get("demonstrated_impact") == "Weak":
        adj -= 4
    if seniority.alignment == "Strong":
        adj += 3
    elif seniority.alignment == "Weak":
        adj -= 5

    score = max(0.0, min(100.0, base + adj))
    explanation = (
        "Weighted by requirement importance and evidence strength "
        "(core requirements count most). "
        "This is not an estimated chance of being hired — it only reflects "
        "how convincingly the resume demonstrates ability to perform this job."
    )
    return round(score, 1), explanation


def build_report(
    *,
    filename: str,
    language: dict,
    requirements: list[Requirement],
    resume: ParsedResume,
    evidence_graph: list[RequirementEvidence],
    overall: dict[str, str],
    seniority: SeniorityAssessment,
    ats: ATSAssessment,
    unsupported: list[UnsupportedClaim],
    consistency: list[ConsistencyIssue],
    bullet_analyses: list[BulletAnalysis],
    threshold: float,
) -> MatchReport:
    score, explanation = composite_score(evidence_graph, seniority, overall)
    overall = {
        **overall,
        "seniority_alignment": seniority.alignment,
        "ats_compatibility": ats.rating,
    }
    return MatchReport(
        filename=filename,
        language=language,
        overall=overall,
        greenlit=score >= threshold,
        composite_score=score,
        score_explanation=explanation,
        core_strengths=_core_strengths(evidence_graph),
        major_gaps=_major_gaps(evidence_graph),
        strength_table=_core_strength_rows(evidence_graph),
        gap_table=_major_gap_rows(evidence_graph),
        requirement_coverage=_coverage_rows(evidence_graph),
        evidence_graph=[e.to_dict() for e in evidence_graph],
        technical_analysis=[b.to_dict() for b in bullet_analyses],
        seniority=seniority.to_dict(),
        transferable_skills=_transferable(evidence_graph),
        impact_analysis=_impact_analysis(evidence_graph, resume),
        ats=ats.to_dict(),
        unsupported_claims=[c.to_dict() for c in unsupported],
        consistency_issues=[c.to_dict() for c in consistency],
        improvements=_improvements(evidence_graph, unsupported, bullet_analyses, seniority),
        bullet_analyses=[b.to_dict() for b in bullet_analyses],
        rewrites=[r.to_dict() for r in _rewrites(bullet_analyses)],
        missing_evidence=_missing_evidence(evidence_graph),
        threshold=threshold,
    )
