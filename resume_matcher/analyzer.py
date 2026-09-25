"""Orchestrate evidence-based resume analysis against a job description."""

from __future__ import annotations

from .ats import assess_ats
from .evidence import build_evidence_graph
from .jd_parser import parse_job_description
from .language import detect_language
from .report import MatchReport, build_report
from .resume_parse import parse_resume
from .signals import (
    analyze_bullets,
    assess_seniority,
    check_consistency,
    find_unsupported_claims,
    overall_alignment,
)


def analyze_resume(
    job_description: str,
    resume_text: str,
    filename: str = "resume",
    threshold: float = 50.0,
) -> MatchReport:
    """
    Run the full evidence-based assessment for one resume.

    Primary question answered:
    "How convincingly does this resume demonstrate that this candidate
    can perform this specific job?"
    """
    if not (job_description or "").strip():
        return _empty_report(filename, threshold, "Job description is empty.")
    if not (resume_text or "").strip():
        return _empty_report(filename, threshold, "Could not extract text from this resume.")

    language = detect_language(job_description)
    requirements = parse_job_description(job_description)
    resume = parse_resume(resume_text)
    evidence_graph = build_evidence_graph(requirements, resume)
    overall = overall_alignment(evidence_graph, requirements)
    seniority = assess_seniority(resume, job_description)
    ats = assess_ats(resume, filename)
    unsupported = find_unsupported_claims(resume)
    consistency = check_consistency(resume)
    bullets = analyze_bullets(resume, evidence_graph)

    return build_report(
        filename=filename,
        language=language,
        requirements=requirements,
        resume=resume,
        evidence_graph=evidence_graph,
        overall=overall,
        seniority=seniority,
        ats=ats,
        unsupported=unsupported,
        consistency=consistency,
        bullet_analyses=bullets,
        threshold=threshold,
    )


def analyze_resumes(
    job_description: str,
    resumes: list[tuple[str, str]],
    threshold: float = 50.0,
) -> list[MatchReport]:
    """Analyze many resumes; sort greenlit + highest composite score first."""
    reports = [
        analyze_resume(job_description, text, filename=name, threshold=threshold)
        for name, text in resumes
    ]
    reports.sort(key=lambda r: (r.greenlit, r.composite_score), reverse=True)
    return reports


def _empty_report(filename: str, threshold: float, error: str) -> MatchReport:
    from .report import MatchReport

    return MatchReport(
        filename=filename,
        language={"code": "unknown", "name": "Unknown", "confidence": 0.0},
        overall={
            "overall_role_alignment": "Weak",
            "core_requirement_coverage": "Weak",
            "technical_depth": "Weak",
            "experience_relevance": "Weak",
            "demonstrated_impact": "Weak",
            "seniority_alignment": "Weak",
            "ats_compatibility": "Weak",
        },
        greenlit=False,
        composite_score=0.0,
        score_explanation=error,
        core_strengths=[],
        major_gaps=[error],
        requirement_coverage=[],
        evidence_graph=[],
        technical_analysis=[],
        seniority={
            "target_level": "n/a",
            "demonstrated_level": "n/a",
            "alignment": "Weak",
            "signals": [],
            "notes": error,
        },
        transferable_skills=[],
        impact_analysis=[],
        ats={"rating": "Weak", "score": 0.0, "findings": [error], "positives": [], "notes": ""},
        unsupported_claims=[],
        consistency_issues=[],
        improvements=[],
        bullet_analyses=[],
        rewrites=[],
        missing_evidence=[],
        threshold=threshold,
    )
