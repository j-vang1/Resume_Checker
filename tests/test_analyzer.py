"""Tests for evidence-based resume analysis."""

from __future__ import annotations

from resume_matcher.analyzer import analyze_resume
from resume_matcher.concepts import find_concepts_in_text
from resume_matcher.evidence import EvidenceStrength, build_evidence_graph
from resume_matcher.jd_parser import Importance, parse_job_description
from resume_matcher.language import detect_language
from resume_matcher.resume_parse import parse_resume
from resume_matcher.signals import find_unsupported_claims

JOB = """
Senior Hardware Validation Engineer

Responsibilities:
- Own hardware validation and qualification of semiconductor test interfaces
- Perform root cause analysis on intermittent hardware failures
- Lead design of experiments (DOE) for thermal and mechanical stability
- Coordinate corrective actions with suppliers and cross-functional engineering teams
- Develop Python automation for tester log analysis

Preferred:
- Statistical analysis experience
- Customer support for field issues

We are an equal opportunity employer with a competitive benefits package.
"""

RESUME_STRONG = """
Alex Rivera
alex@example.com | (555) 010-2000 | linkedin.com/in/arivera

Summary
Hardware engineer specializing in semiconductor ATE validation and failure analysis.

Skills
Python, SQL, Machine Learning, DOE, SolidWorks, JMP

Experience
Hardware Engineer, TestCo — Jan 2019 - Present
- Investigated intermittent ECID failures and isolated socket contact instability across production ATE systems.
- Performed DOE on plunger pressure and thermal stability to establish stable operating limits at -40°C.
- Qualified new socket interface hardware and wrote acceptance test criteria for production release.
- Diagnosed socket, plunger, and thermal interface failures and coordinated corrective actions with suppliers and engineering teams.
- Built Python automation that processed tester logs and reduced manual analysis time by 60%.
- Designed thermal interface hardware for semiconductor test systems used across 24 testers.

Education
B.S. Mechanical Engineering
"""

RESUME_WEAK = """
Jordan Lee
jordan@example.com

Skills
Python, Leadership, Communication

Experience
Marketing Coordinator, AdsCo — 2021 - 2023
- Responsible for supporting social media campaigns.
- Helped with newsletters and brand partnerships.
- Worked on various promotional events.
"""


def test_language_detection():
    assert detect_language(JOB)["code"] == "en"


def test_jd_parser_weights_core_and_skips_generic():
    reqs = parse_job_description(JOB)
    assert len(reqs) >= 4
    assert any(r.importance == Importance.CORE for r in reqs)
    joined = " ".join(r.text.lower() for r in reqs)
    assert "equal opportunity" not in joined
    assert "benefits package" not in joined
    # Related concepts expanded for validation / RCA
    validation = next(r for r in reqs if "validation" in r.text.lower() or "qualification" in r.text.lower())
    assert validation.related_concepts


def test_semantic_credit_without_exact_phrase():
    """RCA credited from 'investigated intermittent...isolated' without saying 'root cause analysis'."""
    resume = parse_resume(RESUME_STRONG)
    reqs = parse_job_description(JOB)
    graph = build_evidence_graph(reqs, resume)
    rca = next(
        (n for n in graph if "root cause" in n.requirement.lower() or "intermittent" in n.requirement.lower()),
        None,
    )
    # If JD line is about RCA specifically
    if rca is None:
        rca = next(n for n in graph if "root" in n.requirement.lower() or "failure" in n.requirement.lower())
    assert rca.strength in {
        EvidenceStrength.MODERATE,
        EvidenceStrength.STRONG,
        EvidenceStrength.VERY_STRONG,
    }
    assert rca.evidence
    assert "root cause analysis" not in rca.evidence[0].bullet.lower()


def test_hardware_validation_gets_strong_evidence():
    report = analyze_resume(JOB, RESUME_STRONG, filename="strong.pdf", threshold=50)
    assert report.composite_score >= 50
    assert report.greenlit is True
    assert report.overall["overall_role_alignment"] in {"Strong", "Moderate"}
    assert report.requirement_coverage
    # Evidence graph is traceable
    assert any(node["evidence"] for node in report.evidence_graph)


def test_weak_resume_not_greenlit():
    report = analyze_resume(JOB, RESUME_WEAK, filename="weak.pdf", threshold=50)
    assert report.composite_score < 50
    assert report.greenlit is False
    assert report.overall["overall_role_alignment"] == "Weak"


def test_unsupported_skill_flagged():
    resume = parse_resume(RESUME_STRONG)
    claims = find_unsupported_claims(resume)
    skills = {c.skill.lower() for c in claims}
    assert "machine learning" in skills


def test_concepts_detect_debug_chain():
    matches = find_concepts_in_text(
        "Investigated intermittent ECID failures and isolated socket contact instability."
    )
    ids = {m.concept_id for m in matches}
    assert "root_cause_analysis" in ids or "failure_analysis" in ids or "hardware_debug" in ids


def test_report_markdown_sections():
    report = analyze_resume(JOB, RESUME_STRONG, filename="strong.pdf")
    md = report.to_markdown()
    for heading in (
        "RESUME MATCH REPORT",
        "Overall Role Alignment",
        "Requirement Coverage Table",
        "ATS Compatibility",
        "Top Resume Improvements",
    ):
        assert heading in md
    assert "chance of getting hired" not in md.lower()


def test_ats_separate_from_fit():
    report = analyze_resume(JOB, RESUME_STRONG, filename="strong.pdf")
    assert "ats_compatibility" in report.overall or report.ats["rating"]
    # Strong technical fit should not be blocked solely by ATS in scoring explanation
    assert "not an estimated chance" in report.score_explanation.lower() or "not" in report.score_explanation.lower()


def test_threshold_tunable():
    report = analyze_resume(JOB, RESUME_STRONG, filename="strong.pdf", threshold=99)
    assert report.greenlit == (report.composite_score >= 99)


def test_rewrites_do_not_invent_metrics():
    report = analyze_resume(JOB, RESUME_STRONG, filename="strong.pdf")
    for r in report.rewrites:
        assert "only if known" in r["improved_structure"].lower() or r["needs_from_candidate"]
