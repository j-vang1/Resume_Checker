"""
Resume Checker — evidence-based job-fit analysis (not keyword counting).

Run with:
  streamlit run app.py
  # or
  python app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from resume_matcher.analyzer import analyze_resumes
from resume_matcher.jd_parser import parse_job_description
from resume_matcher.language import detect_language
from resume_matcher.parsers import (
    EmptyExtractError,
    UnsupportedFileTypeError,
    extract_text_from_bytes,
)
from resume_matcher.report import MatchReport

_STYLES = """
<style>
  .block-container { padding-top: 1.25rem; max-width: 1280px; }
  .lang-badge {
    display: inline-block; background: #e0f2fe; color: #0369a1;
    border-radius: 999px; padding: 0.25rem 0.75rem; font-size: 0.875rem; font-weight: 600;
  }
  .pill {
    display: inline-block; border-radius: 8px; padding: 0.2rem 0.55rem;
    font-size: 0.8rem; font-weight: 600; margin-right: 0.35rem;
  }
  .pill.strong { background: #d1fae5; color: #065f46; }
  .pill.moderate { background: #fef3c7; color: #92400e; }
  .pill.weak { background: #fee2e2; color: #9f1239; }
  .match-card {
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;
    border: 1px solid #e2e8f0;
  }
  .match-card.greenlit {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-color: #6ee7b7;
  }
  .match-card.rejected {
    background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); border-color: #fdba74;
  }
  .score-big { font-size: 1.85rem; font-weight: 700; line-height: 1; }
  .score-big.green { color: #047857; }
  .score-big.amber { color: #c2410c; }
</style>
"""


def _pill(label: str, value: str) -> str:
    cls = value.lower().split()[0] if value else "weak"
    if cls not in {"strong", "moderate", "weak"}:
        cls = "moderate"
    return f'<span class="pill {cls}">{label}: {value}</span>'


def _short(text: str, limit: int = 90) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _strength_label(raw: str) -> str:
    return (
        (raw or "")
        .replace(" Evidence", "")
        .replace("No Evidence", "None")
    )


def _show_table(rows: list[dict], empty_message: str = "None identified") -> None:
    if not rows:
        st.caption(empty_message)
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _overview_alignment_table(report: MatchReport) -> list[dict]:
    return [
        {"Dimension": "Overall role alignment", "Rating": report.overall.get("overall_role_alignment", "—")},
        {"Dimension": "Core requirement coverage", "Rating": report.overall.get("core_requirement_coverage", "—")},
        {"Dimension": "Technical depth", "Rating": report.overall.get("technical_depth", "—")},
        {"Dimension": "Experience relevance", "Rating": report.overall.get("experience_relevance", "—")},
        {"Dimension": "Demonstrated impact", "Rating": report.overall.get("demonstrated_impact", "—")},
        {
            "Dimension": "Seniority alignment",
            "Rating": report.overall.get(
                "seniority_alignment", report.seniority.get("alignment", "—")
            ),
        },
        {
            "Dimension": "ATS compatibility",
            "Rating": report.overall.get("ats_compatibility", report.ats.get("rating", "—")),
        },
    ]


def _coverage_table(report: MatchReport) -> list[dict]:
    return [
        {
            "Requirement": _short(r["requirement"], 65),
            "Importance": r["importance"],
            "Strength": _strength_label(r["strength"]),
            "Evidence": _short(r.get("evidence_summary", "—"), 85),
            "Notes": _short(r.get("notes", ""), 70),
        }
        for r in report.requirement_coverage
    ]


def _evidence_flat_table(report: MatchReport) -> list[dict]:
    rows: list[dict] = []
    for node in report.evidence_graph:
        if node.get("importance") == "Generic":
            continue
        if not node.get("evidence"):
            rows.append(
                {
                    "Requirement": _short(node["requirement"], 55),
                    "Importance": node["importance"],
                    "Strength": _strength_label(node["strength"]),
                    "Resume evidence": "—",
                    "Ownership": "—",
                    "Depth": "—",
                    "Impact": "—",
                }
            )
            continue
        for ev in node["evidence"][:3]:
            rows.append(
                {
                    "Requirement": _short(node["requirement"], 55),
                    "Importance": node["importance"],
                    "Strength": _strength_label(node["strength"]),
                    "Resume evidence": _short(ev["bullet"], 95),
                    "Ownership": ev.get("ownership", "—"),
                    "Depth": ev.get("technical_depth", "—"),
                    "Impact": ev.get("impact", "—"),
                }
            )
    return rows


def _bullet_table(report: MatchReport) -> list[dict]:
    return [
        {
            "Bullet": _short(b["bullet"], 90),
            "Relevance": b["relevance"],
            "Depth": b["technical_depth"],
            "Ownership": b["ownership"],
            "Impact": b["impact"],
            "Feedback": _short(b["feedback"], 80),
        }
        for b in report.bullet_analyses
    ]


def _render_report(report: MatchReport) -> None:
    green = report.greenlit
    card = "greenlit" if green else "rejected"
    score_cls = "green" if green else "amber"
    label = "GREENLIT" if green else "BELOW THRESHOLD"

    st.markdown(
        f"""
        <div class="match-card {card}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
            <div>
              <div style="font-weight:700;font-size:1.1rem;">{report.filename}</div>
              <div style="font-size:0.8rem;opacity:0.75;margin-top:0.25rem;">
                {label} · threshold {report.threshold:.0f} · alignment {report.overall.get('overall_role_alignment')}
              </div>
            </div>
            <div class="score-big {score_cls}">{report.composite_score:.0f}</div>
          </div>
          <div style="margin-top:0.85rem;">
            {_pill("Overall", report.overall.get("overall_role_alignment", ""))}
            {_pill("Core", report.overall.get("core_requirement_coverage", ""))}
            {_pill("Depth", report.overall.get("technical_depth", ""))}
            {_pill("Impact", report.overall.get("demonstrated_impact", ""))}
            {_pill("Seniority", report.overall.get("seniority_alignment", report.seniority.get("alignment", "")))}
            {_pill("ATS", report.overall.get("ats_compatibility", report.ats.get("rating", "")))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(report.score_explanation)

    tab_overview, tab_coverage, tab_evidence, tab_improve, tab_ats, tab_raw = st.tabs(
        [
            "Overview",
            "Requirement coverage",
            "Evidence graph",
            "Improvements",
            "ATS / consistency",
            "Full report",
        ]
    )

    with tab_overview:
        st.markdown("#### Alignment summary")
        _show_table(_overview_alignment_table(report))

        st.markdown("#### Core strengths")
        strength_rows = report.strength_table or [
            {
                "Requirement": _short(s.split(":")[0], 70),
                "Importance": "—",
                "Strength": "Strong",
                "Evidence": _short(s, 100),
            }
            for s in report.core_strengths
        ]
        _show_table(strength_rows)

        st.markdown("#### Major gaps")
        gap_rows = report.gap_table or (
            [{"Requirement": "—", "Importance": "—", "Strength": "—", "Gap": g} for g in report.major_gaps]
            if report.major_gaps
            else []
        )
        _show_table(gap_rows)

        st.markdown("#### Seniority alignment")
        _show_table(
            [
                {
                    "Target level": report.seniority.get("target_level", "—"),
                    "Demonstrated": report.seniority.get("demonstrated_level", "—"),
                    "Alignment": report.seniority.get("alignment", "—"),
                    "Notes": _short(report.seniority.get("notes", ""), 120),
                }
            ]
        )

        st.markdown("#### Transferable skills")
        _show_table(
            [{"Skill / concept": _short(s, 140)} for s in report.transferable_skills[:10]],
            empty_message="None highlighted",
        )

        st.markdown("#### Impact / accomplishment analysis")
        _show_table(
            [{"Finding": s} for s in report.impact_analysis],
            empty_message="Limited measurable impact language",
        )

    with tab_coverage:
        _show_table(_coverage_table(report), empty_message="No requirements parsed.")

    with tab_evidence:
        st.markdown("#### Evidence by requirement")
        _show_table(_evidence_flat_table(report), empty_message="No evidence graph available.")

        st.markdown("#### Bullet-level analysis")
        _show_table(_bullet_table(report), empty_message="No bullets analyzed.")

    with tab_improve:
        st.markdown("#### Top resume improvements")
        _show_table(
            [{"#": i, "Improvement": item} for i, item in enumerate(report.improvements, 1)],
            empty_message="No priority improvements identified.",
        )

        st.markdown("#### Missing-evidence recommendations")
        _show_table(
            [{"Recommendation": m} for m in report.missing_evidence],
            empty_message="None",
        )

        st.markdown("#### Bullet rewrite opportunities")
        st.caption("Preserves your experience — never invents metrics, tech, or ownership.")
        _show_table(
            [
                {
                    "Original": _short(r["original"], 80),
                    "Improved structure": _short(r["improved_structure"], 120),
                    "Still needed from you": _short(
                        "; ".join(r.get("needs_from_candidate") or []), 80
                    ),
                }
                for r in report.rewrites
            ],
            empty_message="No rewrite suggestions.",
        )

        if report.unsupported_claims:
            st.markdown("#### Weak or unsupported claims")
            _show_table(
                [{"Skill": c["skill"], "Note": c["note"]} for c in report.unsupported_claims]
            )

    with tab_ats:
        st.markdown(
            f"**ATS compatibility:** {report.ats.get('rating')} ({report.ats.get('score')}/100)"
        )
        st.caption(report.ats.get("notes", ""))
        ats_rows = [{"Type": "Positive", "Detail": p} for p in report.ats.get("positives") or []]
        ats_rows += [{"Type": "Finding", "Detail": f} for f in report.ats.get("findings") or []]
        _show_table(ats_rows, empty_message="No ATS notes.")

        st.markdown("#### Consistency review items")
        _show_table(
            [
                {
                    "Severity": i.get("severity", "review"),
                    "Kind": i.get("kind", "—"),
                    "Detail": _short(i.get("detail", ""), 120),
                }
                for i in report.consistency_issues
            ],
            empty_message="None flagged.",
        )

    with tab_raw:
        st.download_button(
            "Download markdown report",
            data=report.to_markdown(),
            file_name=f"{Path(report.filename).stem}_match_report.md",
            mime="text/markdown",
            key=f"dl-{report.filename}",
        )
        st.markdown(report.to_markdown())


def main() -> None:
    st.set_page_config(
        page_title="Resume Checker",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_STYLES, unsafe_allow_html=True)

    st.title("Resume Checker")
    st.caption(
        "Evidence-based job-fit analysis — semantic matching, requirement coverage, "
        "and traceable resume evidence. Not a keyword counter."
    )

    with st.sidebar:
        st.header("Assessment settings")
        threshold = st.slider(
            "Greenlight threshold (composite score)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="Informational score ≥ this value is greenlit. "
            "Score reflects evidence strength vs. weighted requirements — "
            "not probability of being hired.",
        )
        st.divider()
        st.markdown(
            """
            **How this works**
            1. Parse the JD into weighted requirements
            2. Match related concepts (not just exact words)
            3. Build an evidence graph from resume bullets
            4. Score evidence strength, depth, ownership, impact
            5. Assess seniority, transferable skills, ATS (separately)
            """
        )

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Job description")
        job_text = st.text_area(
            "Paste the job description",
            height=340,
            placeholder="Paste the full job description here…",
            label_visibility="collapsed",
        )
        if job_text.strip():
            lang = detect_language(job_text)
            if lang["code"] != "unknown":
                conf = int(float(lang["confidence"]) * 100)
                st.markdown(
                    f'<span class="lang-badge">Language: {lang["name"]} '
                    f'({lang["code"]}) · {conf}% confidence</span>',
                    unsafe_allow_html=True,
                )
            reqs = parse_job_description(job_text)
            with st.expander(f"Parsed requirements ({len(reqs)})", expanded=False):
                for r in reqs[:25]:
                    concepts = ", ".join(r.related_concepts[:6])
                    st.markdown(
                        f"- **{r.importance.value}** — {r.text}"
                        + (f"  \n  _{concepts}_" if concepts else "")
                    )

    with right:
        st.subheader("Resumes")
        uploads = st.file_uploader(
            "Upload PDF or Word resumes",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Upload one or more .pdf or .docx resume files.",
        )
        pasted = st.text_area(
            "Or paste resume text (use if PDF extract fails)",
            height=160,
            placeholder="Paste resume text here as a fallback when a designed PDF extracts poorly…",
            key="pasted_resume",
        )
        parsed: list[tuple[str, str]] = []
        parse_errors: list[str] = []
        if uploads:
            for uploaded in uploads:
                try:
                    text = extract_text_from_bytes(uploaded.getvalue(), uploaded.name)
                    parsed.append((uploaded.name, text))
                except EmptyExtractError as exc:
                    parse_errors.append(str(exc))
                except UnsupportedFileTypeError as exc:
                    parse_errors.append(str(exc))
                except Exception as exc:  # noqa: BLE001
                    parse_errors.append(f"{uploaded.name}: {exc}")
        if pasted.strip():
            parsed.append(("pasted_resume.txt", pasted.strip()))

        if parsed:
            st.caption(f"{len(parsed)} resume(s) ready" + (f" · {len(parse_errors)} failed" if parse_errors else ""))
            with st.expander("Preview extracted text", expanded=False):
                for name, text in parsed:
                    st.markdown(f"**{name}** — {len(text)} characters")
                    st.code(text[:1200] + ("…" if len(text) > 1200 else ""), language=None)
        for err in parse_errors:
            st.error(err)
            st.info(
                "Tip: designed/column PDFs often fail text extraction. "
                "Export a simpler PDF, upload .docx, or paste the resume text above."
            )

    st.divider()
    run = st.button(
        "Analyze resumes",
        type="primary",
        disabled=not (job_text.strip() and parsed),
    )

    if not job_text.strip() and not uploads and not pasted.strip():
        st.info("Paste a job description and upload resumes to generate evidence-based match reports.")
        return

    if run:
        with st.spinner("Building evidence graphs and scoring…"):
            reports = analyze_resumes(
                job_description=job_text,
                resumes=parsed,
                threshold=float(threshold),
            )

        greenlit = [r for r in reports if r.greenlit]
        rejected = [r for r in reports if not r.greenlit]
        m1, m2, m3 = st.columns(3)
        m1.metric("Resumes analyzed", len(reports))
        m2.metric("Greenlit", len(greenlit))
        m3.metric("Below threshold", len(rejected))

        st.subheader("Results")
        if greenlit:
            st.markdown("#### Greenlit candidates")
            for report in greenlit:
                _render_report(report)
        if rejected:
            st.markdown("#### Below threshold")
            for report in rejected:
                _render_report(report)


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    if _running_inside_streamlit():
        main()
    else:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
        raise SystemExit(stcli.main())
elif _running_inside_streamlit():
    main()
