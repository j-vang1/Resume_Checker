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
from resume_matcher.parsers import UnsupportedFileTypeError, extract_text_from_bytes
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
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Core strengths")
            for s in report.core_strengths or ["None identified"]:
                st.markdown(f"- {s}")
        with c2:
            st.markdown("#### Major gaps")
            for g in report.major_gaps or ["None identified"]:
                st.markdown(f"- {g}")

        st.markdown("#### Seniority alignment")
        st.write(report.seniority.get("notes", ""))
        if report.seniority.get("signals"):
            st.caption("Signals: " + "; ".join(report.seniority["signals"]))

        st.markdown("#### Transferable skills")
        for s in report.transferable_skills[:8] or ["None highlighted"]:
            st.markdown(f"- {s}")

        st.markdown("#### Impact / accomplishment analysis")
        for s in report.impact_analysis or ["Limited measurable impact language"]:
            st.markdown(f"- {s}")

    with tab_coverage:
        if report.requirement_coverage:
            st.dataframe(
                [
                    {
                        "Requirement": r["requirement"][:80],
                        "Importance": r["importance"],
                        "Evidence": r["evidence_summary"][:100],
                        "Strength": r["strength"],
                        "Notes": r["notes"][:100],
                    }
                    for r in report.requirement_coverage
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No requirements parsed.")

    with tab_evidence:
        for node in report.evidence_graph:
            if node["importance"] == "Generic":
                continue
            with st.expander(
                f"{node['requirement'][:70]} · {node['importance']} · {node['strength']}",
                expanded=node["importance"] == "Core",
            ):
                if node["related_concepts"]:
                    st.caption("Related concepts: " + ", ".join(node["related_concepts"][:10]))
                if not node["evidence"]:
                    st.write("No supporting resume evidence found.")
                for ev in node["evidence"]:
                    st.markdown(f"- “{ev['bullet']}”")
                    st.caption(
                        f"Relevance {ev['relevance']:.2f} · Ownership {ev['ownership']} · "
                        f"Depth {ev['technical_depth']} · Impact {ev['impact']}"
                    )
                st.write(node["notes"])

        st.markdown("#### Bullet-level analysis")
        for b in report.bullet_analyses:
            with st.expander(b["bullet"][:90] + ("…" if len(b["bullet"]) > 90 else "")):
                st.write(
                    f"Relevance: **{b['relevance']}** · Depth: **{b['technical_depth']}** · "
                    f"Ownership: **{b['ownership']}** · Impact: **{b['impact']}** · "
                    f"Clarity: **{b['clarity']}** · Specificity: **{b['specificity']}**"
                )
                st.write(b["feedback"])

    with tab_improve:
        st.markdown("#### Top resume improvements")
        for i, item in enumerate(report.improvements, 1):
            st.markdown(f"{i}. {item}")

        st.markdown("#### Missing-evidence recommendations")
        for m in report.missing_evidence or ["None"]:
            st.markdown(f"- {m}")

        st.markdown("#### Bullet rewrite opportunities")
        st.caption("Preserves your experience — never invents metrics, tech, or ownership.")
        for r in report.rewrites:
            st.markdown(f"**Original:** {r['original']}")
            st.markdown(f"**Improved structure:** {r['improved_structure']}")
            if r.get("needs_from_candidate"):
                st.caption("Still needed from you: " + "; ".join(r["needs_from_candidate"]))
            st.divider()

        if report.unsupported_claims:
            st.markdown("#### Weak or unsupported claims")
            for c in report.unsupported_claims:
                st.markdown(f"- **{c['skill']}** — {c['note']}")

    with tab_ats:
        st.markdown(f"**ATS compatibility:** {report.ats.get('rating')} ({report.ats.get('score')}/100)")
        st.caption(report.ats.get("notes", ""))
        if report.ats.get("positives"):
            st.markdown("Positives")
            for p in report.ats["positives"]:
                st.markdown(f"- {p}")
        if report.ats.get("findings"):
            st.markdown("Findings")
            for f in report.ats["findings"]:
                st.markdown(f"- {f}")
        st.markdown("#### Consistency review items")
        if report.consistency_issues:
            for i in report.consistency_issues:
                st.markdown(f"- ({i['severity']}) {i['detail']}")
        else:
            st.write("None flagged.")

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
        parsed: list[tuple[str, str]] = []
        parse_errors: list[str] = []
        if uploads:
            for uploaded in uploads:
                try:
                    text = extract_text_from_bytes(uploaded.getvalue(), uploaded.name)
                    if not text.strip():
                        parse_errors.append(
                            f"{uploaded.name}: no extractable text (scanned PDFs need OCR)."
                        )
                    else:
                        parsed.append((uploaded.name, text))
                except UnsupportedFileTypeError as exc:
                    parse_errors.append(str(exc))
                except Exception as exc:  # noqa: BLE001
                    parse_errors.append(f"{uploaded.name}: {exc}")
            st.caption(
                f"{len(parsed)} resume(s) ready"
                + (f" · {len(parse_errors)} failed" if parse_errors else "")
            )
            for err in parse_errors:
                st.warning(err)

    st.divider()
    run = st.button(
        "Analyze resumes",
        type="primary",
        disabled=not (job_text.strip() and parsed),
    )

    if not job_text.strip() and not uploads:
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
