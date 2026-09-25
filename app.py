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
  .block-container { padding-top: 1rem; max-width: 1400px; }

  /* Force readable dark text on light surfaces (dark-theme safe) */
  .match-card, .match-card *,
  .tier-card, .tier-card *,
  .screen-table, .screen-table * {
    color: #0f172a !important;
  }

  .tier-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 0.5rem 0 1rem 0;
  }
  .tier-card {
    border-radius: 12px;
    padding: 0.9rem 1rem;
    border: 1px solid #cbd5e1;
  }
  .tier-card.good { background: #bbf7d0; border-color: #4ade80; }
  .tier-card.maybe { background: #fde68a; border-color: #fbbf24; }
  .tier-card.poor { background: #fecaca; border-color: #f87171; }
  .tier-card .label {
    font-size: 0.8rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase;
  }
  .tier-card .count {
    font-size: 2.1rem; font-weight: 800; line-height: 1.1; margin-top: 0.15rem;
  }
  .tier-card .hint { font-size: 0.78rem; color: #334155 !important; margin-top: 0.15rem; }

  .match-card {
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.85rem;
    border: 1px solid #cbd5e1;
  }
  .match-card.greenlit { background: #dcfce7; border-color: #86efac; }
  .match-card.rejected { background: #ffedd5; border-color: #fdba74; }
  .match-card .filename { font-weight: 700; font-size: 1.05rem; }
  .match-card .meta { font-size: 0.82rem; color: #334155 !important; margin-top: 0.2rem; }
  .score-big { font-size: 2rem; font-weight: 800; line-height: 1; }
  .score-big.green { color: #14532d !important; }
  .score-big.amber { color: #9a3412 !important; }

  .pill {
    display: inline-block; border-radius: 6px; padding: 0.18rem 0.5rem;
    font-size: 0.78rem; font-weight: 700; margin: 0.15rem 0.25rem 0 0;
  }
  .pill.strong { background: #bbf7d0; color: #14532d !important; }
  .pill.moderate { background: #fde68a; color: #78350f !important; }
  .pill.weak { background: #fecaca; color: #7f1d1d !important; }

  .lang-badge {
    display: inline-block; background: #e0f2fe; color: #0c4a6e !important;
    border-radius: 999px; padding: 0.25rem 0.75rem; font-size: 0.875rem; font-weight: 600;
  }
  .status-chip {
    display: inline-block; font-weight: 800; font-size: 0.75rem;
    letter-spacing: 0.03em; padding: 0.2rem 0.55rem; border-radius: 999px;
  }
  .status-chip.pass { background: #166534; color: #ffffff !important; }
  .status-chip.fail { background: #9a3412; color: #ffffff !important; }

  .screen-wrap {
    border: 1px solid #cbd5e1; border-radius: 10px; overflow: auto;
    max-height: 560px; background: #f8fafc;
  }
  .screen-table {
    width: 100%; border-collapse: collapse; font-size: 0.9rem;
  }
  .screen-table th {
    position: sticky; top: 0; z-index: 1;
    background: #e2e8f0; text-align: left; padding: 0.55rem 0.65rem;
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: #0f172a !important; border-bottom: 1px solid #94a3b8;
  }
  .screen-table td {
    padding: 0.5rem 0.65rem; border-bottom: 1px solid #e2e8f0;
    vertical-align: middle; color: #0f172a !important;
  }
  .screen-table tr.fit-good { background: #dcfce7; }
  .screen-table tr.fit-maybe { background: #fef9c3; }
  .screen-table tr.fit-poor { background: #fee2e2; }
  .screen-table tr:hover { filter: brightness(0.97); }
  .fit-badge {
    display: inline-block; min-width: 5.2rem; text-align: center;
    font-weight: 800; font-size: 0.72rem; letter-spacing: 0.04em;
    padding: 0.22rem 0.45rem; border-radius: 999px;
  }
  .fit-badge.good { background: #166534; color: #fff !important; }
  .fit-badge.maybe { background: #a16207; color: #fff !important; }
  .fit-badge.poor { background: #991b1b; color: #fff !important; }
  .score-cell { font-weight: 800; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .bar {
    display: inline-block; height: 8px; width: 72px; background: #cbd5e1;
    border-radius: 999px; overflow: hidden; vertical-align: middle; margin-left: 0.4rem;
  }
  .bar > span { display: block; height: 100%; border-radius: 999px; }
  .bar.good > span { background: #16a34a; }
  .bar.maybe > span { background: #ca8a04; }
  .bar.poor > span { background: #dc2626; }
  .name-cell { font-weight: 650; max-width: 280px; }
  .muted { color: #475569 !important; font-size: 0.8rem; }
</style>
"""


def _pill(label: str, value: str) -> str:
    cls = (value or "weak").lower().split()[0]
    if cls not in {"strong", "moderate", "weak"}:
        cls = "moderate"
    return f'<span class="pill {cls}">{label}: {value}</span>'


def _short(text: str, limit: int = 90) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _strength_label(raw: str) -> str:
    return (raw or "").replace(" Evidence", "").replace("No Evidence", "None")


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
                    "Similarity": ev.get("semantic_similarity", ev.get("relevance", 0)),
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


def _fit_tier(report: MatchReport) -> str:
    """Return good | maybe | poor for bulk screening."""
    alignment = report.overall.get("overall_role_alignment", "Weak")
    if alignment == "Strong" or (report.greenlit and report.composite_score >= 70):
        return "good"
    if alignment == "Moderate" or (report.greenlit and report.composite_score >= 50):
        return "maybe"
    return "poor"


def _fit_label(tier: str) -> str:
    return {"good": "GOOD FIT", "maybe": "MODERATE", "poor": "POOR FIT"}.get(tier, "POOR FIT")


def _top_signal(report: MatchReport) -> str:
    if report.core_strengths:
        return _short(report.core_strengths[0], 70)
    if report.major_gaps:
        return _short("Gap: " + report.major_gaps[0], 70)
    return "—"


def _leaderboard_rows(reports: list[MatchReport]) -> list[dict]:
    rows = []
    for i, r in enumerate(reports, 1):
        tier = _fit_tier(r)
        rows.append(
            {
                "#": i,
                "Fit": _fit_label(tier),
                "Status": "GREENLIT" if r.greenlit else "BELOW",
                "Score": round(r.composite_score, 1),
                "Candidate": r.filename,
                "Overall": r.overall.get("overall_role_alignment", "—"),
                "Core": r.overall.get("core_requirement_coverage", "—"),
                "Depth": r.overall.get("technical_depth", "—"),
                "Impact": r.overall.get("demonstrated_impact", "—"),
                "Seniority": r.overall.get(
                    "seniority_alignment", r.seniority.get("alignment", "—")
                ),
                "ATS": r.overall.get("ats_compatibility", r.ats.get("rating", "—")),
                "Signal": _top_signal(r),
            }
        )
    return rows


def _filter_sort_reports(
    reports: list[MatchReport],
    *,
    fit_filter: str,
    status_filter: str,
    min_score: float,
    search: str,
    sort_by: str,
) -> list[MatchReport]:
    out = list(reports)
    q = (search or "").strip().lower()

    if fit_filter == "Good fit only":
        out = [r for r in out if _fit_tier(r) == "good"]
    elif fit_filter == "Moderate only":
        out = [r for r in out if _fit_tier(r) == "maybe"]
    elif fit_filter == "Poor fit only":
        out = [r for r in out if _fit_tier(r) == "poor"]

    if status_filter == "Greenlit only":
        out = [r for r in out if r.greenlit]
    elif status_filter == "Below threshold only":
        out = [r for r in out if not r.greenlit]

    out = [r for r in out if r.composite_score >= min_score]

    if q:
        out = [r for r in out if q in r.filename.lower()]

    tier_order = {"good": 0, "maybe": 1, "poor": 2}
    if sort_by == "Score (low → high)":
        out.sort(key=lambda r: r.composite_score)
    elif sort_by == "Score (high → low)":
        out.sort(key=lambda r: r.composite_score, reverse=True)
    elif sort_by == "Name (A → Z)":
        out.sort(key=lambda r: r.filename.lower())
    elif sort_by == "Name (Z → A)":
        out.sort(key=lambda r: r.filename.lower(), reverse=True)
    elif sort_by == "Fit (good → poor)":
        out.sort(key=lambda r: (tier_order[_fit_tier(r)], -r.composite_score))
    else:  # Best first (fit + score)
        out.sort(key=lambda r: (tier_order[_fit_tier(r)], -r.composite_score))

    return out


def _board_records(reports: list[MatchReport]) -> list[dict]:
    """Compact rows for the screening dataframe."""
    rows = []
    for i, r in enumerate(reports, 1):
        tier = _fit_tier(r)
        marker = {"good": "✅", "maybe": "🟡", "poor": "🔴"}.get(tier, "🔴")
        rows.append(
            {
                "#": i,
                "Fit": f"{marker} {_fit_label(tier)}",
                "Score": float(round(r.composite_score, 1)),
                "Candidate": r.filename,
                "Gate": "GREENLIT" if r.greenlit else "BELOW",
                "Core": r.overall.get("core_requirement_coverage", "—"),
                "Depth": r.overall.get("technical_depth", "—"),
                "Impact": r.overall.get("demonstrated_impact", "—"),
                "Top signal": _top_signal(r),
                "_tier": tier,
            }
        )
    return rows


def _show_screening_board(reports: list[MatchReport]) -> None:
    """Native Streamlit table — no custom HTML (avoids raw-code rendering bugs)."""
    if not reports:
        st.caption("No candidates in this view.")
        return

    rows = _board_records(reports)
    # Drop internal helper column from display
    display_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]

    try:
        import pandas as pd

        df = pd.DataFrame(display_rows)

        def _row_style(row):
            tier = rows[row.name]["_tier"]
            colors = {
                "good": "background-color: #dcfce7; color: #14532d",
                "maybe": "background-color: #fef9c3; color: #713f12",
                "poor": "background-color: #fee2e2; color: #7f1d1d",
            }
            style = colors.get(tier, "")
            return [style] * len(row)

        styled = df.style.apply(_row_style, axis=1)
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            height=min(560, 48 + 36 * max(len(df), 3)),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=100,
                    format="%.0f",
                    width="small",
                ),
                "Fit": st.column_config.TextColumn("Fit", width="medium"),
                "Candidate": st.column_config.TextColumn("Candidate", width="large"),
                "Top signal": st.column_config.TextColumn("Top signal", width="large"),
                "#": st.column_config.NumberColumn("#", width="small"),
            },
        )
    except Exception:  # noqa: BLE001 — fallback without pandas styling
        st.dataframe(
            display_rows,
            use_container_width=True,
            hide_index=True,
            height=min(560, 48 + 36 * max(len(display_rows), 3)),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=100,
                    format="%.0f",
                ),
            },
        )


def _board_csv(reports: list[MatchReport]) -> str:
    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "Fit",
            "Status",
            "Score",
            "Candidate",
            "Overall",
            "Core",
            "Depth",
            "Impact",
            "Seniority",
            "ATS",
            "Signal",
        ],
    )
    writer.writeheader()
    for row in _leaderboard_rows(reports):
        writer.writerow({k: row[k] for k in writer.fieldnames})
    return buf.getvalue()


def _render_candidate_header(report: MatchReport) -> None:
    green = report.greenlit
    card = "greenlit" if green else "rejected"
    score_cls = "green" if green else "amber"
    chip = "pass" if green else "fail"
    label = "GREENLIT" if green else "BELOW THRESHOLD"

    st.markdown(
        f"""
        <div class="match-card {card}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
            <div>
              <span class="status-chip {chip}">{label}</span>
              <div class="filename" style="margin-top:0.45rem;">{report.filename}</div>
              <div class="meta">
                Threshold {report.threshold:.0f} · Overall {report.overall.get('overall_role_alignment', '—')}
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


def _render_report_details(report: MatchReport) -> None:
    tab_overview, tab_coverage, tab_evidence, tab_improve, tab_ats, tab_raw = st.tabs(
        [
            "Overview",
            "Coverage",
            "Evidence",
            "Improvements",
            "ATS",
            "Full report",
        ]
    )

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Alignment**")
            _show_table(_overview_alignment_table(report))
            st.markdown("**Seniority**")
            _show_table(
                [
                    {
                        "Target": report.seniority.get("target_level", "—"),
                        "Demonstrated": report.seniority.get("demonstrated_level", "—"),
                        "Alignment": report.seniority.get("alignment", "—"),
                    }
                ]
            )
        with c2:
            st.markdown("**Core strengths**")
            _show_table(report.strength_table or [])
            st.markdown("**Major gaps**")
            _show_table(report.gap_table or [])

        st.markdown("**Transferable skills**")
        _show_table(
            [{"Skill / concept": _short(s, 140)} for s in report.transferable_skills[:10]],
            empty_message="None highlighted",
        )
        st.markdown("**Impact findings**")
        _show_table(
            [{"Finding": s} for s in report.impact_analysis],
            empty_message="Limited measurable impact language",
        )

    with tab_coverage:
        _show_table(_coverage_table(report), empty_message="No requirements parsed.")

    with tab_evidence:
        _show_table(_evidence_flat_table(report), empty_message="No evidence graph available.")
        st.markdown("**Bullet-level analysis**")
        _show_table(_bullet_table(report), empty_message="No bullets analyzed.")

    with tab_improve:
        _show_table(
            [{"#": i, "Improvement": item} for i, item in enumerate(report.improvements, 1)],
            empty_message="No priority improvements identified.",
        )
        st.markdown("**Missing evidence**")
        _show_table(
            [{"Recommendation": m} for m in report.missing_evidence],
            empty_message="None",
        )
        st.markdown("**Rewrite opportunities**")
        st.caption("Preserves actual experience — never invents metrics or ownership.")
        _show_table(
            [
                {
                    "Original": _short(r["original"], 80),
                    "Improved structure": _short(r["improved_structure"], 120),
                    "Still needed": _short("; ".join(r.get("needs_from_candidate") or []), 80),
                }
                for r in report.rewrites
            ],
            empty_message="No rewrite suggestions.",
        )
        if report.unsupported_claims:
            st.markdown("**Unsupported claims**")
            _show_table(
                [{"Skill": c["skill"], "Note": c["note"]} for c in report.unsupported_claims]
            )

    with tab_ats:
        st.markdown(
            f"**ATS:** {report.ats.get('rating')} ({report.ats.get('score')}/100)"
        )
        st.caption(report.ats.get("notes", ""))
        ats_rows = [{"Type": "Positive", "Detail": p} for p in report.ats.get("positives") or []]
        ats_rows += [{"Type": "Finding", "Detail": f} for f in report.ats.get("findings") or []]
        _show_table(ats_rows, empty_message="No ATS notes.")
        st.markdown("**Consistency**")
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


def _render_results(reports: list[MatchReport]) -> None:
    good = [r for r in reports if _fit_tier(r) == "good"]
    maybe = [r for r in reports if _fit_tier(r) == "maybe"]
    poor = [r for r in reports if _fit_tier(r) == "poor"]
    greenlit_n = sum(1 for r in reports if r.greenlit)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("✅ Good fit", len(good), help="Strong match — prioritize first")
    t2.metric("🟡 Moderate", len(maybe), help="Partial match — review carefully")
    t3.metric("🔴 Poor fit", len(poor), help="Weak evidence — likely pass")
    t4.metric("Greenlit", greenlit_n, help=f"Of {len(reports)} resumes analyzed")

    st.markdown("### Screening board")
    c1, c2, c3, c4 = st.columns([1.2, 1.1, 1.1, 1.4])
    with c1:
        fit_filter = st.selectbox(
            "Fit tier",
            ["All", "Good fit only", "Moderate only", "Poor fit only"],
            index=0,
        )
    with c2:
        status_filter = st.selectbox(
            "Greenlight gate",
            ["All", "Greenlit only", "Below threshold only"],
            index=0,
        )
    with c3:
        min_score = st.slider("Min score", 0, 100, 0, 5)
    with c4:
        sort_by = st.selectbox(
            "Sort",
            [
                "Best first (fit + score)",
                "Fit (good → poor)",
                "Score (high → low)",
                "Score (low → high)",
                "Name (A → Z)",
                "Name (Z → A)",
            ],
            index=0,
        )

    search = st.text_input("Search candidate filename", placeholder="Type to filter by name…")

    filtered = _filter_sort_reports(
        reports,
        fit_filter=fit_filter,
        status_filter=status_filter,
        min_score=float(min_score),
        search=search,
        sort_by=sort_by,
    )

    left_meta, right_meta = st.columns([2, 1])
    with left_meta:
        st.caption(
            f"Showing {len(filtered)} of {len(reports)} · "
            "Green rows = good · Yellow = moderate · Red = poor"
        )
    with right_meta:
        st.download_button(
            "Download board CSV",
            data=_board_csv(filtered),
            file_name="resume_screening_board.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if not filtered:
        st.warning("No candidates match the current filters.")
        return

    view_mode = st.radio(
        "Board layout",
        ["All candidates", "Grouped by fit"],
        horizontal=True,
        index=0,
    )

    if view_mode == "Grouped by fit":
        groups = [
            ("✅ Good fit — review first", "good", [r for r in filtered if _fit_tier(r) == "good"]),
            ("🟡 Moderate — needs judgment", "maybe", [r for r in filtered if _fit_tier(r) == "maybe"]),
            ("🔴 Poor fit — likely pass", "poor", [r for r in filtered if _fit_tier(r) == "poor"]),
        ]
        for title, tier, group in groups:
            with st.expander(f"{title} ({len(group)})", expanded=bool(group) and tier != "poor"):
                _show_screening_board(group)
    else:
        _show_screening_board(filtered)

    st.markdown("### Open one candidate")
    options = [
        f"{_fit_label(_fit_tier(r))} · {r.composite_score:.0f} · {r.filename}"
        for r in filtered
    ]
    default_idx = 0
    for i, r in enumerate(filtered):
        if _fit_tier(r) == "good":
            default_idx = i
            break
    selected_label = st.selectbox("Candidate", options, index=default_idx)
    selected = filtered[options.index(selected_label)]

    _render_candidate_header(selected)
    _render_report_details(selected)


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
        "Screen 100 resumes at a glance: green = good fit, yellow = moderate, red = poor fit."
    )

    with st.sidebar:
        st.header("Settings")
        from resume_matcher.llm_reasoner import llm_available
        from resume_matcher.semantic import scoring_backend_info, warmup

        try:
            backend_label = warmup()
        except Exception:  # noqa: BLE001
            backend_label = scoring_backend_info()["label"]
        st.success(f"Scoring: {backend_label}")
        if llm_available():
            st.info("OPENAI_API_KEY detected — LLM judge enabled for core requirements.")
        else:
            st.caption(
                "Set OPENAI_API_KEY to add an LLM judge on top of embeddings."
            )

        threshold = st.slider(
            "Greenlight threshold",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="Score ≥ this value is greenlit. Informational only — not a hire probability.",
        )
        st.divider()
        st.markdown(
            """
            **Flow**
            1. Paste JD
            2. Upload / paste resumes
            3. Analyze
            4. Filter & sort the board
            5. Open one candidate for detail
            """
        )

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Job description")
        job_text = st.text_area(
            "Paste the job description",
            height=300,
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
            "Or paste resume text (if PDF extract fails)",
            height=140,
            placeholder="Paste resume text here as a fallback…",
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
            st.caption(
                f"{len(parsed)} resume(s) ready"
                + (f" · {len(parse_errors)} failed" if parse_errors else "")
            )
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
        st.info("Paste a job description and upload resumes to begin.")
        return

    if run:
        with st.spinner("Scoring resumes…"):
            reports = analyze_resumes(
                job_description=job_text,
                resumes=parsed,
                threshold=float(threshold),
            )
        st.session_state["reports"] = reports
        st.session_state["threshold"] = float(threshold)

    reports = st.session_state.get("reports")
    if reports:
        # Re-apply threshold from sidebar without re-running full analysis
        current_threshold = float(threshold)
        if current_threshold != st.session_state.get("threshold"):
            for r in reports:
                r.threshold = current_threshold
                r.greenlit = r.composite_score >= current_threshold
            st.session_state["threshold"] = current_threshold
        _render_results(reports)


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
