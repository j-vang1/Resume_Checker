"""
Resume Matcher — paste a job description, upload resumes, get match scores.

PR #1 scoring (keyword overlap + TF-IDF) with a table-first screening board.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import streamlit as st

from resume_matcher.language import detect_language
from resume_matcher.matching import MatchResult, extract_keywords, score_resumes
from resume_matcher.parsers import (
    EmptyExtractError,
    UnsupportedFileTypeError,
    extract_text_from_bytes,
)

_STYLES = """
<style>
  .block-container { padding-top: 1.1rem; max-width: 1400px; }
  .match-card, .match-card * { color: #0f172a !important; }
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
  .kw {
    display: inline-block; background: #f1f5f9; border-radius: 6px;
    padding: 0.15rem 0.5rem; margin: 0.15rem; font-size: 0.8rem; color: #0f172a !important;
  }
  .kw.hit { background: #d1fae5; color: #065f46 !important; }
  .kw.miss { background: #fee2e2; color: #9f1239 !important; }
</style>
"""


def _fit_tier(result: MatchResult) -> str:
    if result.error:
        return "poor"
    if result.match_percent >= 70:
        return "good"
    if result.match_percent >= 50 or result.greenlit:
        return "maybe"
    return "poor"


def _fit_label(tier: str) -> str:
    return {"good": "GOOD FIT", "maybe": "MODERATE", "poor": "POOR FIT"}.get(tier, "POOR FIT")


def _short(text: str, limit: int = 90) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _board_records(results: list[MatchResult]) -> list[dict]:
    rows = []
    for i, r in enumerate(results, 1):
        tier = _fit_tier(r)
        marker = {"good": "✅", "maybe": "🟡", "poor": "🔴"}.get(tier, "🔴")
        rows.append(
            {
                "#": i,
                "Fit": f"{marker} {_fit_label(tier)}",
                "Score": float(round(r.match_percent, 1)),
                "Candidate": r.filename,
                "Gate": "GREENLIT" if r.greenlit else "BELOW",
                "Keywords %": round(r.keyword_overlap_percent, 1),
                "Similarity %": round(r.semantic_similarity_percent, 1),
                "Matched": ", ".join(r.matched_keywords[:8]) or "—",
                "Missing": ", ".join(r.missing_keywords[:8]) or "—",
                "_tier": tier,
            }
        )
    return rows


def _show_board(results: list[MatchResult]) -> None:
    if not results:
        st.caption("No candidates in this view.")
        return
    rows = _board_records(results)
    display = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    try:
        import pandas as pd

        df = pd.DataFrame(display)

        def _row_style(row):
            tier = rows[row.name]["_tier"]
            colors = {
                "good": "background-color: #dcfce7; color: #14532d",
                "maybe": "background-color: #fef9c3; color: #713f12",
                "poor": "background-color: #fee2e2; color: #7f1d1d",
            }
            return [colors.get(tier, "")] * len(row)

        st.dataframe(
            df.style.apply(_row_style, axis=1),
            use_container_width=True,
            hide_index=True,
            height=min(560, 48 + 36 * max(len(df), 3)),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%.0f", width="small"
                ),
                "Candidate": st.column_config.TextColumn("Candidate", width="large"),
                "Matched": st.column_config.TextColumn("Matched", width="medium"),
                "Missing": st.column_config.TextColumn("Missing", width="medium"),
            },
        )
    except Exception:  # noqa: BLE001
        st.dataframe(display, use_container_width=True, hide_index=True)


def _board_csv(results: list[MatchResult]) -> str:
    buf = io.StringIO()
    fieldnames = [
        "Fit",
        "Score",
        "Candidate",
        "Gate",
        "Keywords %",
        "Similarity %",
        "Matched",
        "Missing",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in _board_records(results):
        writer.writerow({k: row[k] for k in fieldnames})
    return buf.getvalue()


def _filter_sort(
    results: list[MatchResult],
    *,
    fit_filter: str,
    status_filter: str,
    min_score: float,
    search: str,
    sort_by: str,
) -> list[MatchResult]:
    out = [r for r in results if not r.error or True]
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

    out = [r for r in out if r.match_percent >= min_score]
    if q:
        out = [r for r in out if q in r.filename.lower()]

    tier_order = {"good": 0, "maybe": 1, "poor": 2}
    if sort_by == "Score (low → high)":
        out.sort(key=lambda r: r.match_percent)
    elif sort_by == "Score (high → low)":
        out.sort(key=lambda r: r.match_percent, reverse=True)
    elif sort_by == "Name (A → Z)":
        out.sort(key=lambda r: r.filename.lower())
    elif sort_by == "Name (Z → A)":
        out.sort(key=lambda r: r.filename.lower(), reverse=True)
    else:
        out.sort(key=lambda r: (tier_order[_fit_tier(r)], -r.match_percent))
    return out


def _render_detail(result: MatchResult, threshold: float) -> None:
    if result.error:
        st.error(f"**{result.filename}** — {result.error}")
        return

    green = result.greenlit
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
              <div class="filename" style="margin-top:0.45rem;">{result.filename}</div>
              <div class="meta">Threshold {threshold:.0f}% · Fit {_fit_label(_fit_tier(result))}</div>
            </div>
            <div class="score-big {score_cls}">{result.match_percent:.1f}%</div>
          </div>
          <div class="meta" style="margin-top:0.75rem;">
            Keyword overlap: <b>{result.keyword_overlap_percent:.1f}%</b>
            &nbsp;·&nbsp;
            Text similarity: <b>{result.semantic_similarity_percent:.1f}%</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_keywords = st.tabs(["Overview", "Phrases"])
    with tab_overview:
        st.dataframe(
            [
                {
                    "Metric": "Match %",
                    "Value": f"{result.match_percent:.1f}%",
                },
                {
                    "Metric": "Keyword overlap",
                    "Value": f"{result.keyword_overlap_percent:.1f}%",
                },
                {
                    "Metric": "TF-IDF similarity",
                    "Value": f"{result.semantic_similarity_percent:.1f}%",
                },
                {
                    "Metric": "Greenlit",
                    "Value": "Yes" if result.greenlit else "No",
                },
                {
                    "Metric": "Fit tier",
                    "Value": _fit_label(_fit_tier(result)),
                },
            ],
            use_container_width=True,
            hide_index=True,
        )
    with tab_keywords:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Matched phrases**")
            if result.matched_keywords:
                st.markdown(
                    " ".join(f'<span class="kw hit">{kw}</span>' for kw in result.matched_keywords),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("None")
        with c2:
            st.markdown("**Missing from resume**")
            if result.missing_keywords:
                st.markdown(
                    " ".join(f'<span class="kw miss">{kw}</span>' for kw in result.missing_keywords),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("None")


def _render_results(results: list[MatchResult], threshold: float) -> None:
    good = [r for r in results if _fit_tier(r) == "good"]
    maybe = [r for r in results if _fit_tier(r) == "maybe"]
    poor = [r for r in results if _fit_tier(r) == "poor"]
    greenlit_n = sum(1 for r in results if r.greenlit)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("✅ Good fit (≥70%)", len(good))
    t2.metric("🟡 Moderate", len(maybe))
    t3.metric("🔴 Poor fit", len(poor))
    t4.metric("Greenlit", greenlit_n)

    st.markdown("### Screening board")
    c1, c2, c3, c4 = st.columns([1.2, 1.1, 1.1, 1.4])
    with c1:
        fit_filter = st.selectbox(
            "Fit tier",
            ["All", "Good fit only", "Moderate only", "Poor fit only"],
        )
    with c2:
        status_filter = st.selectbox(
            "Greenlight gate",
            ["All", "Greenlit only", "Below threshold only"],
        )
    with c3:
        min_score = st.slider("Min score", 0, 100, 0, 5)
    with c4:
        sort_by = st.selectbox(
            "Sort",
            [
                "Best first (fit + score)",
                "Score (high → low)",
                "Score (low → high)",
                "Name (A → Z)",
                "Name (Z → A)",
            ],
        )

    search = st.text_input("Search candidate filename", placeholder="Type to filter by name…")
    filtered = _filter_sort(
        results,
        fit_filter=fit_filter,
        status_filter=status_filter,
        min_score=float(min_score),
        search=search,
        sort_by=sort_by,
    )

    left_meta, right_meta = st.columns([2, 1])
    with left_meta:
        st.caption(
            f"Showing {len(filtered)} of {len(results)} · "
            "Green ≥70% · Yellow greenlit/50–69% · Red below"
        )
    with right_meta:
        st.download_button(
            "Download board CSV",
            data=_board_csv(filtered),
            file_name="resume_match_board.csv",
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
    )
    if view_mode == "Grouped by fit":
        for title, tier, group in [
            ("✅ Good fit — review first", "good", [r for r in filtered if _fit_tier(r) == "good"]),
            ("🟡 Moderate — needs judgment", "maybe", [r for r in filtered if _fit_tier(r) == "maybe"]),
            ("🔴 Poor fit — likely pass", "poor", [r for r in filtered if _fit_tier(r) == "poor"]),
        ]:
            with st.expander(f"{title} ({len(group)})", expanded=bool(group) and tier != "poor"):
                _show_board(group)
    else:
        _show_board(filtered)

    st.markdown("### Open one candidate")
    options = [
        f"{_fit_label(_fit_tier(r))} · {r.match_percent:.0f}% · {r.filename}" for r in filtered
    ]
    default_idx = next((i for i, r in enumerate(filtered) if _fit_tier(r) == "good"), 0)
    selected_label = st.selectbox("Candidate", options, index=default_idx)
    selected = filtered[options.index(selected_label)]
    _render_detail(selected, threshold)


def main() -> None:
    st.set_page_config(
        page_title="Resume Matcher",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_STYLES, unsafe_allow_html=True)

    st.title("Resume Matcher")
    st.caption(
        "Paste a job description, upload resumes, score keyword + TF-IDF match. "
        "Screen with color-coded tables."
    )

    with st.sidebar:
        st.header("Match settings")
        threshold = st.slider(
            "Greenlight threshold (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="Resumes at or above this score are greenlit.",
        )
        keyword_weight = st.slider(
            "Keyword weight",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            help="1.0 = keywords only, 0.0 = TF-IDF similarity only.",
        )
        st.divider()
        st.markdown(
            f"""
            **Scoring (PR #1)**
            - Keywords from the job description (skips stock lead-ins)
            - Reads past resume lead-ins like Demonstrate / Review / Responsible for
            - Keyword overlap + TF-IDF similarity on the substance
            - Combined: `{keyword_weight:.0%} × keywords + {1 - keyword_weight:.0%} × similarity`
            - Greenlight if score ≥ **{threshold}%**
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
            else:
                st.info(lang.get("error") or "Could not detect language.")
            keywords = extract_keywords(job_text)
            if keywords:
                with st.expander("Detected phrases / keywords", expanded=False):
                    st.markdown(
                        " ".join(f'<span class="kw">{kw}</span>' for kw in keywords),
                        unsafe_allow_html=True,
                    )

    with right:
        st.subheader("Resumes")
        uploads = st.file_uploader(
            "Upload PDF or Word resumes",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )
        pasted = st.text_area(
            "Or paste resume text (if PDF extract fails)",
            height=140,
            placeholder="Paste resume text as a fallback…",
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

    st.divider()
    run = st.button(
        "Match resumes",
        type="primary",
        disabled=not (job_text.strip() and parsed),
    )

    if not job_text.strip() and not uploads and not pasted.strip():
        st.info("Paste a job description and upload resumes to begin.")
        return

    if run:
        with st.spinner("Scoring resumes…"):
            results = score_resumes(
                job_description=job_text,
                resumes=parsed,
                threshold=float(threshold),
                keyword_weight=float(keyword_weight),
            )
        st.session_state["results"] = results
        st.session_state["threshold"] = float(threshold)

    results = st.session_state.get("results")
    if results:
        current_threshold = float(threshold)
        if current_threshold != st.session_state.get("threshold"):
            for r in results:
                r.greenlit = r.match_percent >= current_threshold
            st.session_state["threshold"] = current_threshold
        _render_results(results, current_threshold)


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
