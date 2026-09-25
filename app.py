"""
Resume Matcher — paste a job description, upload resumes, get match scores.
"""

from __future__ import annotations

import streamlit as st

from resume_matcher.language import detect_language
from resume_matcher.matching import extract_keywords, score_resumes
from resume_matcher.parsers import UnsupportedFileTypeError, extract_text_from_bytes

st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; max-width: 1200px; }
      .match-card {
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border: 1px solid #e2e8f0;
      }
      .match-card.greenlit {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-color: #6ee7b7;
      }
      .match-card.rejected {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border-color: #fda4af;
      }
      .match-pct {
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1;
      }
      .match-pct.green { color: #047857; }
      .match-pct.red { color: #be123c; }
      .lang-badge {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.875rem;
        font-weight: 600;
      }
      .kw {
        display: inline-block;
        background: #f1f5f9;
        border-radius: 6px;
        padding: 0.15rem 0.5rem;
        margin: 0.15rem;
        font-size: 0.8rem;
      }
      .kw.hit { background: #d1fae5; color: #065f46; }
      .kw.miss { background: #fee2e2; color: #9f1239; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_result_card(result, threshold: float) -> None:
    status_class = "greenlit" if result.greenlit else "rejected"
    pct_class = "green" if result.greenlit else "red"
    label = "GREENLIT" if result.greenlit else "BELOW THRESHOLD"

    if result.error:
        st.warning(f"**{result.filename}** — {result.error}")
        return

    matched_html = " ".join(f'<span class="kw hit">{kw}</span>' for kw in result.matched_keywords[:25])
    missing_html = " ".join(f'<span class="kw miss">{kw}</span>' for kw in result.missing_keywords[:25])

    st.markdown(
        f"""
        <div class="match-card {status_class}">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;">
            <div>
              <div style="font-weight:700;font-size:1.05rem;">{result.filename}</div>
              <div style="font-size:0.8rem;opacity:0.75;margin-top:0.2rem;">{label} · threshold {threshold:.0f}%</div>
            </div>
            <div class="match-pct {pct_class}">{result.match_percent:.1f}%</div>
          </div>
          <div style="margin-top:0.75rem;font-size:0.85rem;opacity:0.85;">
            Keyword overlap: <b>{result.keyword_overlap_percent:.1f}%</b>
            &nbsp;·&nbsp;
            Text similarity: <b>{result.semantic_similarity_percent:.1f}%</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Keyword details — {result.filename}", expanded=False):
        if result.matched_keywords:
            st.markdown("**Matched keywords**")
            st.markdown(matched_html or "—", unsafe_allow_html=True)
        if result.missing_keywords:
            st.markdown("**Missing from resume**")
            st.markdown(missing_html or "—", unsafe_allow_html=True)


def main() -> None:
    st.title("Resume Matcher")
    st.caption(
        "Paste a job description, upload resumes (PDF / Word), and filter candidates by match score."
    )

    # ---- Sidebar: threshold + scoring weights ----
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
            help="How much keyword overlap counts vs. overall text similarity. "
            "1.0 = keywords only, 0.0 = TF-IDF similarity only.",
        )
        st.divider()
        st.markdown(
            f"""
            **How scoring works**
            - Extract key terms from the job description
            - Measure keyword overlap with each resume
            - Measure TF-IDF cosine similarity
            - Combine: `{keyword_weight:.0%} × keywords + {1 - keyword_weight:.0%} × similarity`
            - Greenlight if score ≥ **{threshold}%**
            """
        )

    left, right = st.columns(2, gap="large")

    # ---- Left: Job description ----
    with left:
        st.subheader("Job description")
        job_text = st.text_area(
            "Paste the job description",
            height=320,
            placeholder="Paste the full job description here…",
            label_visibility="collapsed",
        )

        lang_info = None
        if job_text.strip():
            lang_info = detect_language(job_text)
            if lang_info["code"] != "unknown":
                conf = int(float(lang_info["confidence"]) * 100)
                st.markdown(
                    f'<span class="lang-badge">Language: {lang_info["name"]} '
                    f"({lang_info['code']}) · {conf}% confidence</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(lang_info.get("error") or "Could not detect language.")

            keywords = extract_keywords(job_text)
            if keywords:
                with st.expander("Detected keywords from job description", expanded=False):
                    st.markdown(
                        " ".join(f'<span class="kw">{kw}</span>' for kw in keywords),
                        unsafe_allow_html=True,
                    )

    # ---- Right: Resume uploads ----
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
                except Exception as exc:  # noqa: BLE001 — surface parse failures in UI
                    parse_errors.append(f"{uploaded.name}: {exc}")

            st.caption(f"{len(parsed)} resume(s) ready" + (f" · {len(parse_errors)} failed" if parse_errors else ""))
            for err in parse_errors:
                st.warning(err)

    st.divider()

    run = st.button(
        "Match resumes",
        type="primary",
        disabled=not (job_text.strip() and parsed),
        use_container_width=False,
    )

    if not job_text.strip() and not uploads:
        st.info("Paste a job description on the left and upload resumes on the right to begin.")
        return

    if run:
        with st.spinner("Scoring resumes…"):
            results = score_resumes(
                job_description=job_text,
                resumes=parsed,
                threshold=float(threshold),
                keyword_weight=float(keyword_weight),
            )

        greenlit = [r for r in results if r.greenlit and not r.error]
        rejected = [r for r in results if not r.greenlit or r.error]

        m1, m2, m3 = st.columns(3)
        m1.metric("Resumes scored", len(results))
        m2.metric("Greenlit", len(greenlit))
        m3.metric("Below threshold", len(rejected))

        st.subheader("Results")
        if greenlit:
            st.markdown("#### Greenlit candidates")
            for result in greenlit:
                _render_result_card(result, float(threshold))

        if rejected:
            st.markdown("#### Below threshold")
            for result in rejected:
                _render_result_card(result, float(threshold))


main()
