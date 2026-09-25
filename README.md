# Resume Matcher

Paste a job description, upload multiple PDF/Word resumes, and score each candidate in **three steps**: keywords → phrases → combined. Resumes at or above your threshold (default **50%**) are greenlit.

Results are shown in a **color-coded screening board** (tables) so you can quickly see who’s a good / moderate / poor fit.

## Features

- Job description paste with automatic **language detection**
- Multi-file upload for `.pdf` / `.docx` (+ paste-text fallback)
- Match scoring in steps: **keywords**, then **phrases**, then **combined** (+ TF-IDF)
- Tunable greenlight threshold, lexical weight, and phrase-vs-keyword weight
- Screening board with filter / sort / CSV export and per-candidate keyword + phrase lists

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## How matching works

1. Strip stock lead-in verbs from JD and resume bullets (`Demonstrate`, `Review`, `Responsible for`, `Led`, … — up to ~5 words)
2. **Keyword check** — extract single-word skills/terms from the JD; measure overlap with the resume
3. **Phrase check** — extract 2–5 word phrases from the JD; measure overlap with the resume
4. **Combined** — blend keyword + phrase overlaps into a lexical score, then mix with TF-IDF cosine similarity
5. Greenlight if score ≥ threshold (default 50%)

Default blend:

- lexical = `45% × keyword overlap + 55% × phrase overlap`
- score = `55% × lexical + 45% × TF-IDF`

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q
```
