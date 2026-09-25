# Resume Matcher

Paste a job description, upload multiple PDF/Word resumes, and score each candidate with **keyword overlap + TF-IDF similarity** (same scoring as PR #1). Resumes at or above your threshold (default **50%**) are greenlit.

Results are shown in a **color-coded screening board** (tables) so you can quickly see who’s a good / moderate / poor fit.

## Features

- Job description paste with automatic **language detection**
- Multi-file upload for `.pdf` / `.docx` (+ paste-text fallback)
- Match scoring: keyword overlap + TF-IDF cosine similarity
- Tunable greenlight threshold and keyword weight
- Screening board with filter / sort / CSV export and per-candidate keyword tables

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## How matching works

1. Extract keywords from the job description
2. Measure keyword overlap with each resume
3. Measure TF-IDF cosine similarity
4. Combined score (default): `55% × keywords + 45% × similarity`
5. Greenlight if score ≥ threshold (default 50%)

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q
```
