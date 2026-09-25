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

1. Strip stock lead-in verbs from JD and resume bullets (`Demonstrate`, `Review`, `Responsible for`, `Led`, … — up to ~5 words)
2. Extract **2–5 word phrases** (plus strong skill unigrams) from the remaining JD substance
3. Measure phrase/keyword overlap with each resume (whole-word / whole-phrase only)
4. Measure TF-IDF cosine similarity on the stripped text
5. Combined score (default): `55% × keywords + 45% × similarity`
6. Greenlight if score ≥ threshold (default 50%)

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q
```
