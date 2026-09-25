# Resume Matcher

Paste a job description, upload multiple PDF/Word resumes, and score each candidate by keyword overlap and text similarity. Resumes at or above your threshold (default **50%**) are greenlit.

## Features

- **Job description paste** with automatic **language detection**
- **Multi-file upload** for `.pdf` and `.docx` resumes
- **Match scoring** combining keyword overlap + TF-IDF cosine similarity
- **Tunable greenlight threshold** (default 50%) and keyword vs. similarity weight
- Per-resume breakdown of matched / missing keywords

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## How matching works

1. Keywords are extracted from the job description.
2. Each resume is scored on:
   - **Keyword overlap** — share of JD keywords found in the resume
   - **Text similarity** — TF-IDF cosine similarity between JD and resume
3. Combined score (default): `55% × keywords + 45% × similarity`
4. Resumes with score ≥ threshold are **greenlit** (default threshold: 50%)

Tune both the threshold and the keyword weight in the sidebar.

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q
```

## Project layout

```
app.py                  # Streamlit UI
resume_matcher/
  language.py           # Language detection
  parsers.py            # PDF / DOCX text extraction
  matching.py           # Scoring & greenlight logic
tests/
  test_matching.py
requirements.txt
```

## Notes

- Scanned image-only PDFs need OCR first; this app extracts embedded text only.
- Legacy `.doc` files are not supported — convert to `.docx` or `.pdf`.
