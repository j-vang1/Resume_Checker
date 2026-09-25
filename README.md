# Resume Checker

Evidence-based resume ↔ job-description analysis. The checker estimates how convincingly a resume demonstrates that a candidate can **perform a specific job** — using semantic concept matching, evidence graphs, and strength ratings — **not** exact keyword counts.

## Features

- Parse JDs into weighted requirements (Core / Important / Preferred / Supporting)
- Semantic matching via related concepts (e.g. “investigated intermittent failures” ↔ root cause analysis)
- Evidence graph linking each requirement to real resume bullets
- Evidence strength, technical depth, ownership, and impact scoring
- Seniority alignment, transferable skills, unsupported skill flags
- ATS compatibility assessed **separately** from job-fit
- Tunable greenlight threshold (default 50) on an informational composite score
- Full markdown report download

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
# or: python app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q
```

## Project layout

```
app.py                     # Streamlit UI
resume_matcher/
  analyzer.py              # Orchestrator
  jd_parser.py             # JD → weighted requirements
  resume_parse.py          # Resume → bullets / skills / dates
  concepts.py              # Concept / synonym graph
  evidence.py              # Evidence graph + strength
  signals.py               # Seniority, impact, consistency
  ats.py                   # ATS compatibility (separate)
  report.py                # Final report assembly
  language.py / parsers.py
tests/
```

## Scoring note

The composite score is informational and weighted by requirement importance + evidence strength. It is **not** a probability of being hired.
