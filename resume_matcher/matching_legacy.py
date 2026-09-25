"""Score resumes against a job description using keywords + TF-IDF similarity."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Tokens that are usually noise for keyword matching
_STOPWORDS = frozenset(
    """
    a an the and or but if then else when at by for with about against between
    into through during before after above below to from up down in out on off
    over under again further once here there all any both each few more most
    other some such no nor not only own same so than too very can will just
    don should now is are was were be been being have has had do does did
    of this that these those it its as we you your he she they them their our
    i me my myself yourself yourselves himself herself itself ourselves
    what which who whom how why where who whom whomst whats whos wheres
    will would could should may might must shall
    """.split()
)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}")


@dataclass
class MatchResult:
    """Match score and detail for one resume."""

    filename: str
    match_percent: float
    greenlit: bool
    keyword_overlap_percent: float
    semantic_similarity_percent: float
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def extract_keywords(text: str, top_n: int = 40) -> list[str]:
    """
    Extract important keywords from text using TF-IDF over a single document
    (term frequency with light filtering). Falls back to frequent tokens.
    """
    tokens = [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 1]
    if not tokens:
        return []

    # Prefer multi-character technical terms and repeated skills
    freq: dict[str, int] = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1

    # Score: frequency * length bonus for longer skill-like terms
    scored = sorted(
        freq.items(),
        key=lambda item: (item[1] * (1 + min(len(item[0]), 12) / 12), item[0]),
        reverse=True,
    )
    return [term for term, _ in scored[:top_n]]


def _keyword_overlap(
    job_keywords: list[str], resume_text: str
) -> tuple[float, list[str], list[str]]:
    """Return overlap %, matched keywords, and missing keywords."""
    if not job_keywords:
        return 0.0, [], []

    resume_tokens = set(_tokenize(resume_text))
    matched: list[str] = []
    missing: list[str] = []
    for kw in job_keywords:
        # Exact token match or substring for compounds like "machine-learning"
        if kw in resume_tokens or any(kw in token for token in resume_tokens):
            matched.append(kw)
        else:
            missing.append(kw)

    overlap = (len(matched) / len(job_keywords)) * 100.0
    return overlap, matched, missing


def _tfidf_similarity(job_text: str, resume_text: str) -> float:
    """Cosine similarity of TF-IDF vectors, returned as a 0–100 percentage."""
    if not job_text.strip() or not resume_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b",
    )
    try:
        matrix = vectorizer.fit_transform([job_text, resume_text])
    except ValueError:
        return 0.0

    sim = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    return max(0.0, min(100.0, sim * 100.0))


def score_resume(
    job_description: str,
    resume_text: str,
    filename: str = "resume",
    threshold: float = 50.0,
    keyword_weight: float = 0.55,
    job_keywords: list[str] | None = None,
) -> MatchResult:
    """
    Score a single resume against a job description.

    Combined score = keyword_weight * keyword_overlap + (1 - keyword_weight) * tfidf.
    A resume is greenlit when match_percent >= threshold.
    """
    if not (job_description or "").strip():
        return MatchResult(
            filename=filename,
            match_percent=0.0,
            greenlit=False,
            keyword_overlap_percent=0.0,
            semantic_similarity_percent=0.0,
            error="Job description is empty.",
        )

    if not (resume_text or "").strip():
        return MatchResult(
            filename=filename,
            match_percent=0.0,
            greenlit=False,
            keyword_overlap_percent=0.0,
            semantic_similarity_percent=0.0,
            error="Could not extract text from this resume.",
        )

    keywords = job_keywords if job_keywords is not None else extract_keywords(job_description)
    overlap, matched, missing = _keyword_overlap(keywords, resume_text)
    semantic = _tfidf_similarity(job_description, resume_text)

    weight = float(np.clip(keyword_weight, 0.0, 1.0))
    combined = weight * overlap + (1.0 - weight) * semantic
    combined = round(float(combined), 1)

    return MatchResult(
        filename=filename,
        match_percent=combined,
        greenlit=combined >= threshold,
        keyword_overlap_percent=round(overlap, 1),
        semantic_similarity_percent=round(semantic, 1),
        matched_keywords=matched,
        missing_keywords=missing,
    )


def score_resumes(
    job_description: str,
    resumes: list[tuple[str, str]],
    threshold: float = 50.0,
    keyword_weight: float = 0.55,
) -> list[MatchResult]:
    """
    Score many resumes. ``resumes`` is a list of (filename, text) pairs.
    Results are sorted by match_percent descending.
    """
    keywords = extract_keywords(job_description)
    results = [
        score_resume(
            job_description=job_description,
            resume_text=text,
            filename=name,
            threshold=threshold,
            keyword_weight=keyword_weight,
            job_keywords=keywords,
        )
        for name, text in resumes
    ]
    results.sort(key=lambda r: (r.greenlit, r.match_percent), reverse=True)
    return results