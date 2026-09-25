"""Score resumes against a job description using keywords + TF-IDF similarity.

Both JDs and resumes often lead bullets with stock verbs ("Demonstrate",
"Review", "Responsible for", "Led", …). Matching reads past those lead-ins
so similarity is judged on the substance that follows.
"""

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

# Stock lead-in verbs/words common at the start of JD bullets and resume bullets.
# Matching skips these so "Demonstrated thermal plunger design" aligns with
# "Review thermal plunger design" on the meaningful terms.
_LEAD_IN_WORDS = frozenset(
    """
    demonstrate demonstrates demonstrated demonstrating
    review reviews reviewed reviewing
    approve approves approved approving
    provide provides provided providing
    apply applies applied applying
    perform performs performed performing
    conduct conducts conducted conducting
    manage manages managed managing
    support supports supported supporting
    assist assists assisted assisting
    lead leads led leading
    drive drives drove driven driving
    develop develops developed developing
    design designs designed designing
    build builds built building
    create creates created creating
    implement implements implemented implementing
    deliver delivers delivered delivering
    own owned owning
    ensure ensures ensured ensuring
    maintain maintains maintained maintaining
    coordinate coordinates coordinated coordinating
    collaborate collaborates collaborated collaborating
    work works worked working
    help helps helped helping
    use used using utilize utilized utilizing
    analyze analyzes analyzed analyzing
    evaluate evaluates evaluated evaluating
    identify identifies identified identifying
    establish establishes established establishing
    improve improves improved improving
    optimize optimizes optimized optimizing
    execute executes executed executing
    oversee oversees oversaw overseeing
    handle handles handled handling
    participate participated participating
    contribute contributed contributing
    able willingness willing
    responsible responsibility
    experience experienced
    proven strong excellent solid deep
    ability skills knowledge understanding
    must should required preferred
    successfully highly extensively
    """.split()
)

# Multi-word lead-in phrases stripped from the start of a line (longest first)
_LEAD_IN_PHRASES = tuple(
    sorted(
        [
            "responsible for",
            "responsibility for",
            "demonstrated ability to",
            "proven ability to",
            "ability to",
            "able to",
            "experience with",
            "experience in",
            "experienced in",
            "experienced with",
            "familiar with",
            "knowledge of",
            "understanding of",
            "skilled in",
            "proficient in",
            "expertise in",
            "worked on",
            "worked with",
            "helped with",
            "assisted with",
            "participated in",
            "involved in",
            "in charge of",
            "duties included",
            "tasks included",
            "successfully",
        ],
        key=len,
        reverse=True,
    )
)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}")
_BULLET_SPLIT_RE = re.compile(r"[\n\r]+|(?<=[.!;])\s+(?=[A-Z])")


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


def strip_lead_ins(text: str, max_words: int = 5) -> str:
    """
    Remove up to ``max_words`` leading stock verbs/fillers from a line.

    Examples:
      "Demonstrated root cause analysis on ATE failures"
        → "root cause analysis on ATE failures"
      "Responsible for reviewing test socket designs"
        → "test socket designs"
      "Review and approve test socket designs"
        → "test socket designs"
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    # Drop bullet markers
    cleaned = re.sub(r"^\s*(?:[-*•●▪◦]|\d+[.)])\s*", "", cleaned)
    lower = cleaned.lower()

    # Strip known multi-word phrases repeatedly from the front
    changed = True
    while changed:
        changed = False
        for phrase in _LEAD_IN_PHRASES:
            if lower.startswith(phrase):
                cleaned = cleaned[len(phrase) :].lstrip(" :,-")
                lower = cleaned.lower()
                changed = True
                break

    # Strip up to max_words single lead-in tokens (and joining words)
    words = cleaned.split()
    skipped = 0
    while words and skipped < max_words:
        bare = re.sub(r"[^a-zA-Z0-9+#.\-]", "", words[0]).lower()
        if bare in _LEAD_IN_WORDS or bare in {"and", "or", "to", "of", "for", "the", "a", "an"}:
            words.pop(0)
            skipped += 1
            continue
        break

    return " ".join(words).strip(" :,-")


def normalize_for_matching(text: str) -> str:
    """Normalize a full JD or resume by stripping lead-ins on each line/bullet."""
    if not (text or "").strip():
        return ""
    parts: list[str] = []
    for raw in _BULLET_SPLIT_RE.split(text):
        line = raw.strip()
        if not line:
            continue
        stripped = strip_lead_ins(line)
        parts.append(stripped if stripped else line)
    return "\n".join(parts)


def extract_keywords(text: str, top_n: int = 40) -> list[str]:
    """Extract important keywords after stripping stock lead-in verbs."""
    normalized = normalize_for_matching(text)
    tokens = [
        t
        for t in _tokenize(normalized)
        if t not in _STOPWORDS and t not in _LEAD_IN_WORDS and len(t) > 1
    ]
    if not tokens:
        return []

    freq: dict[str, int] = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1

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

    # Match against lead-in-stripped resume text so "Demonstrated X" still hits X
    resume_tokens = set(_tokenize(normalize_for_matching(resume_text)))
    resume_tokens |= set(_tokenize(resume_text))

    matched: list[str] = []
    missing: list[str] = []
    for kw in job_keywords:
        if kw in _LEAD_IN_WORDS:
            continue
        if kw in resume_tokens or any(kw in token for token in resume_tokens):
            matched.append(kw)
        else:
            missing.append(kw)

    denom = len(matched) + len(missing)
    overlap = (len(matched) / denom) * 100.0 if denom else 0.0
    return overlap, matched, missing


def _tfidf_similarity(job_text: str, resume_text: str) -> float:
    """Cosine similarity after stripping stock lead-in verbs on both sides."""
    job_norm = normalize_for_matching(job_text)
    resume_norm = normalize_for_matching(resume_text)
    if not job_norm.strip() or not resume_norm.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b",
    )
    try:
        matrix = vectorizer.fit_transform([job_norm, resume_norm])
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
    Lead-in verbs ("Demonstrate", "Review", "Responsible for", …) are stripped
    before keyword extraction and similarity so matching focuses on substance.
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
