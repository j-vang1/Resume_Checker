"""Score resumes against a job description using keywords + TF-IDF similarity.

Both JDs and resumes often lead bullets with stock verbs ("Demonstrate",
"Review", "Responsible for", "Led", …). Matching reads past those lead-ins
so similarity is judged on the substance that follows.

Keywords prefer 2–5 word phrases (e.g. "test socket designs", "root cause
analysis") so matched/missing results show real hiring phrases, not lone
filler words. Strong skill unigrams (Python, ATE, …) still fill remaining slots.
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
            "we are hiring a",
            "we are hiring an",
            "we are hiring",
            "we are looking for a",
            "we are looking for an",
            "we are looking for",
            "looking for a",
            "looking for an",
            "looking for",
            "candidates should know",
            "candidate should know",
            "you will",
            "you will be",
            "the ideal candidate",
            "ideal candidate",
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
    return [t.lower().strip(".-") for t in _TOKEN_RE.findall(text or "") if t.lower().strip(".-")]


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


# Short tech acronyms allowed as keywords even if length < 4
_SHORT_TECH = frozenset(
    {
        "c",
        "c++",
        "c#",
        "r",
        "go",
        "js",
        "ts",
        "aws",
        "gcp",
        "sql",
        "ate",
        "slt",
        "doe",
        "pcb",
        "ic",
        "rf",
        "ml",
        "ai",
        "ui",
        "ux",
        "qa",
        "ci",
        "cd",
        "os",
        "hw",
        "sw",
        "fa",
        "npi",
        "hvm",
        "dft",
        "soc",
        "esi",
        "esd",
        "api",
        "etl",
        "bi",
    }
)


# Generic single words that rarely carry hiring signal alone
_WEAK_UNIGRAMS = frozenset(
    """
    related community services company management materials abilities programs
    reports program process current following force home effectiveness
    orientation evaluations guidelines activities potential licensing essential
    campaigns tracking pipeline meetings internal families records source
    pounds needed requirements mentors network job marketing role team teams
    including based using make made making other others also well good great
    high low new old year years month months day days time times level levels
    type types area areas part parts way ways candidate candidates position
    positions description descriptions
    """.split()
)


def _is_usable_token(token: str) -> bool:
    """True if a single token can appear as content inside a phrase keyword.

    Lead-in verbs are stripped at line starts separately; a small set of
    noun-like lead-in forms (e.g. "designs") may still appear mid-phrase.
    """
    if not token or token in _STOPWORDS:
        return False
    if token in _SHORT_TECH:
        return True
    if len(token) < 3:
        return False
    return True


# Lead-in tokens that are also common nouns and OK mid/end of a phrase
_LEAD_IN_OK_IN_PHRASE = frozenset(
    """
    design designs support management process report reports lead drive
    control controls review reviews build builds
    """.split()
)


# Words that must not appear inside a phrase keyword (fillers / conjunctions)
_PHRASE_BLOCKLIST = frozenset(
    {
        "and",
        "or",
        "with",
        "without",
        "within",
        "across",
        "via",
        "per",
        "vs",
        "versus",
        "including",
        "such",
        "also",
        "etc",
        "hiring",
        "looking",
        "candidates",
        "candidate",
        "know",
        "should",
        "must",
        "please",
        "ideal",
    }
)


def _is_usable_keyword(term: str) -> bool:
    """True if a keyword term (1–5 words) is worth matching on."""
    if not term:
        return False
    parts = [p for p in term.split() if p]
    if not parts or len(parts) > 5:
        return False
    if len(parts) == 1:
        token = parts[0]
        if not _is_usable_token(token):
            return False
        # Lone lead-in verbs / weak fillers are not useful keywords
        if token in _LEAD_IN_WORDS or token in _WEAK_UNIGRAMS or token in _PHRASE_BLOCKLIST:
            return False
        return True
    # Phrases: no stopwords / blockers; don't start on a stock lead-in verb
    if parts[0] in _LEAD_IN_WORDS or parts[0] in _PHRASE_BLOCKLIST:
        return False
    for p in parts:
        if p in _PHRASE_BLOCKLIST or p in _STOPWORDS:
            return False
        if p in _LEAD_IN_WORDS and p not in _LEAD_IN_OK_IN_PHRASE:
            return False
        if not _is_usable_token(p):
            return False
    return len(parts) >= 2


def _phrase_score(phrase: str, count: int) -> float:
    """Prefer solid multi-word phrases; demote unigrams."""
    words = phrase.split()
    n = len(words)
    content = sum(1 for w in words if _is_usable_token(w))
    # Prefer longer phrases so UI shows 3–5 word terms when available
    if n == 5:
        length_bonus = 2.8
    elif n == 4:
        length_bonus = 2.6
    elif n == 3:
        length_bonus = 2.4
    elif n == 2:
        length_bonus = 2.1
    else:
        length_bonus = 0.85
    content_bonus = 1.0 + 0.12 * max(0, content - 1)
    char_bonus = 1.0 + min(len(phrase), 28) / 28.0
    return count * length_bonus * content_bonus * char_bonus


def _iter_phrase_spans(text: str) -> list[list[str]]:
    """
    Tokenize JD into short spans so skill lists don't become one giant n-gram.

    Splits on newlines, commas, semicolons, and " and " / " or " separators.
    """
    normalized = normalize_for_matching(text)
    spans: list[list[str]] = []
    for raw_line in normalized.splitlines():
        chunks = re.split(r"[,;/]|(?:\s+and\s+)|(?:\s+or\s+)", raw_line, flags=re.IGNORECASE)
        for chunk in chunks:
            toks = [t for t in _tokenize(chunk) if t]
            if toks:
                spans.append(toks)
    return spans


def _extract_ngrams_from_span(tokens: list[str], n: int) -> list[str]:
    """Build n-grams that start/end on content words within one span."""
    out: list[str] = []
    if n == 1:
        for t in tokens:
            if t in _SHORT_TECH or (_is_usable_token(t) and t not in _WEAK_UNIGRAMS):
                if _is_usable_keyword(t):
                    out.append(t)
        return out

    for i in range(len(tokens) - n + 1):
        window = tokens[i : i + n]
        if not _is_usable_token(window[0]) or not _is_usable_token(window[-1]):
            continue
        phrase = " ".join(window)
        if _is_usable_keyword(phrase):
            out.append(phrase)
    return out


def extract_keywords(text: str, top_n: int = 40) -> list[str]:
    """
    Extract important keywords after stripping stock lead-in verbs.

    Prefers 2–5 word phrases (e.g. "test socket designs", "root cause analysis")
    so matched/missing UI shows real hiring phrases, not lone filler words.
    Strong unigrams (skills, acronyms) fill remaining slots.
    """
    spans = _iter_phrase_spans(text)
    if not spans:
        return []

    freq: dict[str, int] = {}
    for tokens in spans:
        for n in (5, 4, 3, 2, 1):
            for gram in _extract_ngrams_from_span(tokens, n):
                freq[gram] = freq.get(gram, 0) + 1

    if not freq:
        return []

    scored = sorted(
        freq.items(),
        key=lambda item: (_phrase_score(item[0], item[1]), len(item[0].split()), item[0]),
        reverse=True,
    )

    selected: list[str] = []
    for term, _ in scored:
        # Skip if this term is a sub-phrase of an already selected longer term
        if any(term != other and f" {term} " in f" {other} " for other in selected):
            continue
        # Skip unigrams already covered by a selected phrase
        if " " not in term and any(term in other.split() for other in selected):
            continue
        selected.append(term)
        if len(selected) >= top_n:
            break

    # Drop any leftover shorter phrase fully contained in a longer selected one
    selected = [
        t
        for t in selected
        if not any(t != o and f" {t} " in f" {o} " for o in selected)
    ]
    return selected[:top_n]


def _phrase_in_text(phrase: str, text: str) -> bool:
    """Whole-word match for a multi-word phrase (order preserved)."""
    parts = [p for p in phrase.lower().split() if p]
    if not parts:
        return False
    escaped = r"[\s\-_/]+".join(re.escape(p) for p in parts)
    pattern = re.compile(rf"(?i)(?<![a-z0-9]){escaped}(?![a-z0-9])")
    return bool(pattern.search(text or ""))


def _content_tokens(phrase: str) -> list[str]:
    return [p for p in phrase.lower().split() if _is_usable_token(p)]


def _all_content_in_resume(phrase: str, resume_tokens: set[str], resume_text: str) -> bool:
    """
    Soft phrase match: every content word appears as a whole word in the resume.

    Lets "django postgresql" match a resume that lists those skills separately,
    while still avoiding substring false positives via whole-word checks.
    """
    content = _content_tokens(phrase)
    if len(content) < 2:
        return False
    for token in content:
        if token in resume_tokens:
            continue
        pattern = re.compile(rf"(?i)(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])")
        if pattern.search(resume_text or ""):
            continue
        # Light plural tolerance
        if not token.endswith("s") and (token + "s") in resume_tokens:
            continue
        if token.endswith("s") and len(token) > 3 and token[:-1] in resume_tokens:
            continue
        return False
    return True


def _keyword_in_resume(kw: str, resume_text: str, resume_tokens: set[str]) -> bool:
    """
    True only if the keyword appears as a real whole word/phrase in the resume.

    Avoids false hits like keyword 'ate' matching inside 'evaluate'/'create',
    or 'test' matching inside 'latest'. Multi-word phrases match on exact
    phrase first, then on all content words present (whole-word).
    """
    if not kw:
        return False

    parts = kw.split()
    if len(parts) > 1:
        if _phrase_in_text(kw, resume_text):
            return True
        # Light plural tolerance on the last word: "test socket" vs "test sockets"
        last = parts[-1]
        if not last.endswith("s"):
            alt = " ".join(parts[:-1] + [last + "s"])
            if _phrase_in_text(alt, resume_text):
                return True
        elif last.endswith("s") and len(last) > 3:
            alt = " ".join(parts[:-1] + [last[:-1]])
            if _phrase_in_text(alt, resume_text):
                return True
        # Soft match: all content tokens present as whole words
        if _all_content_in_resume(kw, resume_tokens, resume_text):
            return True
        return False

    if kw in resume_tokens:
        return True

    # Hyphen/slash compounds: "signal-integrity" or token "machine-learning"
    for token in resume_tokens:
        if "-" in token or "/" in token or "." in token:
            compound_parts = re.split(r"[-/.]", token)
            if kw in compound_parts:
                return True

    # Whole-word search in original text
    pattern = re.compile(rf"(?i)(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])")
    if pattern.search(resume_text or ""):
        return True

    # Simple plural: keyword "socket" vs resume "sockets"
    if not kw.endswith("s"):
        plural = kw + "s"
        if plural in resume_tokens:
            return True
        if re.compile(rf"(?i)(?<![a-z0-9]){re.escape(plural)}(?![a-z0-9])").search(
            resume_text or ""
        ):
            return True
    elif kw.endswith("s") and len(kw) > 3:
        singular = kw[:-1]
        if singular in resume_tokens:
            return True

    return False


def _keyword_overlap(
    job_keywords: list[str], resume_text: str
) -> tuple[float, list[str], list[str]]:
    """Return overlap %, matched keywords, and missing keywords."""
    if not job_keywords:
        return 0.0, [], []

    normalized_resume = normalize_for_matching(resume_text)
    resume_tokens = set(_tokenize(normalized_resume)) | set(_tokenize(resume_text))

    matched: list[str] = []
    missing: list[str] = []
    for kw in job_keywords:
        if not _is_usable_keyword(kw):
            continue
        if _keyword_in_resume(kw, resume_text, resume_tokens) or _keyword_in_resume(
            kw, normalized_resume, resume_tokens
        ):
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
