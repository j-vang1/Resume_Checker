"""Domain concept graph: synonyms, related activities, and transferable skills.

Used to match meaning rather than exact job-description wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical concept → related terms / synonyms / activities / abbreviations
CONCEPT_GRAPH: dict[str, set[str]] = {
    "root_cause_analysis": {
        "root cause",
        "rca",
        "root-cause",
        "investigated",
        "investigate",
        "isolated",
        "isolate",
        "debugged",
        "debug",
        "troubleshooting",
        "troubleshoot",
        "failure analysis",
        "diagnosed",
        "diagnose",
        "fault isolation",
        "intermittent",
        "narrowed down",
        "identified cause",
        "5-why",
        "fishbone",
        "corrective action",
        "capa",
    },
    "hardware_validation": {
        "hardware validation",
        "validation",
        "qualification",
        "qualify",
        "qualified",
        "verification",
        "verify",
        "characterization",
        "characterize",
        "acceptance testing",
        "acceptance test",
        "hw validation",
        "bring-up",
        "bringup",
        "commissioning",
        "signoff",
        "sign-off",
        "test plan",
        "validation plan",
        "qualification criteria",
    },
    "failure_analysis": {
        "failure analysis",
        "fa",
        "failed",
        "failure",
        "failures",
        "defect",
        "defects",
        "yield",
        "reliability",
        "tear-down",
        "teardown",
        "post-mortem",
        "postmortem",
        "fault",
        "broken",
        "continuity failure",
        "ecid",
    },
    "doe": {
        "doe",
        "design of experiments",
        "design of experiment",
        "factorial",
        "response surface",
        "parameter sweep",
        "sweep",
        "isolation experiment",
        "experimental design",
        "matrix experiment",
        "anova",
        "taguchi",
    },
    "cross_functional": {
        "cross-functional",
        "cross functional",
        "crossfunctional",
        "coordinated",
        "coordinate",
        "collaborated",
        "collaborate",
        "partnered",
        "liaised",
        "stakeholder",
        "stakeholders",
        "engineering teams",
        "multi-disciplinary",
        "multidisciplinary",
        "worked with",
        "aligned with",
    },
    "supplier_management": {
        "supplier",
        "suppliers",
        "vendor",
        "vendors",
        "oem",
        "corrective actions with suppliers",
        "supplier quality",
        "sourcing",
        "procurement",
        "vendor management",
        "supplier coordination",
    },
    "hardware_debug": {
        "hardware debug",
        "hw debug",
        "debugged",
        "debug",
        "debugging",
        "board bring-up",
        "bring-up",
        "scope",
        "oscilloscope",
        "logic analyzer",
        "probe",
        "schematic",
        "signal integrity",
        "power integrity",
        "continuity",
        "socket",
        "plunger",
        "thermal interface",
        "ate",
        "tester",
    },
    "test_development": {
        "test development",
        "test program",
        "test software",
        "ate",
        "automated test",
        "test automation",
        "test fixture",
        "fixture",
        "test hardware",
        "test methodology",
        "test coverage",
        "pattern generation",
    },
    "thermal_engineering": {
        "thermal",
        "thermal interface",
        "heat sink",
        "heatsink",
        "cooling",
        "dew point",
        "purge",
        "temperature",
        "thermal cycling",
        "-40",
        "thermal stability",
        "tim",
        "thermal design",
    },
    "mechanical_design": {
        "mechanical design",
        "cad",
        "solidworks",
        "creo",
        "nx",
        "dfm",
        "dfa",
        "tolerance",
        "gd&t",
        "gdt",
        "fixture design",
        "mechanical",
        "machining",
        "prototype",
    },
    "python": {
        "python",
        "pandas",
        "numpy",
        "scipy",
        "pytest",
        "fastapi",
        "django",
        "flask",
        "scripting",
        "automation script",
        "jupyter",
    },
    "data_analysis": {
        "data analysis",
        "analyzed",
        "analysis",
        "statistical",
        "statistics",
        "spc",
        "jmp",
        "minitab",
        "matlab",
        "excel",
        "dashboard",
        "metrics",
        "trend analysis",
    },
    "automation": {
        "automation",
        "automated",
        "automate",
        "scripted",
        "pipeline",
        "ci/cd",
        "orchestration",
        "rpa",
        "workflow automation",
    },
    "sql": {
        "sql",
        "postgres",
        "postgresql",
        "mysql",
        "sqlite",
        "query",
        "queries",
        "database",
        "etl",
    },
    "machine_learning": {
        "machine learning",
        "ml",
        "deep learning",
        "neural network",
        "model training",
        "scikit-learn",
        "sklearn",
        "tensorflow",
        "pytorch",
        "classification",
        "regression model",
        "feature engineering",
    },
    "cloud": {
        "aws",
        "azure",
        "gcp",
        "cloud",
        "kubernetes",
        "k8s",
        "docker",
        "terraform",
        "ec2",
        "s3",
        "lambda",
    },
    "rest_apis": {
        "rest",
        "api",
        "apis",
        "http",
        "graphql",
        "endpoint",
        "microservices",
        "openapi",
        "swagger",
    },
    "leadership": {
        "led",
        "lead",
        "mentored",
        "mentor",
        "coached",
        "managed team",
        "tech lead",
        "technical lead",
        "directed",
        "supervised",
    },
    "architecture": {
        "architected",
        "architecture",
        "system design",
        "designed system",
        "technical direction",
        "standards",
        "framework",
        "platform",
    },
    "process_improvement": {
        "process improvement",
        "lean",
        "six sigma",
        "kaizen",
        "reduced",
        "improved",
        "optimized",
        "optimization",
        "efficiency",
        "throughput",
        "cycle time",
    },
    "customer_support": {
        "customer",
        "client",
        "field support",
        "on-site",
        "escalation",
        "customer issue",
        "application support",
        "technical support",
    },
    "semiconductor_test": {
        "semiconductor",
        "ate",
        "handler",
        "probe card",
        "socket",
        "wafer",
        "die",
        "package test",
        "final test",
        "sort",
        "ecid",
        "tester",
    },
}

# Importance keywords that hint at requirement weight in JD text
CORE_HINTS = (
    "required",
    "must have",
    "must-have",
    "essential",
    "core",
    "minimum qualifications",
    "you will",
    "responsibilities",
    "primary",
    "key responsibility",
)
IMPORTANT_HINTS = (
    "strong experience",
    "experience with",
    "proficient",
    "hands-on",
    "demonstrated",
    "solid",
)
PREFERRED_HINTS = (
    "preferred",
    "nice to have",
    "nice-to-have",
    "bonus",
    "plus",
    "ideal",
    "desired",
    "a plus",
)
GENERIC_PHRASES = (
    "equal opportunity",
    "EOE",
    "competitive salary",
    "benefits package",
    "work-life balance",
    "fast-paced environment",
    "self-starter",
    "team player",
    "excellent communication",
    "detail-oriented",
    "passionate about",
    "we are looking for",
    "join our team",
    "unlimited pto",
    "diversity and inclusion",
    "background check",
    "authorized to work",
)


@dataclass
class ConceptMatch:
    concept_id: str
    label: str
    matched_terms: list[str] = field(default_factory=list)
    score: float = 0.0


def concept_label(concept_id: str) -> str:
    return concept_id.replace("_", " ").title()


def expand_query(text: str) -> set[str]:
    """Return tokens from text plus related concept terms that hit."""
    lower = (text or "").lower()
    expanded: set[str] = set()
    for concept_id, terms in CONCEPT_GRAPH.items():
        hits = [t for t in terms if t in lower]
        if hits or any(tok in lower for tok in concept_id.split("_")):
            expanded.update(terms)
            expanded.add(concept_id.replace("_", " "))
    # Always include original tokens of length > 2
    for token in lower.replace("/", " ").replace("-", " ").split():
        cleaned = "".join(ch for ch in token if ch.isalnum() or ch in "+#.")
        if len(cleaned) > 2:
            expanded.add(cleaned)
    return expanded


def find_concepts_in_text(text: str) -> list[ConceptMatch]:
    """Detect which concepts are evidenced in text."""
    lower = (text or "").lower()
    matches: list[ConceptMatch] = []
    for concept_id, terms in CONCEPT_GRAPH.items():
        hit_terms = sorted({t for t in terms if t in lower}, key=len, reverse=True)
        # Prefer multi-word / distinctive hits
        if not hit_terms:
            continue
        # Score by longest hit and count
        best_len = max(len(t) for t in hit_terms)
        score = min(1.0, 0.35 + 0.1 * len(hit_terms) + 0.02 * best_len)
        matches.append(
            ConceptMatch(
                concept_id=concept_id,
                label=concept_label(concept_id),
                matched_terms=hit_terms[:8],
                score=score,
            )
        )
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def related_concepts_for_requirement(requirement: str) -> list[str]:
    """Return human-readable related concept labels for a requirement phrase."""
    lower = (requirement or "").lower()
    related: list[str] = []
    seen: set[str] = set()
    for concept_id, terms in CONCEPT_GRAPH.items():
        if any(t in lower for t in terms) or any(
            part and part in lower for part in concept_id.split("_")
        ):
            for term in sorted(terms, key=len, reverse=True)[:6]:
                if term not in seen and term not in lower:
                    seen.add(term)
                    related.append(term)
            label = concept_label(concept_id)
            if label.lower() not in seen:
                seen.add(label.lower())
                related.insert(0, label)
    return related[:12]
