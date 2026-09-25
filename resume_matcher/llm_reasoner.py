"""Optional LLM refinement when OPENAI_API_KEY is present.

Embeddings do retrieval/ranking. If an API key exists, this module can
re-judge evidence strength for core requirements with short structured prompts.
Fails soft — never blocks scoring.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .evidence import EvidenceStrength, RequirementEvidence


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def refine_evidence_with_llm(
    job_description: str,
    evidence_graph: list[RequirementEvidence],
    max_requirements: int = 8,
) -> list[RequirementEvidence]:
    """
    Re-score the top Core/Important requirements using an LLM judge.
    Returns the same graph (possibly with updated strengths/notes) or unchanged on failure.
    """
    if not llm_available() or not evidence_graph:
        return evidence_graph

    try:
        from openai import OpenAI
    except ImportError:
        return evidence_graph

    targets = [
        node
        for node in evidence_graph
        if node.importance in {"Core", "Important"}
    ][:max_requirements]
    if not targets:
        return evidence_graph

    payload = []
    for i, node in enumerate(targets):
        payload.append(
            {
                "id": i,
                "requirement": node.requirement,
                "importance": node.importance,
                "bullets": [e.bullet for e in node.evidence[:3]],
                "current_strength": node.strength.value,
            }
        )

    prompt = f"""You are a technical hiring manager. For each requirement, judge how well the resume bullets prove the candidate can do that work.

Job description context (trimmed):
{job_description[:2500]}

Items (JSON):
{json.dumps(payload, indent=2)}

Return ONLY a JSON array of objects:
[{{"id": 0, "strength": "No Evidence"|"Weak Evidence"|"Moderate Evidence"|"Strong Evidence"|"Very Strong Evidence", "note": "one sentence citing evidence or lack of it"}}]

Rules:
- Do not invent experience not in the bullets.
- Synonyms and related engineering work count (e.g. isolated intermittent failures ≈ root cause analysis).
- Recruiter/marketing/agriculture bullets are not evidence for hardware/engineering requirements.
- Prefer "Moderate" over "Strong" when bullets are only tangential.
"""

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("RESUME_CHECKER_LLM_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or "[]"
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        data: list[dict[str, Any]] = json.loads(match.group(0) if match else raw)
    except Exception:  # noqa: BLE001
        return evidence_graph

    strength_map = {s.value: s for s in EvidenceStrength}
    by_id = {int(item["id"]): item for item in data if "id" in item}
    for i, node in enumerate(targets):
        item = by_id.get(i)
        if not item:
            continue
        strength = strength_map.get(str(item.get("strength", "")), None)
        if strength is None:
            continue
        node.strength = strength
        note = str(item.get("note") or "").strip()
        if note:
            node.notes = note
    return evidence_graph
