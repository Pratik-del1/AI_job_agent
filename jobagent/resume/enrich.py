"""Deterministic additions to an extraction, applied after the LLM."""

import re

from jobagent.resume.schema import ResumeExtraction, dedupe


def _mentions(name: str, text: str) -> bool:
    # Whole-term match: "Git" must not match "GitHub", nor "SQL" "MySQL".
    pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _entry_text(*parts) -> str:
    chunks = []
    for part in parts:
        if isinstance(part, list):
            chunks.extend(part)
        elif part:
            chunks.append(part)
    return "\n".join(chunks)


def add_usage_evidence(extraction: ResumeExtraction) -> ResumeExtraction:
    """Point each skill at the roles and projects whose text names it.

    The LLM is asked for this too; this pass guarantees the literal
    mentions. Synonyms ("EDA" vs "exploratory data analysis") are left to
    the LLM.
    """

    entries = []

    for item in extraction.experience:
        label = item.company or item.title
        if label:
            entries.append((
                label,
                _entry_text(
                    item.highlights,
                    item.technologies,
                    item.outcomes,
                ),
            ))

    for project in extraction.projects:
        if project.name:
            entries.append((
                project.name,
                _entry_text(
                    project.description,
                    project.highlights,
                    project.technologies,
                    project.outcomes,
                ),
            ))

    if not entries:
        return extraction

    skills = [
        skill.model_copy(
            update={
                "evidence": dedupe(
                    skill.evidence
                    + [
                        label
                        for label, text in entries
                        if _mentions(skill.name, text)
                    ]
                )
            }
        )
        for skill in extraction.skills
    ]

    return extraction.model_copy(
        update={"skills": skills}
    )
