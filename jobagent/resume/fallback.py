"""Regex resume parsing, used when no LLM is available or the LLM call fails.

Mirrors the rules in ``automation/job_updater.py`` without importing it, so
this package does not load the embedding model. It fills contact details and
skills; experience, education and projects stay as raw section text.
"""

import re

from jobagent.resume.schema import ContactInfo, ResumeExtraction, Skill

# Heading as written in a resume -> canonical section name.
SECTION_HEADINGS = {
    "summary": "summary",
    "profile": "summary",
    "objective": "summary",
    "skills": "skills",
    "technical skills": "skills",
    "experience": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "education": "education",
    "projects": "projects",
    "certifications": "certifications",
    "achievements": "achievements",
}

_HEADING_PATTERN = re.compile(
    r"(?im)^[ \t]*("
    + "|".join(
        re.escape(heading)
        for heading in sorted(SECTION_HEADINGS, key=len, reverse=True)
    )
    + r")[ \t]*:?[ \t]*$"
)


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}

    matches = list(_HEADING_PATTERN.finditer(text))

    for i, match in enumerate(matches):
        name = SECTION_HEADINGS[match.group(1).lower()]
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )
        body = text[match.end():end].strip()

        if not body:
            continue

        if name in sections:
            body = sections[name] + "\n\n" + body
        sections[name] = body

    return sections


def extract_contact(text: str) -> ContactInfo:
    def find(pattern, group=0, flags=0):
        match = re.search(pattern, text, flags)
        return match.group(group).strip() if match else None

    stripped = text.strip()

    return ContactInfo(
        name=stripped.split("\n")[0] if stripped else None,
        email=find(r"[\w\.-]+@[\w\.-]+\.\w+"),
        phone=find(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
        linkedin=find(r"LinkedIn:\s*([^\n|]+)", 1, re.IGNORECASE),
        github=find(r"Github:\s*([^\n|]+)", 1, re.IGNORECASE),
    )


def extract_skills(skills_text: str) -> list[Skill]:
    skills = []

    for line in skills_text.split("\n"):
        # "Languages: Python, SQL" -> "Python, SQL"
        if ":" in line:
            _, line = line.split(":", 1)

        for name in line.split(","):
            name = name.strip()
            if name:
                skills.append(
                    Skill(
                        name=name,
                        evidence=["skills section"],
                    )
                )

    return skills


def fallback_extraction(text: str) -> ResumeExtraction:
    sections = split_sections(text)

    return ResumeExtraction(
        contact=extract_contact(text),
        summary=sections.get("summary"),
        skills=extract_skills(
            sections.get("skills", "")
        ),
    )
