"""Loads ``resources/taxonomy.json`` and compiles its whole-term matchers.

The file is data: skills, role families, seniority terms, location groups
and section-heading patterns can all be extended without touching code.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

DEFAULT_TAXONOMY_FILE = Path(__file__).parent / "resources" / "taxonomy.json"

# Characters that make a neighbouring match part of a longer token:
# "rag" in "leverages", "sql" in "mysql", "c" in "c++".
_TOKEN_CHARS = r"A-Za-z0-9_+#"


def whole_term_pattern(terms) -> re.Pattern:
    """One case-insensitive regex matching any of ``terms`` as a whole term."""

    alternatives = sorted(
        {
            r"\s+".join(re.escape(part) for part in term.split())
            for term in terms
            if term.strip()
        },
        key=len,
        reverse=True,
    )

    # A dot directly before also blocks a match, so "js" is not found
    # inside "Node.js".
    return re.compile(
        rf"(?<![{_TOKEN_CHARS}.])(?:{'|'.join(alternatives)})(?![{_TOKEN_CHARS}])",
        re.IGNORECASE,
    )


@dataclass(frozen=True)
class SkillEntry:
    name: str
    type: str
    pattern: re.Pattern


@dataclass(frozen=True)
class RoleFamily:
    id: str
    label: str
    pattern: re.Pattern


@dataclass(frozen=True)
class SeniorityLevel:
    name: str
    implied_years: float
    pattern: re.Pattern


@dataclass
class Taxonomy:
    skills: list[SkillEntry]
    role_families: list[RoleFamily]
    affinity: dict[tuple[str, str], float]
    default_affinity: float
    seniority_levels: list[SeniorityLevel]
    location_groups: dict[str, list[str]]
    section_headings: dict[str, list[re.Pattern]]
    inline_nice_to_have: Optional[re.Pattern]
    skill_types: dict[str, str] = field(default_factory=dict)
    family_labels: dict[str, str] = field(default_factory=dict)

    def find_skills(self, text: str) -> set[str]:
        """Canonical names of every skill mentioned in ``text``."""

        if not text:
            return set()

        return {
            skill.name
            for skill in self.skills
            if skill.pattern.search(text)
        }

    def role_affinity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        return self.affinity.get(
            (a, b),
            self.affinity.get((b, a), self.default_affinity),
        )


def load_taxonomy(path: Optional[Path] = None) -> Taxonomy:
    path = Path(path) if path else DEFAULT_TAXONOMY_FILE

    data = json.loads(path.read_text(encoding="utf-8"))

    # Soft skills are too vague in postings to score or report as gaps.
    unscored = set(data.get("unscored_skill_types", []))

    skills = [
        SkillEntry(
            name=entry["name"],
            type=entry.get("type", "skill"),
            pattern=whole_term_pattern(
                [entry["name"], *entry.get("aliases", [])]
            ),
        )
        for entry in data["skills"]
        if entry.get("type", "skill") not in unscored
    ]

    names = [skill.name.lower() for skill in skills]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"Duplicate skills in {path.name}: {sorted(duplicates)}"
        )

    families = [
        RoleFamily(
            id=entry["id"],
            label=entry.get("label", entry["id"]),
            pattern=whole_term_pattern(entry["aliases"]),
        )
        for entry in data["role_families"]
    ]

    family_ids = {family.id for family in families}
    affinity = {}
    for a, b, value in data.get("role_affinity", []):
        if a not in family_ids or b not in family_ids:
            raise ValueError(
                f"Unknown role family in role_affinity: {a!r}, {b!r}"
            )
        affinity[(a, b)] = float(value)

    inline = data.get("inline_nice_to_have") or []

    return Taxonomy(
        skills=skills,
        role_families=families,
        affinity=affinity,
        default_affinity=float(data.get("default_role_affinity", 0.0)),
        seniority_levels=[
            SeniorityLevel(
                name=entry["name"],
                implied_years=float(entry["implied_years"]),
                pattern=whole_term_pattern(entry["terms"]),
            )
            for entry in data.get("seniority_levels", [])
        ],
        location_groups={
            key.lower(): [value.lower() for value in values]
            for key, values in data.get("location_groups", {}).items()
        },
        section_headings={
            kind: [
                re.compile(pattern, re.IGNORECASE)
                for pattern in patterns
            ]
            for kind, patterns in data.get("section_headings", {}).items()
        },
        inline_nice_to_have=(
            whole_term_pattern(inline) if inline else None
        ),
        skill_types={skill.name: skill.type for skill in skills},
        family_labels={family.id: family.label for family in families},
    )


@lru_cache
def get_taxonomy(path: Optional[str] = None) -> Taxonomy:
    return load_taxonomy(Path(path) if path else None)
