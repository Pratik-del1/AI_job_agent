"""Structured resume profile.

``ResumeExtraction`` is the schema the LLM fills in. ``ResumeProfile`` adds
what the code itself contributes: parse metadata, raw section text and the
aggregated technology list.
"""

from datetime import datetime
from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    computed_field,
    model_validator,
)


def _blank_to_none(value):
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _clean_str_list(values):
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return dedupe(
        str(value).strip()
        for value in values
        if value is not None
    )


def dedupe(values) -> list[str]:
    """Drop blanks and case-insensitive duplicates, keeping first-seen order."""

    seen = set()
    result = []

    for value in values:
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            result.append(value)

    return result


OptStr = Annotated[Optional[str], BeforeValidator(_blank_to_none)]

StrList = Annotated[list[str], BeforeValidator(_clean_str_list)]

SkillCategory = Literal[
    "language",
    "framework",
    "ml",
    "data",
    "cloud",
    "tool",
    "soft",
    "other",
]

Seniority = Literal[
    "intern",
    "entry",
    "junior",
    "mid",
    "senior",
    "lead",
    "unknown",
]

PreferenceSource = Literal["stated", "inferred", "user"]


class ContactInfo(BaseModel):
    name: OptStr = None
    email: OptStr = None
    phone: OptStr = None
    location: OptStr = None
    linkedin: OptStr = Field(
        default=None,
        description="LinkedIn URL or handle exactly as written.",
    )
    github: OptStr = Field(
        default=None,
        description="GitHub URL or username exactly as written.",
    )
    portfolio: OptStr = None


class Skill(BaseModel):
    name: str = Field(
        min_length=1,
        description="One skill, e.g. 'Python' or 'LangGraph'.",
    )
    category: SkillCategory = "other"
    evidence: StrList = Field(
        default_factory=list,
        description=(
            "Where the resume shows this skill: 'skills section', plus the "
            "employer name of each role and the name of each project "
            "that names or clearly applies it."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _strip_name(cls, data):
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            data = {**data, "name": data["name"].strip()}
        return data


class Experience(BaseModel):
    company: OptStr = None
    title: OptStr = None
    start_date: OptStr = Field(
        default=None,
        description="As written in the resume, e.g. 'June 2024'.",
    )
    end_date: OptStr = Field(
        default=None,
        description="As written, or 'Present' for a current role.",
    )
    location: OptStr = None
    highlights: StrList = Field(
        default_factory=list,
        description="Every bullet point for this role, as written.",
    )
    technologies: StrList = Field(
        default_factory=list,
        description=(
            "Every tool, library, framework or language this role names, "
            "including ones mentioned only inside a bullet."
        ),
    )
    outcomes: StrList = Field(
        default_factory=list,
        description=(
            "Measured results and scale figures from the bullets, "
            "e.g. '30% faster reporting' or '2M rows processed'."
        ),
    )


class Education(BaseModel):
    institution: OptStr = Field(
        default=None,
        description=(
            "Full name as written, keeping a campus or city that the "
            "resume gives alongside it, e.g. 'Example Institute, Cityname'."
        ),
    )
    location: OptStr = None
    degree: OptStr = None
    field_of_study: OptStr = None
    start_date: OptStr = None
    end_date: OptStr = None
    gpa: OptStr = None


class Project(BaseModel):
    name: OptStr = None
    description: OptStr = Field(
        default=None,
        description="One-sentence summary of what the project is.",
    )
    highlights: StrList = Field(
        default_factory=list,
        description="Every bullet point for this project, as written.",
    )
    technologies: StrList = Field(
        default_factory=list,
        description=(
            "Every tool, library, framework or language this project "
            "names, in its tech-stack line or inside a bullet."
        ),
    )
    links: StrList = Field(default_factory=list)
    outcomes: StrList = Field(
        default_factory=list,
        description=(
            "Measured results and scale figures from the bullets, "
            "e.g. '91% test accuracy', '6-class model', '10,000 records'."
        ),
    )


class Preference(BaseModel):
    value: str = Field(min_length=1)
    source: PreferenceSource = Field(
        default="inferred",
        description=(
            "'stated' if the resume says it explicitly, including a role "
            "the candidate calls themselves in a summary or headline; "
            "'inferred' if deduced from the candidate's background."
        ),
    )


class JobPreferences(BaseModel):
    preferred_roles: list[Preference] = Field(default_factory=list)
    preferred_locations: list[Preference] = Field(default_factory=list)
    work_modes: list[Preference] = Field(
        default_factory=list,
        description="remote, hybrid or onsite.",
    )
    employment_types: list[Preference] = Field(
        default_factory=list,
        description="e.g. full-time, internship, contract.",
    )
    industries: list[Preference] = Field(default_factory=list)
    excluded_roles: list[Preference] = Field(default_factory=list)


class ResumeExtraction(BaseModel):
    """Everything read from the resume text."""

    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: OptStr = None
    skills: list[Skill] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: StrList = Field(default_factory=list)
    total_years_experience: Optional[float] = Field(
        default=None,
        ge=0,
        le=60,
        description=(
            "Years of professional work experience, excluding education. "
            "Null if it cannot be determined."
        ),
    )
    seniority: Seniority = "unknown"
    preferences: JobPreferences = Field(default_factory=JobPreferences)

    @model_validator(mode="after")
    def _dedupe_skills(self):
        seen = {}
        for skill in self.skills:
            key = skill.name.lower()
            if key in seen:
                seen[key].evidence = dedupe(
                    seen[key].evidence + skill.evidence
                )
            else:
                seen[key] = skill
        self.skills = list(seen.values())
        return self


class ParseMetadata(BaseModel):
    method: Literal["llm", "fallback"]
    model: Optional[str] = None
    parsed_at: datetime
    source_file: Optional[str] = None
    fallback_reason: Optional[str] = None


class ResumeProfile(ResumeExtraction):
    """The profile the rest of the system consumes."""

    metadata: ParseMetadata
    raw_sections: dict[str, str] = Field(
        default_factory=dict,
        description="Resume text split by section heading.",
    )

    @computed_field
    @property
    def technologies(self) -> list[str]:
        """Every technology named in skills, experience or projects."""

        names = [
            skill.name
            for skill in self.skills
            if skill.category != "soft"
        ]
        for item in self.experience:
            names.extend(item.technologies)
        for project in self.projects:
            names.extend(project.technologies)

        return dedupe(names)
