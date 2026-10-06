"""Deterministic skill matching against the taxonomy."""

from dataclasses import dataclass, field
from typing import Optional

from jobagent.matching.taxonomy import Taxonomy
from jobagent.matching.text import NEUTRAL, NICE_TO_HAVE, REQUIRED, JobText
from jobagent.resume.schema import ResumeProfile

# Strongest first: a skill named in several sections takes the strongest.
TIERS = (REQUIRED, NEUTRAL, NICE_TO_HAVE)


@dataclass
class JobSkills:
    """Skills a posting names, by how binding the section was."""

    by_tier: dict[str, set[str]] = field(
        default_factory=lambda: {tier: set() for tier in TIERS}
    )

    @property
    def all(self) -> set[str]:
        return set().union(*self.by_tier.values())


@dataclass
class SkillResult:
    score: Optional[float]
    matched: list[str]
    missing_required: list[str]
    missing_nice_to_have: list[str]
    missing_other: list[str]
    recognised: int


def candidate_skills(profile: ResumeProfile, taxonomy: Taxonomy) -> set[str]:
    """Canonical skills the candidate has, from skills and technologies.

    Technologies used in a role or project count even when the resume does
    not list them as skills.
    """

    names = [skill.name for skill in profile.skills] + profile.technologies

    found: set[str] = set()
    for name in names:
        found |= taxonomy.find_skills(name)

    return found


def extract_job_skills(
    title: str,
    job_text: JobText,
    taxonomy: Taxonomy,
) -> JobSkills:
    skills = JobSkills()
    taken: set[str] = set()

    for tier in TIERS:
        text = job_text.text(tier)
        if tier == NEUTRAL:
            # A skill in the title ("... Engineer - NLP") is part of the job
            # but is not stated as a requirement.
            text = f"{title}\n{text}"

        found = taxonomy.find_skills(text) - taken
        skills.by_tier[tier] = found
        taken |= found

    return skills


def score_skills(
    job_skills: JobSkills,
    candidate: set[str],
    tier_weights: dict[str, float],
    min_recognised: int,
) -> SkillResult:
    """Weighted coverage of the posting's skills.

    Unavailable (``None``) when the posting names too few recognised skills
    to judge.
    """

    recognised = job_skills.all

    def missing(tier):
        return sorted(job_skills.by_tier[tier] - candidate)

    total = sum(
        tier_weights[tier] * len(job_skills.by_tier[tier])
        for tier in TIERS
    )
    covered = sum(
        tier_weights[tier] * len(job_skills.by_tier[tier] & candidate)
        for tier in TIERS
    )

    score = None
    if len(recognised) >= min_recognised and total > 0:
        score = covered / total

    return SkillResult(
        score=score,
        matched=sorted(recognised & candidate),
        missing_required=missing(REQUIRED),
        missing_nice_to_have=missing(NICE_TO_HAVE),
        missing_other=missing(NEUTRAL),
        recognised=len(recognised),
    )
