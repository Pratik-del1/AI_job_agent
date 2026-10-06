"""Experience matching: years a posting asks for against the candidate's."""

import re
from dataclasses import dataclass
from typing import Optional

from jobagent.matching.taxonomy import Taxonomy
from jobagent.matching.text import NEUTRAL, NICE_TO_HAVE, REQUIRED, JobText

# "3+ years", "3-5 years", "2 to 4 yrs", "minimum 5 years".
_YEARS = re.compile(
    r"(?<![\w.,])(\d{1,2})\s*(?:\+|plus)?\s*"
    r"(?:(?:-|–|to)\s*\d{1,2}\s*\+?\s*)?"
    r"(?:years?|yrs?)\b",
    re.IGNORECASE,
)

MAX_PLAUSIBLE_YEARS = 25


def parse_years_required(text: str) -> Optional[float]:
    """Lower bound of the experience a text asks for.

    With several statements ("3+ years front-end, 2 years services") the
    largest lower bound is the binding one.
    """

    values = [
        int(match.group(1))
        for match in _YEARS.finditer(text or "")
    ]
    values = [value for value in values if value <= MAX_PLAUSIBLE_YEARS]

    return float(max(values)) if values else None


def title_seniority(title: str, taxonomy: Taxonomy):
    """(level name, implied years) for the most senior word in the title."""

    text = re.sub(r"\([^)]*\)", " ", str(title or ""))

    levels = [
        level
        for level in taxonomy.seniority_levels
        if level.pattern.search(text)
    ]

    if not levels:
        return None, None

    level = max(levels, key=lambda item: item.implied_years)
    return level.name, level.implied_years


@dataclass
class ExperienceResult:
    score: Optional[float]
    required_years: Optional[float]
    required_source: Optional[str]
    candidate_years: Optional[float]
    seniority: Optional[str]

    @property
    def gap(self) -> Optional[float]:
        if self.required_years is None or self.candidate_years is None:
            return None
        return self.required_years - self.candidate_years


def score_gap(
    gap: float,
    curve: list[tuple[float, float]],
    floor: float,
) -> float:
    """Score for a shortfall of ``gap`` years; no shortfall scores 1.0."""

    if gap <= 0:
        return 1.0

    for max_gap, score in sorted(curve):
        if gap <= max_gap:
            return score

    return floor


def score_experience(
    title: str,
    job_text: JobText,
    candidate_years: Optional[float],
    taxonomy: Taxonomy,
    curve: list[tuple[float, float]],
    floor: float,
) -> ExperienceResult:
    """Unavailable when the posting gives neither years nor a seniority
    word, or when the candidate's years are unknown."""

    seniority, implied = title_seniority(title, taxonomy)

    required = parse_years_required(
        job_text.text(REQUIRED) + "\n" + job_text.text(NEUTRAL)
    )
    if required is None:
        # Better a figure from a "preferred" line than a guess from the title.
        required = parse_years_required(job_text.text(NICE_TO_HAVE))
    source = "stated" if required is not None else None

    if required is None and implied is not None:
        required, source = implied, "title"

    score = None
    if required is not None and candidate_years is not None:
        score = score_gap(required - candidate_years, curve, floor)

    return ExperienceResult(
        score=score,
        required_years=required,
        required_source=source,
        candidate_years=candidate_years,
        seniority=seniority,
    )
