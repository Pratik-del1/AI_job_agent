"""Candidate preferences: location, work mode, employment type, exclusions.

Only preferences the candidate stated or configured count. A preference the
resume parser inferred, or the candidate's contact location, is never used
here, so with nothing configured the whole component is unavailable.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from jobagent.matching.roles import title_families
from jobagent.matching.taxonomy import Taxonomy, whole_term_pattern
from jobagent.resume.schema import JobPreferences, Preference

EXPLICIT_SOURCES = {"stated", "user"}

_WORK_MODES = {
    "remote": whole_term_pattern(["remote", "work from home", "wfh"]),
    "hybrid": whole_term_pattern(["hybrid"]),
    "onsite": whole_term_pattern(
        ["onsite", "on-site", "on site", "in-office", "in office"]
    ),
}


def explicit(preferences: list[Preference]) -> list[str]:
    return [
        item.value.strip()
        for item in preferences
        if item.source in EXPLICIT_SOURCES and item.value.strip()
    ]


def _normalize(value: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(value or "").lower()).strip()


def job_work_mode(job: dict) -> Optional[str]:
    text = f"{job.get('title', '')} {job.get('location', '')}"

    for mode, pattern in _WORK_MODES.items():
        if pattern.search(text):
            return mode

    return None


def location_matches(
    preferred: list[str],
    job: dict,
    taxonomy: Taxonomy,
) -> bool:
    location = str(job.get("location", "")).lower()

    for value in preferred:
        value = value.lower()

        if value == "remote":
            if job_work_mode(job) == "remote":
                return True
            continue

        # "India" also accepts the cities grouped under it.
        terms = [value, *taxonomy.location_groups.get(value, [])]
        if whole_term_pattern(terms).search(location):
            return True

    return False


@dataclass
class PreferenceResult:
    score: Optional[float]
    details: dict[str, bool] = field(default_factory=dict)
    excluded_role: bool = False


def score_preferences(
    job: dict,
    preferences: JobPreferences,
    taxonomy: Taxonomy,
) -> PreferenceResult:
    """Mean of the preference checks that can be evaluated for this job."""

    details: dict[str, bool] = {}

    locations = explicit(preferences.preferred_locations)
    if locations and str(job.get("location", "")).strip():
        details["location"] = location_matches(locations, job, taxonomy)

    modes = {_normalize(value) for value in explicit(preferences.work_modes)}
    mode = job_work_mode(job)
    if modes and mode:
        details["work_mode"] = mode in {
            value.replace(" ", "") for value in modes
        }

    types = {
        _normalize(value)
        for value in explicit(preferences.employment_types)
    }
    job_type = _normalize(job.get("employment_type", ""))
    if types and job_type:
        details["employment_type"] = job_type in types

    excluded = explicit(preferences.excluded_roles)
    excluded_role = False
    if excluded:
        title = str(job.get("title", ""))
        job_families = set(title_families(title, taxonomy))

        for value in excluded:
            if whole_term_pattern([value]).search(title):
                excluded_role = True
            elif job_families & set(title_families(value, taxonomy)):
                excluded_role = True

        details["not_excluded_role"] = not excluded_role

    score = None
    if details:
        score = sum(details.values()) / len(details)

    return PreferenceResult(
        score=score,
        details=details,
        excluded_role=excluded_role,
    )
