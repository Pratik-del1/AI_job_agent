"""Scores jobs with every component and produces the ranked tables."""

from datetime import datetime
from typing import Optional

import pandas as pd

from jobagent.config import Settings
from jobagent.matching.experience import score_experience
from jobagent.matching.preferences import explicit, score_preferences
from jobagent.matching.ranking import SeenJobs, dedupe_jobs, job_key, sort_key
from jobagent.matching.roles import score_role
from jobagent.matching.scoring import (
    JobMatch,
    build_explanation,
    cap_label,
    combine,
    label_for,
)
from jobagent.matching.semantic import (
    Embedder,
    SentenceTransformerEmbedder,
    candidate_chunks,
    semantic_score,
    title_similarity_fn,
)
from jobagent.matching.skills import (
    candidate_skills,
    extract_job_skills,
    score_skills,
)
from jobagent.matching.taxonomy import Taxonomy, load_taxonomy
from jobagent.matching.text import chunk_lines, clean_html, split_sections
from jobagent.resume.preferences import (
    apply_user_preferences,
    load_user_preferences,
)
from jobagent.resume.schema import ResumeProfile

MATCHER_NAME = "hybrid"

# Columns the dashboard and the legacy CSV already use, filled from the
# hybrid scores so existing readers keep working.
LEGACY_ALIASES = {
    "final_score": "overall_score",
    "similarity_score": "semantic_score",
    "skill_match": "skill_score",
    "role_match": "role_score",
    "experience_match": "experience_score",
}

LIST_SEPARATOR = "; "

RECOMMENDED_COLUMNS = [
    "job_id",
    "title",
    "company",
    "location",
    "employment_type",
    "description",
    "similarity_score",
    "skill_match",
    "role_match",
    "experience_match",
    "final_score",
    "recommendation",
    "apply_url",
    "source",
    "is_new",
    "updated_at",
    "overall_score",
    "skill_score",
    "role_score",
    "experience_score",
    "semantic_score",
    "preference_score",
    "matched_skills",
    "missing_required_skills",
    "missing_nice_to_have_skills",
    "missing_other_skills",
    "role_family",
    "role_match_type",
    "experience_required",
    "experience_required_source",
    "experience_candidate",
    "label_note",
    "explanation",
    "matcher",
]

SCORED_COLUMNS = (
    ["rank", "job_key", "in_application_workflow"]
    + [column for column in RECOMMENDED_COLUMNS if column != "description"]
)


def load_profile(settings: Settings) -> ResumeProfile:
    """The structured resume profile, with the user's preferences applied."""

    path = settings.resume_profile_file

    if not path.exists():
        raise FileNotFoundError(
            f"Resume profile not found: {path}\n"
            "Create it with: python -m jobagent.resume.cli"
        )

    profile = ResumeProfile.model_validate_json(
        path.read_text(encoding="utf-8")
    )

    return apply_user_preferences(
        profile,
        load_user_preferences(settings.preferences_file),
    )


def target_roles(profile: ResumeProfile) -> list[str]:
    """Roles the candidate stated or configured; inferred ones only when
    there are none."""

    roles = explicit(profile.preferences.preferred_roles)

    if not roles:
        roles = [
            item.value
            for item in profile.preferences.preferred_roles
            if item.value.strip()
        ]

    return roles


class HybridMatcher:
    """Scores one job at a time against a fixed candidate profile."""

    def __init__(
        self,
        profile: ResumeProfile,
        settings: Settings,
        taxonomy: Optional[Taxonomy] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.profile = profile
        self.settings = settings
        self.taxonomy = taxonomy or load_taxonomy(settings.taxonomy_file)
        self.embedder = embedder

        self.candidate_skills = candidate_skills(profile, self.taxonomy)
        self.target_roles = target_roles(profile)

        self._candidate_vectors = None
        self._title_similarity = None

        if embedder is not None:
            chunks = candidate_chunks(profile)
            if chunks:
                self._candidate_vectors = embedder.encode(chunks)

            self._title_similarity = title_similarity_fn(
                embedder,
                settings.role_similarity_floor,
                settings.role_similarity_ceiling,
            )

    def match(self, job: dict) -> JobMatch:
        settings = self.settings
        title = str(job.get("title") or "")

        job_text = split_sections(
            clean_html(job.get("description", "")),
            self.taxonomy,
        )

        skills = score_skills(
            extract_job_skills(title, job_text, self.taxonomy),
            self.candidate_skills,
            settings.skill_tier_weights,
            settings.match_min_recognised_skills,
        )

        role = score_role(
            title,
            self.target_roles,
            self.taxonomy,
            title_similarity=self._title_similarity,
            fallback_cap=settings.role_fallback_cap,
        )

        experience = score_experience(
            title,
            job_text,
            self.profile.total_years_experience,
            self.taxonomy,
            settings.match_experience_curve,
            settings.match_experience_floor,
        )

        semantic, semantic_raw = None, None
        if self.embedder is not None:
            semantic, semantic_raw = semantic_score(
                self._candidate_vectors,
                chunk_lines(
                    [title, *job_text.relevant_lines],
                    settings.semantic_chunk_words,
                    settings.semantic_chunk_overlap,
                ),
                self.embedder,
                settings.semantic_top_k,
                settings.semantic_similarity_floor,
                settings.semantic_similarity_ceiling,
            )

        preference = score_preferences(
            job,
            self.profile.preferences,
            self.taxonomy,
        )

        scores = {
            "skill": skills.score,
            "role": role.score,
            "experience": experience.score,
            "semantic": semantic,
            "preference": preference.score,
        }

        overall, weights_used = combine(scores, settings.match_weights)

        label = label_for(
            overall,
            settings.match_label_high_priority,
            settings.match_label_good_match,
            settings.match_label_consider,
        )
        label, label_note = self._apply_caps(label, experience, preference)

        match = JobMatch(
            overall_score=overall,
            skill_score=skills.score,
            role_score=role.score,
            experience_score=experience.score,
            semantic_score=semantic,
            preference_score=preference.score,
            matched_skills=skills.matched,
            missing_required_skills=skills.missing_required,
            missing_nice_to_have_skills=skills.missing_nice_to_have,
            missing_other_skills=skills.missing_other,
            role_family=role.family,
            role_match_type=role.match_type,
            experience_required=experience.required_years,
            experience_required_source=experience.required_source,
            experience_candidate=experience.candidate_years,
            recommendation=label,
            label_note=label_note,
            weights_used=weights_used,
        )

        match.explanation = build_explanation(
            match,
            self._notes(skills, role, experience, semantic_raw, preference),
        )

        return match

    def _apply_caps(self, label, experience, preference):
        """Configurable label caps. The overall score is never changed."""

        settings = self.settings
        notes = []

        gap = experience.gap
        cap_gap = settings.match_experience_cap_gap_years
        if cap_gap is not None and gap is not None and gap >= cap_gap:
            capped = cap_label(label, settings.match_experience_cap_label)
            if capped != label:
                notes.append(
                    f"Capped from {label} because the role asks for about "
                    f"{gap:g} more years of experience than the resume shows."
                )
                label = capped

        if preference.excluded_role:
            capped = cap_label(label, settings.match_excluded_role_cap_label)
            if capped != label:
                notes.append(
                    f"Capped from {label} because the role is on the "
                    "excluded-roles list."
                )
                label = capped

        return label, " ".join(notes) or None

    def _notes(self, skills, role, experience, semantic_raw, preference):
        notes = {}

        if skills.score is not None:
            notes["skill"] = (
                f"{len(skills.matched)} of {skills.recognised} "
                "recognised skills"
            )
        else:
            notes["skill"] = (
                f"only {skills.recognised} recognised skills in the posting"
            )

        if role.score is not None:
            family = self.taxonomy.family_labels.get(role.family, role.family)
            if role.match_type == "semantic":
                notes["role"] = "title similarity; role family not recognised"
            elif role.target:
                notes["role"] = (
                    f"{family}, {role.match_type} match to {role.target}"
                )
        elif not self.target_roles:
            notes["role"] = "no preferred roles in the profile"

        if experience.required_years is not None:
            source = (
                "stated"
                if experience.required_source == "stated"
                else f"implied by '{experience.seniority}' in the title"
            )
            candidate = (
                "unknown"
                if experience.candidate_years is None
                else f"{experience.candidate_years:g}"
            )
            notes["experience"] = (
                f"asks {experience.required_years:g}+ years ({source}); "
                f"resume shows {candidate}"
            )
        else:
            notes["experience"] = "posting gives no years or seniority"

        if semantic_raw is not None:
            notes["semantic"] = f"raw similarity {semantic_raw:.2f}"

        if preference.details:
            notes["preference"] = ", ".join(
                f"{name.replace('_', ' ')}: {'yes' if ok else 'no'}"
                for name, ok in preference.details.items()
            )
        else:
            notes["preference"] = "no explicit preferences configured"

        return notes


def build_embedder(settings: Settings) -> Embedder:
    return SentenceTransformerEmbedder(settings.semantic_model)


def score_jobs(
    jobs: list[dict],
    matcher: HybridMatcher,
    seen: Optional[SeenJobs] = None,
    in_workflow_ids=frozenset(),
    updated_at: Optional[str] = None,
) -> pd.DataFrame:
    """Every job scored and ranked, best match first.

    Duplicate records of one posting are dropped. Jobs already in the
    application workflow are scored too and flagged, not removed.
    """

    settings = matcher.settings
    updated_at = updated_at or datetime.now().isoformat(timespec="seconds")

    unique, _ = dedupe_jobs(jobs)

    rows = []
    for job in unique:
        key = job_key(job)
        match = matcher.match(job)

        row = {
            "job_key": key,
            "job_id": str(job.get("job_id") or ""),
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "location": job.get("location", ""),
            "employment_type": job.get("employment_type", ""),
            "description": job.get("description", ""),
            "apply_url": job.get("apply_url", ""),
            "source": job.get("source", ""),
            "is_new": seen.is_new(key) if seen is not None else False,
            "updated_at": updated_at,
            "in_application_workflow": (
                str(job.get("job_id") or "") in in_workflow_ids
            ),
            "matcher": MATCHER_NAME,
        }

        data = match.model_dump(exclude={"weights_used"})
        for name, value in data.items():
            if isinstance(value, list):
                value = LIST_SEPARATOR.join(value)
            row[name] = value

        for legacy, hybrid in LEGACY_ALIASES.items():
            row[legacy] = row[hybrid]

        rows.append(row)

    rows.sort(
        key=lambda row: sort_key(
            row["overall_score"],
            row["is_new"],
            row["job_key"],
            settings.match_tie_tolerance,
        )
    )

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return pd.DataFrame(
        rows,
        columns=["rank", "job_key", "in_application_workflow"]
        + RECOMMENDED_COLUMNS,
    )


def top_recommendations(scored: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """The best ``top_n`` jobs that are not already being applied to."""

    available = scored[~scored["in_application_workflow"]]

    return (
        available
        .head(top_n)[RECOMMENDED_COLUMNS]
        .reset_index(drop=True)
    )


def run_hybrid(
    jobs: list[dict],
    settings: Settings,
    in_workflow_ids=frozenset(),
    embedder: Optional[Embedder] = None,
    seen: Optional[SeenJobs] = None,
    updated_at: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score all jobs, write ``scored_jobs.csv`` and return
    (all scored jobs, top recommendations)."""

    matcher = HybridMatcher(
        load_profile(settings),
        settings,
        embedder=embedder or build_embedder(settings),
    )

    scored = score_jobs(
        jobs,
        matcher,
        seen=seen,
        in_workflow_ids=in_workflow_ids,
        updated_at=updated_at,
    )

    settings.scored_jobs_file.parent.mkdir(parents=True, exist_ok=True)
    scored[SCORED_COLUMNS].to_csv(
        settings.scored_jobs_file,
        index=False,
    )

    return scored, top_recommendations(scored, settings.top_jobs)
