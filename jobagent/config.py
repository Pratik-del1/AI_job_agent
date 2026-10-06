"""Settings for the jobagent package, read from the environment or ``.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_prefix="JOBAGENT_",
        extra="ignore",
        populate_by_name=True,
    )

    llm_provider: str = "google"

    # No default: the model name comes from JOBAGENT_LLM_MODEL in .env.
    llm_model: Optional[str] = None

    # Keys are read from their conventional names, without the prefix.
    google_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "JOBAGENT_GOOGLE_API_KEY",
        ),
    )

    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "JOBAGENT_OPENAI_API_KEY",
        ),
    )

    # Per-request timeout, and retries for transient errors (503, 429,
    # timeouts). The wait doubles each retry, starting at the backoff value.
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = Field(default=2, ge=0)
    llm_retry_backoff_seconds: float = Field(default=2.0, ge=0)

    # Keep in step with RESUME_FILE in automation/job_updater.py and
    # resume_path in data/candidate_profile.json.
    resume_file: Path = DATA_DIR / "Pratik-Resume-2.pdf"

    resume_profile_file: Path = DATA_DIR / "resume_profile.json"

    preferences_file: Path = DATA_DIR / "preferences.json"

    # ------------------------------------------------------------
    # Job matching
    # ------------------------------------------------------------

    # "legacy" keeps the original scorer in automation/job_updater.py.
    matcher: Literal["legacy", "hybrid"] = "legacy"

    # Component weights. They need not sum to 1: a component that cannot
    # be scored for a job is left out and the rest are renormalised.
    match_weight_skill: float = Field(default=0.30, ge=0)
    match_weight_role: float = Field(default=0.30, ge=0)
    match_weight_experience: float = Field(default=0.20, ge=0)
    match_weight_semantic: float = Field(default=0.10, ge=0)
    match_weight_preference: float = Field(default=0.10, ge=0)

    # How much a skill counts by the section that names it. "neutral" is a
    # posting, or part of one, that does not say whether it is required.
    match_skill_weight_required: float = 1.0
    match_skill_weight_neutral: float = 0.7
    match_skill_weight_nice_to_have: float = 0.4

    # Fewer recognised skills than this and the skill score is unavailable.
    match_min_recognised_skills: int = 3

    # Experience score by shortfall in years: (up to this gap, score).
    match_experience_curve: list[tuple[float, float]] = [
        (1, 0.85),
        (2, 0.65),
        (3, 0.40),
        (5, 0.20),
    ]
    match_experience_floor: float = 0.05

    # Label thresholds on the overall score.
    match_label_high_priority: float = 0.80
    match_label_good_match: float = 0.70
    match_label_consider: float = 0.60

    # Label caps. They change the label only, never the score. Set the
    # gap to null (or the label to empty) to switch a cap off.
    match_experience_cap_gap_years: Optional[float] = 4.0
    match_experience_cap_label: str = "CONSIDER"
    match_excluded_role_cap_label: str = "LOW MATCH"

    # Semantic component. The anchors map raw cosine similarity onto 0-1
    # and are specific to the model; recalibrate them if it changes.
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_chunk_words: int = 120
    semantic_chunk_overlap: int = 20
    semantic_top_k: int = 3
    semantic_similarity_floor: float = 0.25
    semantic_similarity_ceiling: float = 0.70

    # Unknown job titles: title-to-role similarity anchors, and the most
    # such a match can score.
    role_similarity_floor: float = 0.30
    role_similarity_ceiling: float = 0.75
    role_fallback_cap: float = 0.70

    # Scores closer than this are a tie, broken in favour of new jobs.
    match_tie_tolerance: float = Field(default=0.01, gt=0)

    top_jobs: int = 10

    taxonomy_file: Optional[Path] = None
    scored_jobs_file: Path = DATA_DIR / "scored_jobs.csv"
    seen_jobs_file: Path = DATA_DIR / "seen_jobs.json"
    eval_dir: Path = BASE_DIR / "eval"

    @property
    def match_weights(self) -> dict[str, float]:
        return {
            "skill": self.match_weight_skill,
            "role": self.match_weight_role,
            "experience": self.match_weight_experience,
            "semantic": self.match_weight_semantic,
            "preference": self.match_weight_preference,
        }

    @property
    def skill_tier_weights(self) -> dict[str, float]:
        return {
            "required": self.match_skill_weight_required,
            "neutral": self.match_skill_weight_neutral,
            "nice_to_have": self.match_skill_weight_nice_to_have,
        }

    @field_validator(
        "resume_file",
        "resume_profile_file",
        "preferences_file",
        "taxonomy_file",
        "scored_jobs_file",
        "seen_jobs_file",
        "eval_dir",
    )
    @classmethod
    def _relative_to_project(cls, value: Optional[Path]) -> Optional[Path]:
        # So paths in .env work from any working directory.
        if value is None:
            return None
        value = Path(value)
        return value if value.is_absolute() else BASE_DIR / value


@lru_cache
def get_settings() -> Settings:
    return Settings()
