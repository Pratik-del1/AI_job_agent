from datetime import datetime

import pytest
from pydantic import ValidationError

from jobagent.resume.schema import (
    ParseMetadata,
    ResumeExtraction,
    ResumeProfile,
    Skill,
)


def make_profile(payload):
    return ResumeProfile(
        **ResumeExtraction.model_validate(payload).model_dump(),
        metadata=ParseMetadata(
            method="llm",
            model="test-model",
            parsed_at=datetime(2026, 1, 1),
        ),
    )


def test_blank_strings_become_none(extraction_payload):
    extraction = ResumeExtraction.model_validate(extraction_payload)

    assert extraction.contact.portfolio is None
    assert extraction.contact.name == "Asha Rao"


def test_empty_extraction_is_valid():
    extraction = ResumeExtraction()

    assert extraction.skills == []
    assert extraction.seniority == "unknown"
    assert extraction.preferences.preferred_roles == []


def test_duplicate_skills_are_merged_case_insensitively():
    extraction = ResumeExtraction(
        skills=[
            {"name": "Python", "evidence": ["skills section"]},
            {"name": " python ", "evidence": ["Example Corp"]},
            {"name": "SQL"},
        ]
    )

    assert [skill.name for skill in extraction.skills] == ["Python", "SQL"]
    assert extraction.skills[0].evidence == [
        "skills section",
        "Example Corp",
    ]


def test_string_lists_are_cleaned():
    extraction = ResumeExtraction(
        experience=[
            {"technologies": ["SQL", " sql", "", "Python"]}
        ]
    )

    assert extraction.experience[0].technologies == ["SQL", "Python"]


def test_technologies_aggregates_and_skips_soft_skills(extraction_payload):
    profile = make_profile(extraction_payload)

    assert profile.technologies == [
        "Python",
        "SQL",
        "PyTorch",
        "Tableau",
        "sentence-transformers",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"seniority": "wizard"},
        {"total_years_experience": -1},
        {"skills": [{"name": "Python", "category": "magic"}]},
        {"preferences": {"preferred_roles": [{"value": "x", "source": "y"}]}},
    ],
)
def test_invalid_values_are_rejected(payload):
    with pytest.raises(ValidationError):
        ResumeExtraction.model_validate(payload)


def test_blank_skill_name_is_rejected():
    with pytest.raises(ValidationError):
        Skill(name="   ")


def test_profile_round_trips_through_json(extraction_payload):
    profile = make_profile(extraction_payload)

    restored = ResumeProfile.model_validate_json(
        profile.model_dump_json()
    )

    assert restored == profile
    assert '"technologies"' in profile.model_dump_json()
