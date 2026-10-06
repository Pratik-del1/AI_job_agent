import json

import pytest

from jobagent.resume import cli
from jobagent.resume.fallback import fallback_extraction, split_sections
from jobagent.resume.loader import clean_resume_text, load_resume_text
from jobagent.resume.parser import parse_resume_file, parse_resume_text
from jobagent.resume.preferences import (
    apply_user_preferences,
    load_user_preferences,
)
from jobagent.resume.schema import ResumeExtraction


class FakeExtractor:
    """Stands in for the LLM: returns or raises each response in turn."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def write_pdf(path, text):
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=9)
    document.save(str(path))
    document.close()


# ------------------------------------------------------------
# LLM path
# ------------------------------------------------------------

def test_llm_extraction_builds_profile(
    sample_text, settings, extraction_payload
):
    extractor = FakeExtractor(extraction_payload)

    profile = parse_resume_text(
        sample_text,
        settings=settings,
        extractor=extractor,
        source_file="resume.pdf",
    )

    assert profile.metadata.method == "llm"
    assert profile.metadata.model == settings.llm_model
    assert profile.metadata.fallback_reason is None
    assert profile.metadata.source_file == "resume.pdf"
    assert profile.experience[0].company == "Example Corp"
    assert profile.education[0].degree == "B.Tech"
    assert profile.preferences.preferred_roles[0].source == "inferred"
    assert "skills" in profile.raw_sections
    assert len(extractor.calls) == 1
    assert sample_text in extractor.calls[0][-1][1]


def test_extractor_may_return_a_model(
    sample_text, settings, extraction_payload
):
    extractor = FakeExtractor(
        ResumeExtraction.model_validate(extraction_payload)
    )

    profile = parse_resume_text(
        sample_text, settings=settings, extractor=extractor
    )

    assert profile.metadata.method == "llm"
    assert profile.contact.email == "asha.rao@example.com"


def test_invalid_output_is_retried_with_the_error(
    sample_text, settings, extraction_payload
):
    extractor = FakeExtractor(
        {"seniority": "wizard"},
        extraction_payload,
    )

    profile = parse_resume_text(
        sample_text, settings=settings, extractor=extractor
    )

    assert profile.metadata.method == "llm"
    assert len(extractor.calls) == 2
    assert "did not match the schema" in extractor.calls[1][-1][1]


def test_repeated_invalid_output_falls_back(sample_text, settings):
    extractor = FakeExtractor(
        {"seniority": "wizard"},
        {"seniority": "wizard"},
    )

    profile = parse_resume_text(
        sample_text, settings=settings, extractor=extractor
    )

    assert profile.metadata.method == "fallback"
    assert "ValidationError" in profile.metadata.fallback_reason
    assert len(extractor.calls) == 2
    assert profile.contact.email == "asha.rao@example.com"


def test_api_error_falls_back_without_retry(sample_text, settings):
    extractor = FakeExtractor(RuntimeError("connection refused"))

    profile = parse_resume_text(
        sample_text, settings=settings, extractor=extractor
    )

    assert profile.metadata.method == "fallback"
    assert profile.metadata.model is None
    assert "connection refused" in profile.metadata.fallback_reason
    assert len(extractor.calls) == 1


def test_missing_api_key_falls_back(sample_text, settings):
    profile = parse_resume_text(sample_text, settings=settings)

    assert profile.metadata.method == "fallback"
    assert "GOOGLE_API_KEY" in profile.metadata.fallback_reason


def test_use_llm_false_never_calls_the_extractor(sample_text, settings):
    extractor = FakeExtractor()

    profile = parse_resume_text(
        sample_text,
        settings=settings,
        extractor=extractor,
        use_llm=False,
    )

    assert profile.metadata.method == "fallback"
    assert extractor.calls == []


# ------------------------------------------------------------
# Regex fallback
# ------------------------------------------------------------

def test_fallback_extracts_contact_and_skills(sample_text):
    extraction = fallback_extraction(sample_text)

    assert extraction.contact.name == "Asha Rao"
    assert extraction.contact.email == "asha.rao@example.com"
    assert extraction.contact.phone == "+91-9876543210"
    assert extraction.contact.github == "asha-rao"
    assert extraction.contact.linkedin == "Asha Rao"
    assert [skill.name for skill in extraction.skills] == [
        "Python",
        "SQL",
        "PyTorch",
        "FastAPI",
    ]
    assert extraction.experience == []


def test_split_sections_uses_canonical_names():
    sections = split_sections(
        "Name\nTECHNICAL SKILLS\nPython\nWork Experience\nAnalyst\n"
        "Profile\nCurious engineer."
    )

    assert sections == {
        "skills": "Python",
        "experience": "Analyst",
        "summary": "Curious engineer.",
    }


def test_fallback_handles_text_without_sections():
    extraction = fallback_extraction("")

    assert extraction.contact.name is None
    assert extraction.skills == []


# ------------------------------------------------------------
# Loader
# ------------------------------------------------------------

def test_clean_resume_text_collapses_whitespace():
    text = "Name ​ \n\n\n\n \n EDUCATION\xa0  \n  B.Tech   CS  "

    assert clean_resume_text(text) == "Name\n\nEDUCATION\nB.Tech CS"


def test_load_resume_text_reads_pdf(tmp_path, sample_text):
    path = tmp_path / "resume.pdf"
    write_pdf(path, sample_text)

    text = load_resume_text(path)

    assert "asha.rao@example.com" in text
    assert "SKILLS" in text


def test_load_resume_text_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_resume_text(tmp_path / "missing.pdf")


def test_parse_resume_file_defaults_to_configured_resume(
    sample_text, settings, extraction_payload
):
    write_pdf(settings.resume_file, sample_text)

    profile = parse_resume_file(
        settings=settings,
        extractor=FakeExtractor(extraction_payload),
    )

    assert profile.metadata.source_file == "resume.pdf"
    assert profile.metadata.method == "llm"


# ------------------------------------------------------------
# Preferences
# ------------------------------------------------------------

def test_user_preferences_override_only_supplied_fields(
    sample_text, settings, extraction_payload, tmp_path
):
    extraction_payload["preferences"]["work_modes"] = [
        {"value": "onsite", "source": "inferred"}
    ]
    profile = parse_resume_text(
        sample_text,
        settings=settings,
        extractor=FakeExtractor(extraction_payload),
    )

    path = tmp_path / "preferences.json"
    path.write_text(
        json.dumps(
            {
                "preferred_roles": ["ML Engineer", " AI Engineer "],
                "excluded_roles": ["Sales"],
            }
        )
    )

    updated = apply_user_preferences(
        profile, load_user_preferences(path)
    )

    roles = updated.preferences.preferred_roles
    assert [role.value for role in roles] == ["ML Engineer", "AI Engineer"]
    assert {role.source for role in roles} == {"user"}
    assert updated.preferences.excluded_roles[0].value == "Sales"
    assert updated.preferences.work_modes[0].source == "inferred"
    # The original profile is not mutated.
    assert profile.preferences.preferred_roles[0].value == "Data Analyst"


def test_missing_preferences_file_means_no_overrides(tmp_path):
    assert load_user_preferences(tmp_path / "preferences.json") == {}


@pytest.mark.parametrize(
    "content",
    [
        '{"favourite_colour": ["blue"]}',
        '{"preferred_roles": "ML Engineer"}',
        '["ML Engineer"]',
    ],
)
def test_malformed_preferences_are_rejected(tmp_path, content):
    path = tmp_path / "preferences.json"
    path.write_text(content)

    with pytest.raises(ValueError):
        load_user_preferences(path)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def test_cli_writes_profile_json(
    tmp_path, sample_text, settings, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    write_pdf(settings.resume_file, sample_text)
    settings.preferences_file.write_text(
        json.dumps({"preferred_locations": ["Bengaluru"]})
    )

    assert cli.main(["--no-llm"]) == 0

    saved = json.loads(settings.resume_profile_file.read_text())

    assert saved["metadata"]["method"] == "fallback"
    assert saved["contact"]["email"] == "asha.rao@example.com"
    assert saved["preferences"]["preferred_locations"] == [
        {"value": "Bengaluru", "source": "user"}
    ]
    assert "Python" in saved["technologies"]
    assert "Saved:" in capsys.readouterr().out
