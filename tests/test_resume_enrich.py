from jobagent.resume.enrich import add_usage_evidence
from jobagent.resume.parser import parse_resume_text
from jobagent.resume.schema import ResumeExtraction


def evidence(extraction):
    return {
        skill.name: skill.evidence
        for skill in extraction.skills
    }


def test_new_fields_are_kept(extraction_payload):
    extraction = ResumeExtraction.model_validate(extraction_payload)

    project = extraction.projects[0]
    assert len(project.highlights) == 2
    assert project.outcomes == ["1,275 resume-job pairs"]
    assert extraction.experience[0].outcomes == ["12 dashboards delivered"]
    assert extraction.education[0].location == "Pune"


def test_new_fields_default_to_empty():
    extraction = ResumeExtraction(
        projects=[{"name": "X"}],
        experience=[{"company": "Y"}],
        education=[{"institution": "Z"}],
    )

    assert extraction.projects[0].highlights == []
    assert extraction.experience[0].outcomes == []
    assert extraction.education[0].location is None


def test_evidence_gains_roles_and_projects_that_name_the_skill(
    extraction_payload,
):
    extraction = add_usage_evidence(
        ResumeExtraction.model_validate(extraction_payload)
    )

    result = evidence(extraction)

    # Named in a role bullet and its technologies.
    assert result["SQL"] == ["Example Corp"]
    # Existing evidence is kept, not duplicated.
    assert result["Python"] == ["skills section", "Example Corp"]
    # Named in a project bullet, different capitalisation.
    assert result["PyTorch"] == ["Resume Matcher"]
    # Not mentioned anywhere.
    assert result["Communication"] == []


def test_evidence_matches_whole_terms_only():
    extraction = add_usage_evidence(
        ResumeExtraction(
            skills=[
                {"name": "SQL"},
                {"name": "Git"},
                {"name": "C++"},
            ],
            projects=[
                {
                    "name": "Store",
                    "highlights": [
                        "Stored data in MySQL; code hosted on GitHub.",
                        "Hot path rewritten in C++.",
                    ],
                }
            ],
        )
    )

    assert evidence(extraction) == {
        "SQL": [],
        "Git": [],
        "C++": ["Store"],
    }


def test_original_extraction_is_not_mutated(extraction_payload):
    original = ResumeExtraction.model_validate(extraction_payload)

    add_usage_evidence(original)

    assert evidence(original)["SQL"] == []


def test_parser_applies_evidence_step(
    sample_text, settings, extraction_payload
):
    class Extractor:
        def invoke(self, messages):
            return extraction_payload

    profile = parse_resume_text(
        sample_text, settings=settings, extractor=Extractor()
    )

    assert evidence(profile)["SQL"] == ["Example Corp"]
    assert profile.projects[0].highlights[0].startswith("Fine-tuned")


def test_prompt_includes_todays_date(sample_text, settings):
    seen = []

    class Extractor:
        def invoke(self, messages):
            seen.append(messages)
            return {}

    parse_resume_text(
        sample_text, settings=settings, extractor=Extractor()
    )

    assert "Today's date:" in seen[0][-1][1]
