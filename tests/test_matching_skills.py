import pytest

from jobagent.matching.skills import (
    candidate_skills,
    extract_job_skills,
    score_skills,
)
from jobagent.matching.taxonomy import load_taxonomy, whole_term_pattern
from jobagent.matching.text import (
    chunk_lines,
    clean_html,
    split_sections,
)

from conftest import ML_JOB_HTML

TIER_WEIGHTS = {"required": 1.0, "neutral": 0.7, "nice_to_have": 0.4}


def job_skills(html, taxonomy, title=""):
    return extract_job_skills(
        title,
        split_sections(clean_html(html), taxonomy),
        taxonomy,
    )


# ------------------------------------------------------------
# Whole-term matching
# ------------------------------------------------------------

def test_rag_does_not_match_leverages(taxonomy):
    assert taxonomy.find_skills("Our platform leverages storage.") == set()
    assert taxonomy.find_skills("Experience building RAG pipelines") == {
        "Retrieval-Augmented Generation"
    }


def test_excel_does_not_match_excellent(taxonomy):
    assert taxonomy.find_skills("Excellent communicator, excellence") == set()
    assert taxonomy.find_skills("Advanced Excel modelling") == {"Excel"}


@pytest.mark.parametrize(
    "text",
    ["MySQLdb", "GitHubber", "javascripting", "dockerized-ish", "pythonic"],
)
def test_skills_inside_longer_words_are_not_matched(taxonomy, text):
    assert taxonomy.find_skills(text) == set()


def test_symbols_and_punctuation_are_handled(taxonomy):
    assert taxonomy.find_skills("C++, C# and Node.js.") == {
        "C++",
        "C#",
        "Node.js",
    }
    assert taxonomy.find_skills("(Python/SQL)") == {"Python", "SQL"}
    assert taxonomy.find_skills("NLP-based search") >= {
        "Natural Language Processing"
    }


def test_matching_is_case_insensitive_and_spans_whitespace(taxonomy):
    assert taxonomy.find_skills("MACHINE\n  learning") == {"Machine Learning"}


def test_whole_term_pattern_prefers_longer_alias():
    pattern = whole_term_pattern(["react", "react native"])
    assert pattern.search("React Native apps").group(0) == "React Native"


# ------------------------------------------------------------
# Aliases
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("ML", "Machine Learning"),
        ("NLP", "Natural Language Processing"),
        ("RAG", "Retrieval-Augmented Generation"),
        ("LLM", "Large Language Model"),
        ("LLMs", "Large Language Model"),
        ("k8s", "Kubernetes"),
        ("sklearn", "Scikit-learn"),
        ("GenAI", "Large Language Model"),
    ],
)
def test_aliases_map_to_canonical_names(taxonomy, alias, canonical):
    assert taxonomy.find_skills(f"Experience with {alias}.") == {canonical}


def test_skills_have_a_type(taxonomy):
    assert taxonomy.skill_types["Python"] == "technology"
    assert taxonomy.skill_types["Machine Learning"] == "skill"


def test_taxonomy_is_data_driven(tmp_path):
    path = tmp_path / "taxonomy.json"
    path.write_text(
        """{
          "skills": [
            {"name": "Underwater Welding", "type": "skill",
             "aliases": ["wet welding"]}
          ],
          "role_families": []
        }"""
    )

    taxonomy = load_taxonomy(path)

    assert taxonomy.find_skills("5 years of wet welding") == {
        "Underwater Welding"
    }


def test_duplicate_skill_names_are_rejected(tmp_path):
    path = tmp_path / "taxonomy.json"
    path.write_text(
        '{"skills": [{"name": "Python"}, {"name": "python"}],'
        ' "role_families": []}'
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_taxonomy(path)


def test_soft_skills_are_not_scored(taxonomy):
    assert taxonomy.find_skills("Strong communication skills") == set()


# ------------------------------------------------------------
# Text cleaning and sections
# ------------------------------------------------------------

def test_clean_html_strips_tags_and_keeps_list_items():
    text = clean_html(
        '<p style="x">Hello&nbsp;<b>world</b> &amp; all</p>'
        "<ul><li>One</li><li>Two</li></ul>"
    )

    assert text == "Hello world & all\n- One\n- Two"


def test_sections_are_split_by_heading(taxonomy):
    job_text = split_sections(clean_html(ML_JOB_HTML), taxonomy)

    assert job_text.has_sections
    assert "Kubernetes" in job_text.text("required")
    assert "AWS" in job_text.text("nice_to_have")
    # Company blurb and perks are not part of the job's requirements.
    assert "leverages" not in job_text.relevant_text
    assert "Tableau" not in job_text.relevant_text


def test_inline_plus_marks_a_line_as_nice_to_have(taxonomy):
    job_text = split_sections(
        "Requirements:\n- Python\n- Knowledge of Kafka is a plus",
        taxonomy,
    )

    assert job_text.lines["nice_to_have"] == [
        "- Knowledge of Kafka is a plus"
    ]


def test_description_without_headings_is_neutral(taxonomy):
    job_text = split_sections(
        "We want someone who knows Python, SQL and Docker.",
        taxonomy,
    )

    assert not job_text.has_sections
    assert job_text.lines["required"] == []
    assert len(job_text.lines["neutral"]) == 1


def test_chunks_respect_the_word_limit_and_overlap():
    lines = [" ".join(f"w{i}" for i in range(250))]

    chunks = chunk_lines(lines, max_words=100, overlap=20)

    assert all(len(chunk.split()) <= 100 for chunk in chunks)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]
    assert chunks[-1].split()[-1] == "w249"
    assert chunk_lines([], 100) == []


# ------------------------------------------------------------
# Required vs nice-to-have
# ------------------------------------------------------------

def test_job_skills_are_tiered(taxonomy):
    skills = job_skills(ML_JOB_HTML, taxonomy)

    assert skills.by_tier["required"] == {
        "Python",
        "Machine Learning",
        "SQL",
        "Kubernetes",
    }
    assert skills.by_tier["nice_to_have"] == {"AWS", "FastAPI"}
    assert skills.by_tier["neutral"] == set()


def test_skill_in_two_sections_takes_the_stronger_tier(taxonomy):
    skills = job_skills(
        "<h3>Requirements</h3><ul><li>Python</li></ul>"
        "<h3>Nice to have</h3><ul><li>Python and AWS</li></ul>",
        taxonomy,
    )

    assert skills.by_tier["required"] == {"Python"}
    assert skills.by_tier["nice_to_have"] == {"AWS"}


def test_candidate_skills_include_project_technologies(profile, taxonomy):
    skills = candidate_skills(profile, taxonomy)

    assert {"Python", "SQL", "Machine Learning", "Excel", "FastAPI"} <= skills
    # Not listed as skills, but used in a project.
    assert {"Docker", "Pydantic"} <= skills


def test_skill_score_weights_required_over_nice_to_have(profile, taxonomy):
    result = score_skills(
        job_skills(ML_JOB_HTML, taxonomy),
        candidate_skills(profile, taxonomy),
        TIER_WEIGHTS,
        min_recognised=3,
    )

    # Required: 3 of 4 matched. Nice-to-have: 1 of 2 matched.
    assert result.score == pytest.approx((3 * 1.0 + 1 * 0.4) / (4 * 1.0 + 2 * 0.4))
    assert result.matched == ["FastAPI", "Machine Learning", "Python", "SQL"]
    assert result.missing_required == ["Kubernetes"]
    assert result.missing_nice_to_have == ["AWS"]
    assert result.missing_other == []
    assert result.recognised == 6


def test_missing_nice_to_have_costs_less_than_missing_required(taxonomy):
    candidate = {"Python", "SQL"}

    def score(html):
        return score_skills(
            job_skills(html, taxonomy), candidate, TIER_WEIGHTS, 3
        ).score

    required = score(
        "<h3>Requirements</h3><ul><li>Python, SQL, Kubernetes</li></ul>"
    )
    optional = score(
        "<h3>Requirements</h3><ul><li>Python, SQL</li></ul>"
        "<h3>Nice to have</h3><ul><li>Kubernetes</li></ul>"
    )

    assert optional > required


def test_unsectioned_skills_are_neutral_not_required(taxonomy):
    result = score_skills(
        job_skills("Python, SQL and Kubernetes.", taxonomy),
        {"Python", "SQL"},
        TIER_WEIGHTS,
        3,
    )

    assert result.missing_required == []
    assert result.missing_other == ["Kubernetes"]
    assert result.score == pytest.approx(2 / 3)


def test_too_few_recognised_skills_is_unavailable(taxonomy):
    result = score_skills(
        job_skills("Must know Python.", taxonomy),
        {"Python"},
        TIER_WEIGHTS,
        min_recognised=3,
    )

    assert result.score is None
    assert result.matched == ["Python"]
    assert result.recognised == 1
