import pytest

from jobagent.matching.experience import (
    parse_years_required,
    score_experience,
    score_gap,
    title_seniority,
)
from jobagent.matching.roles import normalize_title, score_role, title_families
from jobagent.matching.text import split_sections

TARGETS = ["Data Analyst", "AI/ML Engineer"]

CURVE = [(1, 0.85), (2, 0.65), (3, 0.40), (5, 0.20)]


# ------------------------------------------------------------
# Role families
# ------------------------------------------------------------

def test_title_is_normalised(taxonomy):
    assert (
        normalize_title(
            "Sr. Machine Learning Engineer (Speech) - AI (Bangalore/Noida)",
            taxonomy,
        )
        == "machine learning engineer ai"
    )


@pytest.mark.parametrize(
    "title, family",
    [
        ("Machine Learning Engineer", "ml_engineering"),
        ("Senior ML Engineer", "ml_engineering"),
        ("AI/ML Engineer", "ml_engineering"),
        ("AI Engineer", "ai_engineering"),
        ("GenAI Engineer", "genai_engineering"),
        ("LLM Engineer", "genai_engineering"),
        ("Agentic AI Engineer", "agentic_ai"),
        ("NLP Engineer", "nlp_engineering"),
        ("Data Analyst", "data_analysis"),
        ("Data Scientist", "data_science"),
        ("Senior Backend Engineer", "backend_engineering"),
        ("SDET (Noida, India)", "qa_testing"),
        ("Sales Development Manager", "sales"),
        ("Revops Lead", "revenue_operations"),
    ],
)
def test_titles_map_to_role_families(taxonomy, title, family):
    assert title_families(title, taxonomy)[0] == family


def test_intern_inside_a_word_is_not_seniority(taxonomy):
    assert title_seniority("International Sales Engineer", taxonomy) == (
        None,
        None,
    )
    assert title_seniority("Data Analyst Intern", taxonomy)[0] == "junior"


def test_same_family_is_an_exact_match(taxonomy):
    result = score_role("Senior Machine Learning Engineer", TARGETS, taxonomy)

    assert result.score == 1.0
    assert result.match_type == "exact"
    assert result.family == "ml_engineering"
    assert result.target == "AI/ML Engineer"


def test_related_families_score_between_unrelated_and_exact(taxonomy):
    scores = {
        title: score_role(title, ["AI/ML Engineer"], taxonomy).score
        for title in [
            "ML Engineer",
            "AI Engineer",
            "GenAI Engineer",
            "NLP Engineer",
            "Agentic AI Engineer",
            "Backend Engineer",
            "Sales Development Manager",
        ]
    }

    assert scores["ML Engineer"] == 1.0
    # Related, but not identical to an ML Engineer or to each other.
    related = [
        scores["AI Engineer"],
        scores["GenAI Engineer"],
        scores["NLP Engineer"],
        scores["Agentic AI Engineer"],
    ]
    assert all(0.5 < value < 1.0 for value in related)
    assert len(set(related)) > 1
    assert scores["Backend Engineer"] < min(related)
    assert scores["Sales Development Manager"] < scores["Backend Engineer"]


def test_related_match_is_labelled(taxonomy):
    result = score_role("Data Scientist", TARGETS, taxonomy)

    assert result.match_type == "related"
    assert 0 < result.score < 1


def test_second_family_in_title_pulls_the_score_up(taxonomy):
    plain = score_role("Backend Engineer", ["AI/ML Engineer"], taxonomy)
    agents = score_role(
        "Backend Engineer - AI Agents", ["AI/ML Engineer"], taxonomy
    )

    assert agents.family == "backend_engineering"
    assert agents.other_families == ["agentic_ai"]
    assert plain.score < agents.score < 1.0


def test_candidate_roles_drive_the_score(taxonomy):
    as_analyst = score_role("Data Analyst", ["Data Analyst"], taxonomy)
    as_sales = score_role("Data Analyst", ["Account Executive"], taxonomy)

    assert as_analyst.score == 1.0
    assert as_sales.score < 0.5


def test_no_target_roles_is_unavailable(taxonomy):
    assert score_role("Data Analyst", [], taxonomy).score is None


def test_unknown_title_falls_back_to_title_similarity(taxonomy):
    calls = []

    def similarity(title, roles):
        calls.append((title, roles))
        return 0.9

    result = score_role(
        "Chief Happiness Wizard",
        TARGETS,
        taxonomy,
        title_similarity=similarity,
        fallback_cap=0.7,
    )

    assert result.match_type == "semantic"
    assert result.family is None
    # A fuzzy match can never count as much as a recognised family.
    assert result.score == 0.7
    assert calls == [
        ("chief happiness wizard", ["data analyst", "ai/ml engineer"])
    ]


def test_unknown_title_without_an_embedder_is_unavailable(taxonomy):
    result = score_role("Chief Happiness Wizard", TARGETS, taxonomy)

    assert result.score is None


# ------------------------------------------------------------
# Years and seniority
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "text, years",
    [
        ("3+ years of industry experience", 3),
        ("Minimum 3-6 years of backend experience", 3),
        ("2 to 4 years of experience", 2),
        ("1-2 years of full-time work experience", 1),
        ("at least 5 yrs in sales", 5),
        ("3+ years front-end and 2 years of services", 3),
        ("10 plus years leading teams", 10),
        ("Founded in 2019, we serve 10,000 customers", None),
        ("Open to candidates of all backgrounds", None),
    ],
)
def test_years_required_are_parsed(text, years):
    assert parse_years_required(text) == years


def test_title_seniority_uses_the_most_senior_word(taxonomy):
    assert title_seniority("Senior Staff Engineer", taxonomy) == ("lead", 6.0)
    assert title_seniority("Principal Engineer", taxonomy)[0] == "principal"
    assert title_seniority("Engineer (Senior team)", taxonomy) == (None, None)


def test_gap_curve():
    assert score_gap(-2, CURVE, 0.05) == 1.0
    assert score_gap(0, CURVE, 0.05) == 1.0
    assert score_gap(0.5, CURVE, 0.05) == 0.85
    assert score_gap(2, CURVE, 0.05) == 0.65
    assert score_gap(4.5, CURVE, 0.05) == 0.20
    assert score_gap(9, CURVE, 0.05) == 0.05


def experience(title, text, years, taxonomy):
    return score_experience(
        title, split_sections(text, taxonomy), years, taxonomy, CURVE, 0.05
    )


def test_stated_years_take_precedence_over_the_title(taxonomy):
    result = experience(
        "Senior Engineer", "Requirements:\n- 2+ years of Python", 0.5, taxonomy
    )

    assert result.required_years == 2
    assert result.required_source == "stated"
    assert result.score == 0.65
    assert result.gap == 1.5


def test_title_seniority_is_used_when_no_years_are_stated(taxonomy):
    result = experience("Senior Engineer", "Build things.", 0.5, taxonomy)

    assert result.required_years == 4
    assert result.required_source == "title"
    assert result.seniority == "senior"
    assert 0 < result.score < 1


def test_senior_title_is_not_an_automatic_rejection(taxonomy):
    result = experience(
        "Senior Engineer", "Requirements:\n- 2+ years", 3, taxonomy
    )

    assert result.score == 1.0


def test_no_years_and_no_seniority_is_unavailable(taxonomy):
    result = experience("Engineer", "Build things.", 0.5, taxonomy)

    assert result.score is None
    assert result.required_years is None


def test_unknown_candidate_years_is_unavailable(taxonomy):
    result = experience("Engineer", "5+ years required", None, taxonomy)

    assert result.score is None
    assert result.required_years == 5
