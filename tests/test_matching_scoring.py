import numpy as np
import pytest

from jobagent.matching.pipeline import HybridMatcher, target_roles
from jobagent.matching.preferences import score_preferences
from jobagent.matching.scoring import (
    COMPONENTS,
    cap_label,
    combine,
    label_for,
)
from jobagent.matching.semantic import (
    candidate_chunks,
    rescale,
    semantic_score,
)
from jobagent.resume.schema import JobPreferences

from conftest import make_job, make_profile

WEIGHTS = {
    "skill": 0.30,
    "role": 0.30,
    "experience": 0.20,
    "semantic": 0.10,
    "preference": 0.10,
}


def matcher_for(profile, settings, embedder=None):
    return HybridMatcher(profile, settings, embedder=embedder)


def prefs(**fields):
    return JobPreferences(
        **{
            name: [{"value": value, "source": source} for value, source in items]
            for name, items in fields.items()
        }
    )


# ------------------------------------------------------------
# Combining components
# ------------------------------------------------------------

def test_all_components_use_the_configured_weights():
    scores = dict(skill=1.0, role=0.5, experience=0.0, semantic=1.0, preference=0.5)

    overall, used = combine(scores, WEIGHTS)

    assert overall == pytest.approx(0.30 + 0.15 + 0 + 0.10 + 0.05)
    assert used == pytest.approx(WEIGHTS)


def test_missing_component_weight_is_redistributed_proportionally():
    scores = dict(skill=0.8, role=0.6, experience=0.4, semantic=0.2, preference=None)

    overall, used = combine(scores, WEIGHTS)

    assert "preference" not in used
    assert sum(used.values()) == pytest.approx(1.0)
    # Each remaining weight grows by the same factor, 1 / 0.9.
    assert used["skill"] == pytest.approx(0.30 / 0.90)
    assert used["semantic"] == pytest.approx(0.10 / 0.90)
    assert used["skill"] / used["semantic"] == pytest.approx(3.0)
    assert overall == pytest.approx(
        (0.30 * 0.8 + 0.30 * 0.6 + 0.20 * 0.4 + 0.10 * 0.2) / 0.90
    )


def test_missing_component_is_not_scored_as_a_default_value():
    with_gap, _ = combine(
        dict(skill=1.0, role=1.0, experience=None, semantic=None, preference=None),
        WEIGHTS,
    )

    assert with_gap == pytest.approx(1.0)


def test_no_available_component_scores_zero():
    overall, used = combine({name: None for name in COMPONENTS}, WEIGHTS)

    assert overall == 0.0
    assert used == {}


def test_zero_weight_component_is_ignored():
    overall, used = combine(
        dict(skill=1.0, role=0.0, experience=None, semantic=None, preference=None),
        {**WEIGHTS, "role": 0.0},
    )

    assert overall == 1.0
    assert list(used) == ["skill"]


def test_labels_follow_thresholds():
    assert label_for(0.80, 0.8, 0.7, 0.6) == "HIGH PRIORITY"
    assert label_for(0.79, 0.8, 0.7, 0.6) == "GOOD MATCH"
    assert label_for(0.60, 0.8, 0.7, 0.6) == "CONSIDER"
    assert label_for(0.59, 0.8, 0.7, 0.6) == "LOW MATCH"


def test_cap_only_lowers_a_label():
    assert cap_label("HIGH PRIORITY", "CONSIDER") == "CONSIDER"
    assert cap_label("LOW MATCH", "CONSIDER") == "LOW MATCH"
    assert cap_label("HIGH PRIORITY", "") == "HIGH PRIORITY"
    assert cap_label("HIGH PRIORITY", None) == "HIGH PRIORITY"


# ------------------------------------------------------------
# Semantic component
# ------------------------------------------------------------

@pytest.mark.parametrize("value", [-1.0, 0.0, 0.2, 0.45, 0.7, 0.99, 5.0])
def test_rescale_is_bounded(value):
    assert 0.0 <= rescale(value, 0.25, 0.70) <= 1.0


def test_rescale_anchors():
    assert rescale(0.25, 0.25, 0.70) == 0.0
    assert rescale(0.70, 0.25, 0.70) == 1.0
    assert rescale(0.475, 0.25, 0.70) == pytest.approx(0.5)

    with pytest.raises(ValueError):
        rescale(0.5, 0.7, 0.7)


def test_semantic_score_is_bounded_and_orders_related_text(profile, embedder):
    vectors = embedder.encode(candidate_chunks(profile))

    related, raw_related = semantic_score(
        vectors,
        ["Served a churn model with FastAPI and Docker."],
        embedder, 3, 0.0, 1.0,
    )
    unrelated, raw_unrelated = semantic_score(
        vectors,
        ["Preheat the oven and bake the chocolate cake."],
        embedder, 3, 0.0, 1.0,
    )

    assert 0.0 <= unrelated < related <= 1.0
    assert raw_unrelated < raw_related


def test_semantic_score_is_clipped_at_the_anchors(profile, embedder):
    vectors = embedder.encode(candidate_chunks(profile))
    chunk = [candidate_chunks(profile)[0]]

    high, _ = semantic_score(vectors, chunk, embedder, 3, 0.0, 0.5)
    low, _ = semantic_score(vectors, ["zzz qqq"], embedder, 3, 0.5, 0.9)

    assert high == 1.0
    assert low == 0.0


def test_semantic_score_uses_the_best_chunks(profile, embedder):
    vectors = embedder.encode(candidate_chunks(profile))
    good = "Served a churn model with FastAPI."
    noise = ["lorem ipsum dolor", "sit amet consectetur", "adipiscing elit"]

    top_one, _ = semantic_score(vectors, [good, *noise], embedder, 1, 0.0, 1.0)
    top_all, _ = semantic_score(vectors, [good, *noise], embedder, 4, 0.0, 1.0)

    assert top_one > top_all


def test_semantic_score_without_text_is_unavailable(profile, embedder):
    vectors = embedder.encode(candidate_chunks(profile))

    assert semantic_score(vectors, [], embedder, 3, 0.0, 1.0) == (None, None)
    assert semantic_score(None, ["text"], embedder, 3, 0.0, 1.0) == (None, None)


def test_candidate_is_split_into_facets(profile):
    chunks = candidate_chunks(profile)

    assert chunks[0].startswith("Data analyst")
    assert chunks[1].startswith("Skills: Python")
    assert any("Churn Model" in chunk for chunk in chunks)


# ------------------------------------------------------------
# Preferences
# ------------------------------------------------------------

def test_no_explicit_preferences_is_unavailable(profile, taxonomy):
    job = make_job("Data Analyst", location="Noida")

    result = score_preferences(job, profile.preferences, taxonomy)

    assert result.score is None
    assert result.details == {}


def test_contact_location_is_not_a_location_preference(
    match_settings, ml_job
):
    # The profile's contact location is India and the job is in Noida,
    # yet without a configured preference nothing is scored.
    profile = make_profile(contact={"name": "A", "location": "Noida"})

    match = matcher_for(profile, match_settings).match(ml_job)

    assert match.preference_score is None
    assert "preference" not in match.weights_used


def test_inferred_preferences_are_ignored(taxonomy):
    preferences = prefs(preferred_locations=[("Noida", "inferred")])

    result = score_preferences(make_job("Data Analyst"), preferences, taxonomy)

    assert result.score is None


def test_location_preference_matches_city_and_country_group(taxonomy):
    job = make_job("Data Analyst", location="Noida")

    def score(location):
        return score_preferences(
            job, prefs(preferred_locations=[(location, "user")]), taxonomy
        ).score

    assert score("Noida") == 1.0
    assert score("India") == 1.0
    assert score("Berlin") == 0.0


def test_remote_preference_matches_remote_jobs(taxonomy):
    preferences = prefs(preferred_locations=[("Remote", "user")])

    remote = make_job("Engineer (Remote)", location="India")
    onsite = make_job("Engineer", location="Noida")

    assert score_preferences(remote, preferences, taxonomy).score == 1.0
    assert score_preferences(onsite, preferences, taxonomy).score == 0.0


def test_preference_score_is_the_mean_of_available_checks(taxonomy):
    preferences = prefs(
        preferred_locations=[("Berlin", "user")],
        employment_types=[("full-time", "user")],
        work_modes=[("remote", "user")],
    )
    job = make_job("Data Analyst", location="Noida", employment_type="Full Time")

    result = score_preferences(job, preferences, taxonomy)

    # Work mode is unknown for this job, so it is not counted.
    assert result.details == {"location": False, "employment_type": True}
    assert result.score == 0.5


def test_excluded_role_is_flagged(taxonomy):
    preferences = prefs(excluded_roles=[("Sales", "user")])

    sales = score_preferences(
        make_job("Sales Development Manager"), preferences, taxonomy
    )
    analyst = score_preferences(make_job("Data Analyst"), preferences, taxonomy)

    assert sales.excluded_role and sales.score == 0.0
    assert not analyst.excluded_role and analyst.score == 1.0


# ------------------------------------------------------------
# Whole match: breakdown, explanation, determinism
# ------------------------------------------------------------

def test_score_breakdown_fields(profile, match_settings, embedder, ml_job):
    match = matcher_for(profile, match_settings, embedder).match(ml_job)

    assert match.role_score == 1.0
    assert match.role_family == "ml_engineering"
    assert match.role_match_type == "exact"
    assert match.experience_required == 2.0
    assert match.experience_required_source == "stated"
    assert match.experience_candidate == 0.5
    assert match.experience_score == 0.65
    assert match.matched_skills == ["FastAPI", "Machine Learning", "Python", "SQL"]
    assert match.missing_required_skills == ["Kubernetes"]
    assert match.missing_nice_to_have_skills == ["AWS"]
    assert 0.0 <= match.semantic_score <= 1.0
    assert match.preference_score is None

    for value in (match.overall_score, match.skill_score):
        assert 0.0 <= value <= 1.0

    # The overall score is exactly the weighted sum of the parts shown.
    assert sum(match.weights_used.values()) == pytest.approx(1.0)
    assert match.overall_score == pytest.approx(
        sum(match.contributions().values())
    )
    assert match.overall_score == pytest.approx(
        (
            0.30 * match.skill_score
            + 0.30 * 1.0
            + 0.20 * 0.65
            + 0.10 * match.semantic_score
        ) / 0.90
    )


def test_explanation_is_built_from_the_scores(
    profile, match_settings, embedder, ml_job
):
    match = matcher_for(profile, match_settings, embedder).match(ml_job)
    text = match.explanation

    assert text.startswith(f"Overall Match: {round(match.overall_score * 100)}%")
    assert "- Skill Match: " in text
    assert "- Role Match: 100% (ML Engineer, exact match to AI/ML Engineer)" in text
    assert "- Experience Match: 65% (asks 2+ years (stated); resume shows 0.5)" in text
    assert "- Preference Match: n/a (no explicit preferences configured)" in text
    assert "Matched Skills:\n- FastAPI\n- Machine Learning\n- Python\n- SQL" in text
    assert "Missing Required Skills:\n- Kubernetes" in text
    assert "Missing Nice-to-Have Skills:\n- AWS" in text
    assert "Not scored (weight shared among the other components): Preference Match" in text


def test_scoring_is_deterministic(profile, match_settings, embedder, ml_job):
    first = matcher_for(profile, match_settings, embedder).match(ml_job)
    second = matcher_for(profile, match_settings, embedder).match(ml_job)

    assert first == second
    assert first.explanation == second.explanation


def test_score_does_not_depend_on_other_jobs(
    profile, match_settings, embedder, ml_job
):
    matcher = matcher_for(profile, match_settings, embedder)

    alone = matcher.match(ml_job)
    matcher.match(make_job("Sales Development Manager", "Sell things."))
    again = matcher.match(ml_job)

    assert alone == again


def test_without_an_embedder_semantic_is_unavailable(
    profile, match_settings, ml_job
):
    match = matcher_for(profile, match_settings).match(ml_job)

    assert match.semantic_score is None
    assert set(match.weights_used) == {"skill", "role", "experience"}


def test_weights_come_from_settings(profile, match_settings, ml_job):
    role_only = match_settings.model_copy(
        update=dict(
            match_weight_skill=0.0,
            match_weight_experience=0.0,
            match_weight_semantic=0.0,
            match_weight_preference=0.0,
        )
    )

    match = matcher_for(profile, role_only).match(ml_job)

    assert match.weights_used == {"role": 1.0}
    assert match.overall_score == match.role_score == 1.0


def test_thin_posting_reports_unavailable_components(profile, match_settings):
    job = make_job("Wizard", "<p>Come and do magic with us.</p>")

    match = matcher_for(profile, match_settings).match(job)

    assert match.skill_score is None
    assert match.role_score is None
    assert match.experience_score is None
    assert match.overall_score == 0.0
    assert match.recommendation == "LOW MATCH"
    assert "Not scored" in match.explanation


# ------------------------------------------------------------
# Seniority and label caps
# ------------------------------------------------------------

SENIOR_HTML = """
<h3>Requirements</h3>
<ul><li>6+ years of experience</li><li>Python, SQL, machine learning, FastAPI</li></ul>
"""


def test_seniority_gap_lowers_experience_but_not_the_other_components(
    profile, match_settings
):
    job = make_job("Senior Machine Learning Engineer", SENIOR_HTML)

    match = matcher_for(profile, match_settings).match(job)

    assert match.experience_score == 0.05
    assert match.skill_score == 1.0
    assert match.role_score == 1.0
    # Strong skill and role alignment still shows in the overall score.
    assert match.overall_score == pytest.approx((0.3 + 0.3 + 0.2 * 0.05) / 0.8)
    assert match.overall_score > 0.75


def test_experience_cap_changes_the_label_only(profile, match_settings):
    job = make_job("Senior Machine Learning Engineer", SENIOR_HTML)

    capped = matcher_for(profile, match_settings).match(job)
    uncapped = matcher_for(
        profile,
        match_settings.model_copy(
            update={"match_experience_cap_gap_years": None}
        ),
    ).match(job)

    assert uncapped.recommendation == "GOOD MATCH"
    assert uncapped.label_note is None

    assert capped.recommendation == "CONSIDER"
    assert "Capped from GOOD MATCH" in capped.label_note
    assert "Capped from GOOD MATCH" in capped.explanation
    assert capped.overall_score == uncapped.overall_score


def test_cap_label_is_configurable(profile, match_settings):
    job = make_job("Senior Machine Learning Engineer", SENIOR_HTML)

    match = matcher_for(
        profile,
        match_settings.model_copy(
            update={"match_experience_cap_label": "LOW MATCH"}
        ),
    ).match(job)

    assert match.recommendation == "LOW MATCH"


def test_small_gap_is_not_capped(profile, match_settings, ml_job):
    match = matcher_for(profile, match_settings).match(ml_job)

    assert match.label_note is None


def test_excluded_role_caps_the_label(match_settings):
    profile = make_profile(
        preferences={
            "preferred_roles": [{"value": "AI/ML Engineer", "source": "stated"}],
            "excluded_roles": [{"value": "Machine Learning Engineer", "source": "user"}],
        }
    )
    job = make_job(
        "Machine Learning Engineer",
        "<h3>Requirements</h3><ul><li>Python, SQL, machine learning</li></ul>",
    )

    match = matcher_for(profile, match_settings).match(job)

    assert match.recommendation == "LOW MATCH"
    assert "excluded-roles list" in match.label_note
    assert match.overall_score > 0.6


# ------------------------------------------------------------
# Target roles
# ------------------------------------------------------------

def test_stated_roles_are_preferred_over_inferred():
    profile = make_profile(
        preferences={
            "preferred_roles": [
                {"value": "Data Analyst", "source": "stated"},
                {"value": "Product Manager", "source": "inferred"},
            ]
        }
    )

    assert target_roles(profile) == ["Data Analyst"]


def test_inferred_roles_are_used_when_nothing_is_stated():
    profile = make_profile(
        preferences={
            "preferred_roles": [{"value": "Data Analyst", "source": "inferred"}]
        }
    )

    assert target_roles(profile) == ["Data Analyst"]
