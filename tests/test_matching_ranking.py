import json

import pandas as pd
import pytest

from jobagent.matching import evaluate
from jobagent.matching.pipeline import (
    RECOMMENDED_COLUMNS,
    SCORED_COLUMNS,
    HybridMatcher,
    run_hybrid,
    score_jobs,
    top_recommendations,
)
from jobagent.matching.ranking import (
    SeenJobs,
    dedupe_jobs,
    job_key,
    normalize_url,
    sort_key,
)

from conftest import ML_JOB_HTML, make_job

ANALYST_HTML = """
<h3>Requirements</h3>
<ul><li>1+ years of experience</li><li>Python, SQL, Excel</li></ul>
"""

SALES_HTML = """
<h3>Requirements</h3>
<ul><li>6+ years of experience in B2B sales</li><li>Salesforce, prospecting, marketing</li></ul>
"""


def ranked(rows, tolerance=0.01):
    """rows: (key, score, is_new) -> keys in rank order."""

    return [
        key
        for key, score, is_new in sorted(
            rows, key=lambda row: sort_key(row[1], row[2], row[0], tolerance)
        )
    ]


# ------------------------------------------------------------
# Identity
# ------------------------------------------------------------

def test_job_key_prefers_the_job_id():
    job = make_job("Data Analyst", job_id="abc-123")

    assert job_key(job) == "id:abc-123"


def test_job_key_is_stable_across_fetches():
    first = make_job("Data Analyst", job_id="abc")
    later = {**first, "description": "edited", "post_date": "999", "title": "Data Analyst II"}

    assert job_key(first) == job_key(later)


def test_job_key_falls_back_to_url_then_hash():
    by_url = {**make_job("Data Analyst"), "job_id": ""}
    by_hash = {**by_url, "apply_url": ""}

    assert job_key(by_url) == "url:jobs.example.com/co/data-analyst"
    assert job_key(by_hash).startswith("hash:")
    assert job_key(by_hash) == job_key(dict(by_hash))
    assert job_key(by_hash) != job_key({**by_hash, "location": "Pune"})


def test_url_normalisation_ignores_query_and_apply_suffix():
    assert (
        normalize_url("https://Jobs.Example.com/co/abc/apply?source=feed#top")
        == normalize_url("https://jobs.example.com/co/abc/")
        == "jobs.example.com/co/abc"
    )
    assert normalize_url("") == ""


# ------------------------------------------------------------
# Duplicates
# ------------------------------------------------------------

def test_exact_duplicate_records_are_dropped():
    job = make_job("Data Analyst", job_id="a")

    unique, merged = dedupe_jobs([job, dict(job)])

    assert merged == 1
    assert len(unique) == 1
    assert unique[0]["location"] == "Noida"


def test_one_posting_listed_per_city_becomes_one_record_with_both_cities():
    noida = make_job("ML Engineer", job_id="a", location="Noida")
    bangalore = make_job("ML Engineer", job_id="a", location="Bangalore")

    unique, merged = dedupe_jobs([noida, bangalore, dict(noida)])

    assert merged == 2
    assert [job["location"] for job in unique] == ["Noida / Bangalore"]
    # Inputs are not modified.
    assert noida["location"] == "Noida"


def test_same_role_with_its_own_id_in_another_city_is_kept():
    noida = make_job("ML Engineer", job_id="a", location="Noida")
    pune = make_job("ML Engineer", job_id="b", location="Pune")

    unique, merged = dedupe_jobs([noida, pune])

    assert merged == 0
    assert [job["job_id"] for job in unique] == ["a", "b"]


def test_same_url_under_a_different_id_is_one_posting():
    first = make_job("ML Engineer", job_id="a", apply_url="https://x.io/j/1/apply")
    second = make_job("ML Engineer", job_id="b", apply_url="https://x.io/j/1?ref=feed")

    unique, merged = dedupe_jobs([first, second])

    assert merged == 1
    assert unique[0]["job_id"] == "a"


# ------------------------------------------------------------
# Seen jobs
# ------------------------------------------------------------

def test_first_run_has_no_history_so_nothing_is_new(tmp_path):
    seen = SeenJobs(tmp_path / "seen.json")

    assert not seen.has_history
    assert not seen.is_new("id:a")


def test_new_means_never_fetched_before(tmp_path):
    path = tmp_path / "seen.json"

    first = SeenJobs(path)
    first.record(["id:a", "id:b"])
    first.save()

    second = SeenJobs(path)
    assert not second.is_new("id:a")
    assert second.is_new("id:c")

    second.record(["id:a", "id:c"])
    second.save()

    third = SeenJobs(path)
    # Seen once, never new again, whether or not it was recommended.
    assert not third.is_new("id:c")
    assert not third.is_new("id:b")


def test_first_seen_date_is_kept(tmp_path):
    path = tmp_path / "seen.json"
    seen = SeenJobs(path)
    seen.record(["id:a"])
    seen.save()
    stamp = json.loads(path.read_text())["jobs"]["id:a"]

    again = SeenJobs(path)
    again.record(["id:a"])
    again.save()

    assert json.loads(path.read_text())["jobs"]["id:a"] == stamp


# ------------------------------------------------------------
# Rank order
# ------------------------------------------------------------

def test_better_match_ranks_first_regardless_of_newness():
    order = ranked(
        [
            ("new-poor", 0.551, True),
            ("old-strong", 0.827, False),
            ("new-mid", 0.672, True),
        ]
    )

    assert order == ["old-strong", "new-mid", "new-poor"]


def test_newness_breaks_ties_between_equal_scores():
    assert ranked([("old", 0.70, False), ("new", 0.70, True)]) == ["new", "old"]


def test_newness_breaks_ties_between_similar_scores():
    # Within the tolerance band the new job goes first, even if a hair lower.
    assert ranked([("old", 0.704, False), ("new", 0.701, True)]) == ["new", "old"]


def test_newness_does_not_outrank_a_clearly_better_score():
    assert ranked([("old", 0.72, False), ("new", 0.70, True)]) == ["old", "new"]


def test_tolerance_is_configurable():
    rows = [("old", 0.72, False), ("new", 0.70, True)]

    assert ranked(rows, tolerance=0.05) == ["new", "old"]


def test_order_is_deterministic_for_identical_rows():
    rows = [("b", 0.5, False), ("a", 0.5, False), ("c", 0.5, False)]

    assert ranked(rows) == ranked(list(reversed(rows))) == ["a", "b", "c"]


# ------------------------------------------------------------
# Pipeline tables
# ------------------------------------------------------------

@pytest.fixture
def jobs():
    return [
        make_job("Sales Development Manager", SALES_HTML, job_id="sales"),
        make_job("Machine Learning Engineer", ML_JOB_HTML, job_id="ml", location="Noida"),
        make_job("Machine Learning Engineer", ML_JOB_HTML, job_id="ml", location="Bangalore"),
        make_job("Data Analyst", ANALYST_HTML, job_id="analyst"),
    ]


def test_all_jobs_are_scored_and_ranked_by_quality(
    profile, match_settings, embedder, jobs
):
    matcher = HybridMatcher(profile, match_settings, embedder=embedder)

    scored = score_jobs(jobs, matcher)

    assert list(scored["job_id"]) == ["analyst", "ml", "sales"]
    assert list(scored["rank"]) == [1, 2, 3]
    assert scored["overall_score"].is_monotonic_decreasing
    assert scored.loc[scored.job_id == "ml", "location"].item() == "Noida / Bangalore"


def test_new_poor_match_does_not_outrank_seen_strong_match(
    profile, match_settings, embedder, jobs, tmp_path
):
    seen = SeenJobs(tmp_path / "seen.json")
    seen.record(["id:analyst", "id:ml"])
    seen.save()
    seen = SeenJobs(tmp_path / "seen.json")

    matcher = HybridMatcher(profile, match_settings, embedder=embedder)
    scored = score_jobs(jobs, matcher, seen=seen)

    assert dict(zip(scored.job_id, scored.is_new)) == {
        "analyst": False,
        "ml": False,
        "sales": True,
    }
    assert list(scored["job_id"]) == ["analyst", "ml", "sales"]


def test_scored_table_has_the_full_breakdown_and_legacy_columns(
    profile, match_settings, embedder, jobs
):
    matcher = HybridMatcher(profile, match_settings, embedder=embedder)
    scored = score_jobs(jobs, matcher)
    row = scored.iloc[1]

    for column in (
        "overall_score", "skill_score", "role_score", "experience_score",
        "semantic_score", "preference_score", "matched_skills",
        "missing_required_skills", "missing_nice_to_have_skills",
        "role_family", "experience_required", "experience_candidate",
        "explanation",
    ):
        assert column in scored.columns

    assert row["matched_skills"] == "FastAPI; Machine Learning; Python; SQL"
    assert row["missing_required_skills"] == "Kubernetes"
    assert row["matcher"] == "hybrid"
    # Columns the dashboard already reads carry the hybrid values.
    assert row["final_score"] == row["overall_score"]
    assert row["skill_match"] == row["skill_score"]
    assert row["role_match"] == row["role_score"]
    assert row["experience_match"] == row["experience_score"]
    assert row["similarity_score"] == row["semantic_score"]


def test_top_recommendations_skip_jobs_already_in_the_workflow(
    profile, match_settings, embedder, jobs
):
    matcher = HybridMatcher(profile, match_settings, embedder=embedder)
    scored = score_jobs(jobs, matcher, in_workflow_ids={"analyst"})

    top = top_recommendations(scored, top_n=1)

    # Still scored and visible in the full table, but not recommended again.
    assert scored.loc[scored.job_id == "analyst", "in_application_workflow"].item()
    assert list(top["job_id"]) == ["ml"]
    assert list(top.columns) == RECOMMENDED_COLUMNS


def test_run_hybrid_writes_every_scored_job(
    profile, match_settings, embedder, jobs
):
    match_settings.resume_profile_file.write_text(profile.model_dump_json())
    settings = match_settings.model_copy(update={"top_jobs": 2})

    scored, top = run_hybrid(jobs, settings, embedder=embedder)

    saved = pd.read_csv(settings.scored_jobs_file)
    assert len(saved) == len(scored) == 3
    assert list(saved.columns) == SCORED_COLUMNS
    assert len(top) == 2
    assert list(top["job_id"]) == ["analyst", "ml"]


def test_run_hybrid_needs_a_resume_profile(match_settings, embedder, jobs):
    with pytest.raises(FileNotFoundError, match="jobagent.resume.cli"):
        run_hybrid(jobs, match_settings, embedder=embedder)


def test_user_preferences_file_is_applied_at_scoring_time(
    profile, match_settings, embedder, jobs
):
    match_settings.resume_profile_file.write_text(profile.model_dump_json())
    match_settings.preferences_file.write_text(
        json.dumps({"preferred_locations": ["Bangalore"]})
    )

    scored, _ = run_hybrid(jobs, match_settings, embedder=embedder)
    by_id = scored.set_index("job_id")["preference_score"]

    assert by_id["ml"] == 1.0
    assert by_id["sales"] == 0.0


# ------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------

LABELS = {"a": 2, "b": 2, "c": 1, "d": 0, "e": 0}


def test_precision_at_k():
    assert evaluate.precision_at_k(["a", "d", "b", "e", "c"], LABELS, 2) == 0.5
    assert evaluate.precision_at_k(["a", "b"], LABELS, 5) == 1.0
    assert evaluate.precision_at_k(["x"], LABELS, 5) is None


def test_ndcg_is_one_for_the_ideal_order_and_lower_otherwise():
    ideal = ["a", "b", "c", "d", "e"]

    assert evaluate.ndcg_at_k(ideal, LABELS, 10) == pytest.approx(1.0)
    assert evaluate.ndcg_at_k(list(reversed(ideal)), LABELS, 10) < 0.7
    assert evaluate.ndcg_at_k(ideal, {}, 10) is None


def test_unlabelled_jobs_are_ignored_by_the_metrics():
    assert evaluate.ndcg_at_k(
        ["zzz", "a", "b", "yyy", "c", "d", "e"], LABELS, 10
    ) == pytest.approx(1.0)


def test_pairwise_accuracy():
    perfect = {"a": 0.9, "b": 0.8, "c": 0.5, "d": 0.2, "e": 0.1}
    flat = {key: 0.5 for key in LABELS}

    assert evaluate.pairwise_accuracy(perfect, LABELS) == 1.0
    assert evaluate.pairwise_accuracy(flat, LABELS) == 0.5
    assert evaluate.pairwise_accuracy(
        {key: -value for key, value in perfect.items()}, LABELS
    ) == 0.0
    assert evaluate.pairwise_accuracy(perfect, {}) is None


def test_snapshot_round_trip_and_label_template(tmp_path, jobs):
    path = evaluate.save_snapshot(jobs, tmp_path / "snap.json", source="test")
    assert evaluate.load_snapshot(path) == jobs

    labels_path = evaluate.write_label_template(jobs, tmp_path / "labels.csv")
    template = pd.read_csv(labels_path, dtype=str, keep_default_na=False)

    assert list(template.job_id) == ["sales", "ml", "analyst"]
    assert evaluate.load_labels(labels_path) == {}

    template["label"] = ["not_relevant", "Relevant", "maybe"]
    template.to_csv(labels_path, index=False)

    assert evaluate.load_labels(labels_path) == {
        "sales": 0, "ml": 2, "analyst": 1,
    }


def test_unknown_label_is_rejected(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("job_id,title,location,label\na,T,L,great\n")

    with pytest.raises(ValueError, match="Unknown label"):
        evaluate.load_labels(path)


def test_compare_reports_both_matchers_on_the_same_jobs(
    profile, match_settings, embedder, jobs
):
    matcher = HybridMatcher(profile, match_settings, embedder=embedder)
    hybrid = score_jobs(jobs, matcher)
    legacy = pd.DataFrame(
        {
            "job_id": ["sales", "ml", "analyst"],
            "final_score": [0.70, 0.69, 0.68],
            "similarity_score": [0.94, 0.94, 0.94],
            "skill_match": [0.5, 0.5, 0.5],
            "role_match": [0.0, 1.0, 1.0],
            "experience_match": [0.1, 0.8, 0.8],
        }
    )
    labels = {"sales": 0, "ml": 2, "analyst": 2}

    report = evaluate.compare(
        jobs, labels, match_settings, hybrid=hybrid, legacy=legacy
    )

    assert report["jobs"] == 3 and report["labelled"] == 3
    assert report["hybrid"]["pairwise_accuracy"] == 1.0
    assert report["legacy"]["pairwise_accuracy"] == 0.0
    assert report["components"]["legacy"]["semantic"]["pairwise_accuracy"] == 0.5
    shares = report["contribution_share"]
    assert sum(v for v in shares.values() if v) == pytest.approx(1.0)
    assert "legacy" in evaluate.format_report(report)


def test_report_without_labels_makes_no_quality_claim(
    profile, match_settings, embedder, jobs
):
    matcher = HybridMatcher(profile, match_settings, embedder=embedder)
    hybrid = score_jobs(jobs, matcher)
    legacy = hybrid.rename(columns={})

    report = evaluate.compare(
        jobs, {}, match_settings, hybrid=hybrid, legacy=legacy
    )

    assert report["labelled"] == 0
    assert report["hybrid"]["ndcg@10"] is None
    assert "No labels yet" in evaluate.format_report(report)
