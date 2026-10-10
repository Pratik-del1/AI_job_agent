"""The job updater end to end, with the network, model and files faked."""

import json

import numpy as np
import pandas as pd
import pytest

import job_updater

from conftest import ML_JOB_HTML, FakeEmbedder, make_job, make_profile

QUEUE_COLUMNS = [
    "job_id", "title", "company", "location",
    "apply_url", "match_score", "recommendation", "status",
]

LEGACY_COLUMNS = [
    "job_id", "title", "company", "location", "employment_type",
    "description", "similarity_score", "skill_match", "role_match",
    "experience_match", "final_score", "recommendation", "apply_url",
    "source", "is_new", "updated_at",
]

# The updater fixture replaces find_model_path; this is the real one.
REAL_FIND_MODEL_PATH = job_updater.find_model_path

# Similarity the fake model returns for each job, by title.
SIMILARITY = {
    "Data Analyst": 0.8,
    "Senior Sales Manager": 0.6,
    "Machine Learning Engineer": 0.7,
}


class FakeSentenceTransformer:
    """Candidate embeds to [1, 0]; each job to a vector with a known cosine."""

    def __init__(self, path):
        self.path = path

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            return np.array([1.0, 0.0])

        vectors = []
        for text in texts:
            cosine = next(
                value
                for title, value in SIMILARITY.items()
                if f"Job Title: {title}\n" in text
            )
            vectors.append([cosine, (1 - cosine ** 2) ** 0.5])
        return np.array(vectors)


@pytest.fixture
def jobs():
    return [
        make_job(
            "Data Analyst",
            # "leverages" and "Excellent" are here on purpose: the legacy
            # scorer's substring search counts them as "rag" and "excel".
            "We need python and sql. Our platform leverages AI.",
            job_id="analyst",
        ),
        make_job(
            "Senior Sales Manager",
            "Excellent communication.",
            job_id="sales",
        ),
        make_job("Machine Learning Engineer", ML_JOB_HTML, job_id="ml"),
    ]


@pytest.fixture
def updater(tmp_path, monkeypatch, match_settings, jobs):
    """job_updater wired to temp files and fakes. Returns a namespace."""

    queue_file = tmp_path / "application_queue.csv"
    recommended_file = tmp_path / "recommended_jobs.csv"
    metadata_file = tmp_path / "job_update_metadata.json"

    monkeypatch.setattr(job_updater, "APPLICATION_QUEUE_FILE", queue_file)
    monkeypatch.setattr(job_updater, "RECOMMENDED_JOBS_FILE", recommended_file)
    monkeypatch.setattr(job_updater, "UPDATE_METADATA_FILE", metadata_file)

    state = {"jobs": jobs, "settings": match_settings}

    monkeypatch.setattr(job_updater, "fetch_lever_jobs", lambda: state["jobs"])
    monkeypatch.setattr(job_updater, "load_settings", lambda: state["settings"])
    monkeypatch.setattr(job_updater, "find_model_path", lambda: "fake-model")
    monkeypatch.setattr(job_updater, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(
        job_updater,
        "load_candidate_profile",
        lambda: {
            "skills": ["Python", "SQL"],
            "education": "B.Tech",
            "experience": "Analyst intern",
            "projects": "Churn model",
        },
    )
    monkeypatch.setattr(
        job_updater, "build_matching_embedder", lambda settings: FakeEmbedder()
    )

    match_settings.resume_profile_file.write_text(
        make_profile().model_dump_json()
    )

    class Updater:
        pass

    handle = Updater()
    handle.queue_file = queue_file
    handle.recommended_file = recommended_file
    handle.metadata_file = metadata_file
    handle.state = state
    handle.settings = match_settings

    def use(matcher):
        state["settings"] = match_settings.model_copy(
            update={"matcher": matcher}
        )

    handle.use = use
    handle.run = job_updater.generate_recommendations
    return handle


def write_queue(path, rows):
    pd.DataFrame(rows, columns=QUEUE_COLUMNS).to_csv(path, index=False)


def queue_row(job_id, status):
    return [job_id, job_id.title(), "Example Co", "Noida",
            f"https://jobs.example.com/co/{job_id}/apply", 0.7,
            "GOOD MATCH", status]


# ------------------------------------------------------------
# PENDING applications survive a refresh
# ------------------------------------------------------------

@pytest.mark.parametrize("matcher", ["legacy", "hybrid"])
def test_refresh_does_not_delete_pending_applications(updater, matcher):
    updater.use(matcher)
    write_queue(
        updater.queue_file,
        [
            queue_row("analyst", "PENDING"),
            queue_row("old-job", "PENDING"),
            queue_row("sales", "USER_APPROVED"),
            queue_row("failed-job", "FAILED"),
        ],
    )
    before = updater.queue_file.read_bytes()

    updater.run()
    updater.run()

    assert updater.queue_file.read_bytes() == before

    queue = pd.read_csv(updater.queue_file)
    assert list(queue["status"]) == [
        "PENDING", "PENDING", "USER_APPROVED", "FAILED",
    ]
    metadata = json.loads(updater.metadata_file.read_text())
    assert metadata["removed_legacy_queue_entries"] == 0


@pytest.mark.parametrize("matcher", ["legacy", "hybrid"])
def test_user_approved_jobs_are_still_left_out_of_recommendations(
    updater, matcher
):
    updater.use(matcher)
    write_queue(
        updater.queue_file,
        [
            queue_row("analyst", "PENDING"),
            queue_row("sales", "USER_APPROVED"),
        ],
    )

    recommended = updater.run()

    ids = set(recommended["job_id"])
    # USER_APPROVED is already in the workflow; PENDING is not.
    assert "sales" not in ids
    assert "analyst" in ids
    metadata = json.loads(updater.metadata_file.read_text())
    assert metadata["excluded_jobs"] == 1


def test_deliberate_cleanup_is_still_available(updater):
    write_queue(
        updater.queue_file,
        [queue_row("a", "PENDING"), queue_row("b", "USER_APPROVED")],
    )

    assert job_updater.clean_legacy_queue() == 1
    assert list(pd.read_csv(updater.queue_file)["status"]) == ["USER_APPROVED"]


# ------------------------------------------------------------
# Legacy scorer is unchanged when selected
# ------------------------------------------------------------

def test_legacy_scores_are_unchanged(updater):
    updater.use("legacy")

    updater.run()

    saved = pd.read_csv(updater.recommended_file).set_index("job_id")

    assert list(pd.read_csv(updater.recommended_file).columns) == LEGACY_COLUMNS

    analyst = saved.loc["analyst"]
    # Substring vocabulary hits: python, sql and "rag" (inside "leverages").
    assert analyst["skill_match"] == pytest.approx(2 / 3)
    assert analyst["role_match"] == 1.0
    assert analyst["experience_match"] == 0.8
    assert analyst["similarity_score"] == pytest.approx(0.8)
    assert analyst["final_score"] == pytest.approx(
        0.50 * 0.8 + 0.25 * (2 / 3) + 0.15 * 1.0 + 0.10 * 0.8
    )
    assert analyst["recommendation"] == "GOOD MATCH"

    sales = saved.loc["sales"]
    # "excel" is found inside "Excellent"; the candidate lacks it.
    assert sales["skill_match"] == 0.0
    assert sales["role_match"] == 0.0
    assert sales["experience_match"] == 0.1
    assert sales["final_score"] == pytest.approx(0.50 * 0.6 + 0.10 * 0.1)
    assert sales["recommendation"] == "LOW MATCH"

    metadata = json.loads(updater.metadata_file.read_text())
    assert metadata["matcher"] == "legacy"
    # Legacy mode does not produce the hybrid outputs.
    assert not updater.settings.scored_jobs_file.exists()


def test_legacy_scoring_function_matches_the_formula(jobs):
    frame = job_updater.score_jobs_legacy(
        jobs,
        {"skills": ["Python", "SQL"], "education": "", "experience": "", "projects": ""},
        FakeSentenceTransformer("fake"),
    )

    expected = (
        0.50 * frame["similarity_score"]
        + 0.25 * frame["skill_match"]
        + 0.15 * frame["role_match"]
        + 0.10 * frame["experience_match"]
    )

    assert np.allclose(frame["final_score"], expected)
    assert list(frame["similarity_score"].round(6)) == [0.8, 0.6, 0.7]


def test_legacy_keeps_its_new_first_order(updater):
    updater.use("legacy")
    updater.run()

    newcomer = make_job("Senior Sales Manager", "Excellent.", job_id="sales-2")
    updater.state["jobs"] = [*updater.state["jobs"], newcomer]

    recommended = updater.run()

    # The original ordering rule: new jobs first, then by score.
    assert recommended.iloc[0]["job_id"] == "sales-2"
    assert bool(recommended.iloc[0]["is_new"])
    assert recommended.iloc[0]["final_score"] < recommended.iloc[1]["final_score"]


def test_default_matcher_is_legacy(monkeypatch):
    from jobagent.config import Settings

    monkeypatch.delenv("JOBAGENT_MATCHER", raising=False)

    assert Settings(_env_file=None).matcher == "legacy"


def test_unknown_matcher_is_rejected():
    from pydantic import ValidationError

    from jobagent.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, matcher="magic")


# ------------------------------------------------------------
# Hybrid scorer is used when configured
# ------------------------------------------------------------

def test_hybrid_matcher_is_used_when_configured(updater):
    updater.use("hybrid")

    recommended = updater.run()

    saved = pd.read_csv(updater.recommended_file)
    assert set(saved["matcher"]) == {"hybrid"}
    assert list(saved.columns[: len(LEGACY_COLUMNS)]) == LEGACY_COLUMNS
    for column in ("explanation", "matched_skills", "missing_required_skills",
                   "semantic_score", "preference_score", "role_family"):
        assert column in saved.columns

    # Ranked by match quality.
    assert list(saved["job_id"]) == ["analyst", "ml", "sales"]
    assert saved["final_score"].is_monotonic_decreasing
    assert saved.iloc[0]["explanation"].startswith("Overall Match: ")
    assert len(recommended) == 3

    scored = pd.read_csv(updater.settings.scored_jobs_file)
    assert len(scored) == 3

    metadata = json.loads(updater.metadata_file.read_text())
    assert metadata["matcher"] == "hybrid"
    assert metadata["total_jobs_fetched"] == 3


def test_hybrid_scores_every_job_but_recommends_the_top_n(updater):
    updater.state["settings"] = updater.settings.model_copy(
        update={"matcher": "hybrid", "top_jobs": 1}
    )
    write_queue(updater.queue_file, [queue_row("ml", "USER_APPROVED")])

    recommended = updater.run()

    scored = pd.read_csv(updater.settings.scored_jobs_file)
    assert len(scored) == 3
    assert scored.loc[scored.job_id == "ml", "in_application_workflow"].item()
    assert list(recommended["job_id"]) == ["analyst"]


# ------------------------------------------------------------
# The fine-tuned model is a legacy-only requirement
# ------------------------------------------------------------

@pytest.fixture
def missing_model(updater, tmp_path, monkeypatch):
    """The real model lookup, with no model at any candidate path."""

    candidates = [
        tmp_path / "notebook" / "models" / job_updater.MODEL_NAME,
        tmp_path / "models" / job_updater.MODEL_NAME,
    ]

    monkeypatch.setattr(job_updater, "MODEL_PATH_CANDIDATES", candidates)
    monkeypatch.setattr(job_updater, "find_model_path", REAL_FIND_MODEL_PATH)

    return candidates


def test_hybrid_does_not_need_the_legacy_model(
    updater, missing_model, monkeypatch
):
    def no_legacy_model(path):
        raise AssertionError("hybrid must not load the legacy model")

    monkeypatch.setattr(job_updater, "SentenceTransformer", no_legacy_model)
    updater.use("hybrid")

    recommended = updater.run()

    assert len(recommended) == 3
    assert set(pd.read_csv(updater.recommended_file)["matcher"]) == {"hybrid"}


def test_legacy_reports_the_expected_model_path(updater, missing_model):
    updater.use("legacy")
    write_queue(updater.queue_file, [queue_row("ml", "USER_APPROVED")])
    queue_before = updater.queue_file.read_text()

    with pytest.raises(FileNotFoundError) as error:
        updater.run()

    message = str(error.value)
    assert "Matching model was not found" in message
    assert str(missing_model[0]) in message
    assert "JOBAGENT_MATCHER=hybrid" in message

    # It fails rather than switching matcher, and writes nothing.
    assert not updater.recommended_file.exists()
    assert not updater.settings.scored_jobs_file.exists()
    assert not updater.metadata_file.exists()
    assert updater.queue_file.read_text() == queue_before


# ------------------------------------------------------------
# "New" means never fetched before
# ------------------------------------------------------------

@pytest.mark.parametrize("matcher", ["legacy", "hybrid"])
def test_new_flag_tracks_jobs_seen_not_jobs_recommended(updater, matcher):
    updater.state["settings"] = updater.settings.model_copy(
        update={"matcher": matcher, "top_jobs": 1}
    )

    first = updater.run()
    # No history on the first run, so nothing is called new.
    assert not first["is_new"].any()

    second = updater.run()
    # Jobs outside the previous top list were still seen: not new.
    assert not second["is_new"].any()
    assert json.loads(updater.metadata_file.read_text())["new_jobs"] == 0

    newcomer = make_job("Data Analyst", "python sql excel", job_id="analyst-2")
    updater.state["jobs"] = [*updater.state["jobs"], newcomer]

    updater.run()
    assert json.loads(updater.metadata_file.read_text())["new_jobs"] == 1

    updater.run()
    assert json.loads(updater.metadata_file.read_text())["new_jobs"] == 0


def test_hybrid_new_job_does_not_outrank_a_better_match(updater):
    updater.use("hybrid")
    updater.run()

    newcomer = make_job("Senior Sales Manager", "Excellent.", job_id="sales-2")
    updater.state["jobs"] = [*updater.state["jobs"], newcomer]

    recommended = updater.run()

    # The new job scores the same as the old sales job, so it goes just
    # ahead of it, and behind both better matches.
    assert list(recommended["job_id"]) == ["analyst", "ml", "sales-2", "sales"]
    assert list(recommended["is_new"]) == [False, False, True, False]
