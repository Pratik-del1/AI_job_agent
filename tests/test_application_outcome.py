"""Recording whether the user submitted an application, from the worker's
prompt and from the dashboard. No browser is opened and nothing is
submitted: the queue is a temp file and the prompt is faked."""

import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import application_state
import application_worker

APP_FILE = Path(__file__).resolve().parent.parent / "app" / "app.py"

QUEUE_COLUMNS = [
    "job_id", "title", "company", "location",
    "apply_url", "match_score", "recommendation", "status",
]

BUTTON_LABEL = "✅ Mark as submitted"


def queue_row(job_id, status, title=None):
    return {
        "job_id": job_id,
        "title": title or f"Role {job_id}",
        "company": "Example Co",
        "location": "Noida",
        "apply_url": f"https://jobs.lever.co/example/{job_id}/apply",
        # Awkward on purpose: a float that must come back unchanged.
        "match_score": "0.9229320284225856",
        "recommendation": "HIGH PRIORITY",
        "status": status,
    }


ROWS = [
    queue_row("pending-1", "PENDING"),
    queue_row("approved-1", "USER_APPROVED"),
    queue_row("progress-1", "IN_PROGRESS"),
    queue_row("filled-1", "FORM_FILLED"),
    queue_row("ready-1", "READY_FOR_REVIEW", title="AI Analyst"),
    queue_row("ready-2", "READY_FOR_REVIEW"),
    queue_row("failed-1", "FAILED"),
    queue_row("done-1", "SUBMITTED"),
]


def write_queue(path, rows=ROWS):
    pd.DataFrame(rows, columns=QUEUE_COLUMNS).to_csv(path, index=False)


def statuses(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dict(zip(frame["job_id"], frame["status"]))


def expected(**changes):
    result = {row["job_id"]: row["status"] for row in ROWS}
    result.update({key.replace("_", "-"): value for key, value in changes.items()})
    return result


def lines_for_others(path, job_id):
    """Every queue line except one job's, exactly as written."""

    return [
        line
        for line in path.read_text().splitlines()
        if not line.startswith(f"{job_id},")
    ]


@pytest.fixture
def queue_file(tmp_path):
    path = tmp_path / "application_queue.csv"
    write_queue(path)
    return path


# ------------------------------------------------------------
# The shared rule: READY_FOR_REVIEW -> SUBMITTED, one row only
# ------------------------------------------------------------

def test_mark_submitted_changes_only_the_matching_row(queue_file):
    others_before = lines_for_others(queue_file, "ready-1")

    changed, message = application_state.mark_submitted(queue_file, "ready-1")

    assert changed
    assert "SUBMITTED" in message
    assert statuses(queue_file) == expected(ready_1="SUBMITTED")
    assert lines_for_others(queue_file, "ready-1") == others_before


@pytest.mark.parametrize(
    "job_id",
    ["pending-1", "approved-1", "progress-1", "filled-1", "failed-1"],
)
def test_only_ready_for_review_can_be_marked_submitted(queue_file, job_id):
    before = queue_file.read_bytes()

    changed, message = application_state.mark_submitted(queue_file, job_id)

    assert not changed
    assert "READY_FOR_REVIEW" in message
    assert queue_file.read_bytes() == before


def test_marking_twice_changes_nothing_the_second_time(queue_file):
    application_state.mark_submitted(queue_file, "ready-1")
    after_first = queue_file.read_bytes()

    changed, message = application_state.mark_submitted(queue_file, "ready-1")

    assert not changed
    assert "already" in message
    assert queue_file.read_bytes() == after_first
    assert len(statuses(queue_file)) == len(ROWS)


def test_unknown_job_and_missing_queue_are_reported(queue_file, tmp_path):
    before = queue_file.read_bytes()

    changed, message = application_state.mark_submitted(queue_file, "nope")
    assert not changed
    assert "not in the queue" in message
    assert queue_file.read_bytes() == before

    missing = tmp_path / "missing.csv"
    changed, message = application_state.mark_submitted(missing, "ready-1")
    assert not changed
    assert "not found" in message
    assert not missing.exists()


# ------------------------------------------------------------
# Worker prompt
# ------------------------------------------------------------

@pytest.fixture
def worker(queue_file, monkeypatch):
    monkeypatch.setattr(application_worker, "QUEUE_FILE", queue_file)

    def answer_with(reply):
        def fake_input(prompt=""):
            if isinstance(reply, BaseException):
                raise reply
            return reply

        monkeypatch.setattr("builtins.input", fake_input)

    return answer_with


@pytest.mark.parametrize("reply", ["y", "Y", "yes", " Yes "])
def test_worker_records_submitted_on_an_explicit_yes(
    worker, queue_file, reply
):
    worker(reply)
    others_before = lines_for_others(queue_file, "ready-1")

    assert application_worker.confirm_submission("ready-1")

    assert statuses(queue_file) == expected(ready_1="SUBMITTED")
    assert lines_for_others(queue_file, "ready-1") == others_before


@pytest.mark.parametrize(
    "reply",
    ["n", "no", "", "   ", "maybe", "yep", "submitted", "not yet",
     EOFError(), KeyboardInterrupt()],
)
def test_worker_never_records_submitted_without_a_clear_yes(
    worker, queue_file, reply
):
    worker(reply)
    before = queue_file.read_bytes()

    assert not application_worker.confirm_submission("ready-1")

    assert queue_file.read_bytes() == before
    assert statuses(queue_file)["ready-1"] == "READY_FOR_REVIEW"


def test_worker_yes_cannot_submit_an_application_that_is_not_ready(
    worker, queue_file
):
    worker("yes")
    before = queue_file.read_bytes()

    assert not application_worker.confirm_submission("approved-1")

    assert queue_file.read_bytes() == before


def test_worker_reports_a_queue_it_cannot_write(
    worker, queue_file, monkeypatch, capsys
):
    worker("yes")

    def broken(queue_file, job_id):
        raise OSError("disk full")

    monkeypatch.setattr(application_worker, "mark_submitted", broken)

    assert not application_worker.confirm_submission("ready-1")
    assert "disk full" in capsys.readouterr().out
    assert statuses(queue_file)["ready-1"] == "READY_FOR_REVIEW"


def test_worker_has_no_code_that_clicks_submit():
    automation = Path(application_worker.__file__).parent

    for name in ("application_worker.py", "form_handler.py"):
        source = (automation / name).read_text().lower()
        assert ".click(" not in source
        assert ".press(" not in source
        assert ".submit(" not in source


# ------------------------------------------------------------
# Dashboard button
# ------------------------------------------------------------

@pytest.fixture
def dashboard(tmp_path):
    """The real dashboard script, run against a temp project directory."""

    (tmp_path / "app").mkdir()
    (tmp_path / "data").mkdir()
    script = tmp_path / "app" / "app.py"
    shutil.copy(APP_FILE, script)

    queue_file = tmp_path / "data" / "application_queue.csv"
    write_queue(queue_file)

    pd.DataFrame(
        [
            {
                "job_id": job_id,
                "title": title,
                "company": "Example Co",
                "location": "Noida",
                "final_score": 0.92,
                "skill_match": 0.9,
                "role_match": 0.9,
                "experience_match": 0.9,
                "recommendation": "HIGH PRIORITY",
                "apply_url": f"https://jobs.lever.co/example/{job_id}/apply",
            }
            for job_id, title in [
                ("ready-1", "AI Analyst"),
                ("approved-1", "Role approved-1"),
                ("new-1", "Role new-1"),
            ]
        ]
    ).to_csv(tmp_path / "data" / "recommended_jobs.csv", index=False)

    class Dashboard:
        pass

    handle = Dashboard()
    handle.queue_file = queue_file
    handle.run = lambda: AppTest.from_file(str(script), default_timeout=60).run()
    return handle


def submit_buttons(app):
    return [button for button in app.button if button.label == BUTTON_LABEL]


def test_dashboard_offers_the_button_only_for_ready_for_review(dashboard):
    app = dashboard.run()

    assert not app.exception
    keys = sorted(button.key for button in submit_buttons(app))

    # ready-1 is on a job card and in the queue section; ready-2 is in
    # the queue only. No other status gets a button.
    assert keys == [
        "queue_submitted_ready-1_4",
        "queue_submitted_ready-2_5",
        "submitted_ready-1_0",
    ]


@pytest.mark.parametrize(
    "key", ["submitted_ready-1_0", "queue_submitted_ready-1_4"]
)
def test_dashboard_button_marks_only_that_application(dashboard, key):
    others_before = lines_for_others(dashboard.queue_file, "ready-1")

    app = dashboard.run()
    app.button(key=key).click().run()

    assert not app.exception
    assert statuses(dashboard.queue_file) == expected(ready_1="SUBMITTED")
    assert lines_for_others(dashboard.queue_file, "ready-1") == others_before
    assert "Application marked as SUBMITTED." in [
        note.value for note in app.success
    ]

    # The button is gone for that job, so it cannot be recorded twice.
    assert sorted(button.key for button in submit_buttons(app)) == [
        "queue_submitted_ready-2_5"
    ]


def test_dashboard_click_after_the_worker_recorded_it_changes_nothing(
    dashboard,
):
    app = dashboard.run()

    # The worker records the outcome while the page is still showing
    # the button.
    application_state.mark_submitted(dashboard.queue_file, "ready-1")
    after_worker = dashboard.queue_file.read_bytes()

    app.button(key="queue_submitted_ready-1_4").click().run()

    assert not app.exception
    assert dashboard.queue_file.read_bytes() == after_worker
    assert len(statuses(dashboard.queue_file)) == len(ROWS)
    assert sorted(button.key for button in submit_buttons(app)) == [
        "queue_submitted_ready-2_5"
    ]


def test_dashboard_reports_a_queue_it_cannot_update(dashboard, monkeypatch):
    app = dashboard.run()
    before = dashboard.queue_file.read_bytes()

    def broken(queue_file, job_id):
        raise OSError("disk full")

    monkeypatch.setattr(application_state, "mark_submitted", broken)

    app.button(key="queue_submitted_ready-1_4").click().run()

    assert not app.exception
    assert dashboard.queue_file.read_bytes() == before
    assert "Could not update application queue: disk full" in [
        note.value for note in app.warning
    ]
