# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run everything from the project root.

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env            # then set GOOGLE_API_KEY, JOBAGENT_LLM_MODEL

# Tests (pytest.ini puts both `.` and `automation/` on the import path)
python -m pytest
python -m pytest tests/test_matching_scoring.py
python -m pytest tests/test_matching_scoring.py::test_name
python -m pytest -k "experience"

# Run
python -m jobagent.resume.cli              # resume PDF -> data/resume_profile.json (--no-llm, --resume, --output)
python automation/job_updater.py           # fetch + score jobs -> data/recommended_jobs.csv
streamlit run app/app.py                   # dashboard on :8501
python automation/application_worker.py    # host-side queue worker, opens visible Chromium

# Docker (dashboard + updater only; the worker always runs on the host)
docker compose build && docker compose up

# Matcher evaluation
python -m jobagent.matching.evaluate snapshot
python -m jobagent.matching.evaluate template eval/snapshots/NAME.json
python -m jobagent.matching.evaluate compare eval/snapshots/NAME.json
```

There is no linter or formatter configured.

## Architecture

Three separate processes communicate only through CSV/JSON files in `data/`. The whole `data/` directory is gitignored (it holds the resume and personal details) and does not exist in a fresh clone — create it before running anything, and note the Dockerfile's `COPY data ./data` fails without it.

```
automation/job_updater.py ──> data/recommended_jobs.csv ──> app/app.py (Streamlit)
                                                                  │ approve
                                                                  ▼
                                                    data/application_queue.csv
                                                                  │ poll every 5s
                                                                  ▼
                                              automation/application_worker.py (Playwright)
```

- **Dashboard** (`app/app.py`): reads `recommended_jobs.csv`, appends approved jobs to `application_queue.csv` with status `USER_APPROVED`. "Refresh Jobs" shells out to `python automation/job_updater.py` as a subprocess (5-minute timeout) rather than importing it.
- **Job updater** (`automation/job_updater.py`): fetches postings from the Lever XML feed for one hardcoded company (`COMPANY_SLUG`), scores them, and writes the top 10 that are not already in the application queue.
- **Worker** (`automation/application_worker.py`): picks up `USER_APPROVED` rows, detects the ATS from the URL, fills the form in a headed browser, moves the row to `READY_FOR_REVIEW`, and waits for the user. It must never solve CAPTCHAs or click Submit — that is the project's core design constraint, not an omission. `SUBMITTED` is only ever recorded from the user's own confirmation (the worker's y/N prompt or the dashboard's "Mark as submitted" button), both through `application_state.mark_submitted`.

### Two code styles, two import schemes

- `automation/` and `app/` are the original scripts: run as files, flat sibling imports (`from application_state import ...`), module-level path constants, `print` for output. `job_updater.py` inserts the project root into `sys.path` so it can reach `jobagent`, and imports it lazily inside functions.
- `jobagent/` is a proper package: Pydantic models, typed, `logging`, run with `python -m`. All configuration goes through `jobagent/config.py` (`Settings`, env prefix `JOBAGENT_`, read from `.env`); relative paths in settings resolve against the project root.

Match the style of whichever side you are editing.

### Two matchers

`JOBAGENT_MATCHER` selects the scorer inside `generate_recommendations()`:

- `legacy` (default): `score_jobs_legacy` in `job_updater.py`. Uses its own regex resume parser and a fine-tuned sentence-transformers model expected at `notebook/models/resume-job-matcher-v1` (gitignored; the updater raises if it is missing).
- `hybrid`: `jobagent/matching/`. Deterministic, no LLM. Requires `data/resume_profile.json` from the resume CLI. `pipeline.HybridMatcher.match()` runs five component scorers (`skills`, `roles`, `experience`, `semantic`, `preferences`), and `scoring.combine` takes a weighted mean. A component returning `None` is dropped and the remaining weights renormalised — keep that contract when adding or changing a component. Label caps change the label only, never the score.

The hybrid pipeline writes every scored job to `data/scored_jobs.csv` and fills the legacy column names (`final_score`, `skill_match`, … via `LEGACY_ALIASES`) so the dashboard reads either matcher's output unchanged. New columns must be added to `RECOMMENDED_COLUMNS`.

Skills, aliases, role families, seniority terms, location groups and section-heading patterns live in `jobagent/matching/resources/taxonomy.json`; extend that file rather than adding code. Matching is whole-term (so `sql` does not match inside `mysql`).

### Resume profile

`jobagent/resume/parser.py` asks an LLM (via `jobagent/llm.py`: Gemini by default, OpenAI optional, with retry/backoff on transient errors) to fill the `ResumeExtraction` schema, retries once on validation failure, and otherwise falls back to regex parsing (`fallback.py`), so a profile is always returned. `data/preferences.json` overrides preferences read from the resume.

### Things that are duplicated on purpose or by accident

- The resume path is named in three places that must change together: `JOBAGENT_RESUME_FILE` in `.env`, `RESUME_FILE` in `automation/job_updater.py`, and `resume_path` in `data/candidate_profile.json` (used by the worker's upload).
- Form filling uses `data/candidate_profile.json`, not the structured `resume_profile.json`.
- `application_worker.py` defines its own `detect_ats` and status constants instead of importing `ats_detector.py` / `application_state.py`; a change to states or ATS rules has to be made in both.
- Only the Lever form handler is implemented. `fill_greenhouse_form` and `fill_workday_form` in `automation/form_handler.py` are placeholders that return `{}`.

## Tests

Tests need no network, API key or model. `tests/conftest.py` provides `settings` / `match_settings` fixtures (no `.env`, all paths under `tmp_path`), a `FakeEmbedder` with deterministic bag-of-words vectors, and `make_profile` / `make_job` builders. `tests/test_job_updater.py` imports `job_updater` as a top-level module and fakes the network, model and files — use these rather than touching real `data/` files or loading sentence-transformers.

## Other notes

- `checklist.md` describes an earlier "Referral Recommendation System" plan and does not reflect the current code.
- `eval/snapshots/` is committed; `eval/labels.csv` is gitignored.
