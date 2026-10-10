import os

# Before anything imports sentence_transformers; see jobagent/__init__.py.
os.environ.setdefault("USE_TF", "0")

import pytest

from jobagent.config import Settings

SAMPLE_RESUME = """\
Asha Rao
asha.rao@example.com |Github: asha-rao| +91-9876543210 | LinkedIn: Asha Rao
EDUCATION
Example Institute of Technology          August 2019 - May 2023
B.Tech in Computer Science               GPA: 8.4
SKILLS
Languages: Python, SQL
Frameworks: PyTorch, FastAPI, python
EXPERIENCE
Data Analyst, Example Corp               June 2023 - Present
Built churn dashboards in SQL and Python.
PROJECTS
Resume Matcher
Fine-tuned a sentence-transformers model for resume-job matching.
"""


@pytest.fixture
def sample_text():
    return SAMPLE_RESUME


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Settings with no API key and no .env, pointing at a temp directory."""

    for name in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "JOBAGENT_GOOGLE_API_KEY",
        "JOBAGENT_LLM_PROVIDER",
        "JOBAGENT_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    return Settings(
        _env_file=None,
        llm_provider="google",
        llm_model="test-model",
        google_api_key=None,
        resume_file=tmp_path / "resume.pdf",
        resume_profile_file=tmp_path / "resume_profile.json",
        preferences_file=tmp_path / "preferences.json",
    )


@pytest.fixture
def extraction_payload():
    """What a well-behaved LLM would return for SAMPLE_RESUME."""

    return {
        "contact": {
            "name": "Asha Rao",
            "email": "asha.rao@example.com",
            "phone": "+91-9876543210",
            "github": "asha-rao",
            "portfolio": "",
        },
        "skills": [
            {
                "name": "Python",
                "category": "language",
                "evidence": ["skills section", "Example Corp"],
            },
            {"name": "SQL", "category": "language"},
            {"name": "PyTorch", "category": "framework"},
            {"name": "Communication", "category": "soft"},
        ],
        "experience": [
            {
                "company": "Example Corp",
                "title": "Data Analyst",
                "start_date": "June 2023",
                "end_date": "Present",
                "highlights": [
                    "Built churn dashboards in SQL and Python."
                ],
                "technologies": ["SQL", "Python", "Tableau"],
                "outcomes": ["12 dashboards delivered"],
            }
        ],
        "education": [
            {
                "institution": "Example Institute of Technology",
                "location": "Pune",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "gpa": "8.4",
            }
        ],
        "projects": [
            {
                "name": "Resume Matcher",
                "description": "Resume-to-job matching model.",
                "highlights": [
                    "Fine-tuned a sentence-transformers model in PyTorch.",
                    "Evaluated on 1,275 resume-job pairs with MySQL storage.",
                ],
                "technologies": ["sentence-transformers", "pytorch"],
                "outcomes": ["1,275 resume-job pairs"],
            }
        ],
        "total_years_experience": 2.0,
        "seniority": "junior",
        "preferences": {
            "preferred_roles": [
                {"value": "Data Analyst", "source": "inferred"}
            ]
        },
    }


# ------------------------------------------------------------
# Job matching
# ------------------------------------------------------------

import hashlib
from datetime import datetime

import numpy as np

from jobagent.matching.taxonomy import load_taxonomy
from jobagent.resume.schema import ParseMetadata, ResumeProfile


class FakeEmbedder:
    """Deterministic bag-of-words vectors, so tests need no model."""

    DIM = 256

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        vectors = np.zeros((len(texts), self.DIM), dtype=np.float32)

        for row, text in enumerate(texts):
            for token in str(text).lower().split():
                token = token.strip(".,;:()")
                if not token:
                    continue
                index = int(
                    hashlib.md5(token.encode()).hexdigest(), 16
                ) % self.DIM
                vectors[row, index] += 1.0

            norm = np.linalg.norm(vectors[row])
            if norm == 0:
                vectors[row, 0] = 1.0
            else:
                vectors[row] /= norm

        return vectors


@pytest.fixture(scope="session")
def taxonomy():
    return load_taxonomy()


@pytest.fixture
def embedder():
    return FakeEmbedder()


def make_profile(**overrides):
    data = {
        "contact": {"name": "Asha Rao", "location": "India"},
        "summary": "Data analyst and machine learning engineer.",
        "skills": [
            {"name": "Python"},
            {"name": "SQL"},
            {"name": "Machine Learning"},
            {"name": "Excel"},
            {"name": "FastAPI"},
        ],
        "projects": [
            {
                "name": "Churn Model",
                "highlights": ["Served a churn model with FastAPI."],
                "technologies": ["Docker", "Pydantic"],
            }
        ],
        "total_years_experience": 0.5,
        "seniority": "entry",
        "preferences": {
            "preferred_roles": [
                {"value": "Data Analyst", "source": "stated"},
                {"value": "AI/ML Engineer", "source": "stated"},
            ]
        },
    }
    data.update(overrides)

    return ResumeProfile(
        **data,
        metadata=ParseMetadata(
            method="llm",
            model="test-model",
            parsed_at=datetime(2026, 1, 1),
        ),
    )


@pytest.fixture
def profile():
    return make_profile()


def make_job(
    title,
    description="",
    job_id=None,
    location="Noida",
    employment_type="Full Time",
    apply_url=None,
):
    job_id = job_id or title.lower().replace(" ", "-")
    return {
        "job_id": job_id,
        "title": title,
        "company": "Example Co",
        "description": description,
        "location": location,
        "category": "Engineering",
        "employment_type": employment_type,
        "apply_url": apply_url or f"https://jobs.example.com/co/{job_id}/apply",
        "post_date": "0",
        "source": "Lever",
    }


ML_JOB_HTML = """
<div>About Example Co</div>
<div>Our platform leverages large language models. We value excellent people.</div>
<h3>Requirements:</h3>
<ul>
  <li>2+ years of experience with Python and machine learning</li>
  <li>Strong SQL</li>
  <li>Experience with Kubernetes</li>
</ul>
<h3>Nice to have:</h3>
<ul>
  <li>AWS</li>
  <li>FastAPI</li>
</ul>
<h3>Perks &amp; Benefits</h3>
<ul><li>Docker-themed swag and Tableau licences</li></ul>
"""


@pytest.fixture
def ml_job():
    return make_job("Machine Learning Engineer", ML_JOB_HTML, job_id="ml-1")


@pytest.fixture
def match_settings(tmp_path, monkeypatch):
    """Settings with no .env and every data path in a temp directory."""

    for name in ("JOBAGENT_MATCHER",):
        monkeypatch.delenv(name, raising=False)

    return Settings(
        _env_file=None,
        llm_provider="google",
        llm_model="test-model",
        google_api_key=None,
        matcher="hybrid",
        resume_file=tmp_path / "resume.pdf",
        resume_profile_file=tmp_path / "resume_profile.json",
        preferences_file=tmp_path / "preferences.json",
        scored_jobs_file=tmp_path / "scored_jobs.csv",
        seen_jobs_file=tmp_path / "seen_jobs.json",
        eval_dir=tmp_path / "eval",
    )
