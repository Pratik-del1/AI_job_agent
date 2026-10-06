import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd
import pymupdf
import requests

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

# This file is run as a script, so the project root is not on the
# import path by default. The jobagent package lives there.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

RESUME_FILE = (
    DATA_DIR / "Pratik-Resume-2.pdf"
)

RECOMMENDED_JOBS_FILE = (
    DATA_DIR / "recommended_jobs.csv"
)

APPLICATION_QUEUE_FILE = (
    DATA_DIR / "application_queue.csv"
)

UPDATE_METADATA_FILE = (
    DATA_DIR / "job_update_metadata.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "resume-job-matcher-v1"

MODEL_PATH_CANDIDATES = [

    # Actual model location
    BASE_DIR
    / "notebook"
    / "models"
    / MODEL_NAME,

    # Fallback
    BASE_DIR
    / "models"
    / MODEL_NAME,

    # Fallback
    BASE_DIR.parent
    / "models"
    / MODEL_NAME,
]


def find_model_path():

    for path in MODEL_PATH_CANDIDATES:

        if path.exists():

            return path


    print("\nMODEL SEARCH")
    print("=" * 70)

    for path in MODEL_PATH_CANDIDATES:

        print(f"Checked:")
        print(path)
        print()


    raise FileNotFoundError(
        "\nMatching model was not found."
    )


# ============================================================
# SETTINGS
# ============================================================

COMPANY_SLUG = "levelai"

TOP_JOBS = 10


# ============================================================
# RESUME
# ============================================================

def load_resume_text():

    if not RESUME_FILE.exists():

        raise FileNotFoundError(
            f"Resume not found:\n{RESUME_FILE}"
        )


    document = pymupdf.open(
        str(RESUME_FILE)
    )


    text = ""


    for page in document:

        text += page.get_text()


    document.close()


    return text


# ============================================================
# RESUME SECTIONS
# ============================================================

def split_resume_sections(text):

    sections = {}


    section_names = [

        "education",
        "skills",
        "objective",
        "experience",
        "projects",
        "summary",
        "profile"

    ]


    pattern = (

        r"(?im)^\s*("

        + "|".join(
            section_names
        )

        + r")\s*$"
    )


    matches = list(
        re.finditer(
            pattern,
            text
        )
    )


    for i, match in enumerate(
        matches
    ):

        section_name = (
            match.group(1)
            .lower()
        )


        start = match.end()


        if i + 1 < len(matches):

            end = (
                matches[i + 1]
                .start()
            )

        else:

            end = len(text)


        sections[
            section_name
        ] = text[
            start:end
        ].strip()


    return sections


# ============================================================
# CONTACT INFORMATION
# ============================================================

def extract_contact_information(text):

    profile = {}


    email_match = re.search(
        r"[\w\.-]+@[\w\.-]+\.\w+",
        text
    )


    profile["email"] = (

        email_match.group(0)

        if email_match

        else None
    )


    phone_match = re.search(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
        text
    )


    profile["phone"] = (

        phone_match.group(0)

        if phone_match

        else None
    )


    linkedin_match = re.search(
        r"LinkedIn:\s*([^\n|]+)",
        text,
        re.IGNORECASE
    )


    profile["linkedin"] = (

        linkedin_match.group(1).strip()

        if linkedin_match

        else None
    )


    github_match = re.search(
        r"Github:\s*([^\n|]+)",
        text,
        re.IGNORECASE
    )


    profile["github"] = (

        github_match.group(1).strip()

        if github_match

        else None
    )


    return profile


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_section_text(text):

    if not text:

        return ""


    text = text.replace(
        "\u200b",
        ""
    )


    text = text.replace(
        "\xa0",
        " "
    )


    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )


    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    return text.strip()


# ============================================================
# SKILLS
# ============================================================

def extract_skills(skills_text):

    if not skills_text:

        return []


    skills = []


    for line in skills_text.split(
        "\n"
    ):

        line = line.strip()


        if not line:

            continue


        if ":" in line:

            _, line = line.split(
                ":",
                1
            )


        for skill in line.split(","):

            skill = skill.strip()


            if skill:

                skills.append(
                    skill
                )


    return list(
        dict.fromkeys(
            skills
        )
    )


# ============================================================
# CANDIDATE PROFILE
# ============================================================

def load_candidate_profile():

    resume_text = (
        load_resume_text()
    )


    sections = (
        split_resume_sections(
            resume_text
        )
    )


    contact = (
        extract_contact_information(
            resume_text
        )
    )


    return {

        "name": (
            resume_text
            .strip()
            .split("\n")[0]
            .strip()
        ),

        "email": contact.get(
            "email"
        ),

        "phone": contact.get(
            "phone"
        ),

        "linkedin": contact.get(
            "linkedin"
        ),

        "github": contact.get(
            "github"
        ),

        "skills": extract_skills(
            sections.get(
                "skills",
                ""
            )
        ),

        "education": clean_section_text(
            sections.get(
                "education",
                ""
            )
        ),

        "experience": clean_section_text(
            sections.get(
                "experience",
                ""
            )
        ),

        "projects": clean_section_text(
            sections.get(
                "projects",
                ""
            )
        )
    }


# ============================================================
# CANDIDATE TEXT
# ============================================================

def build_candidate_text(profile):

    sections = []


    if profile.get("skills"):

        sections.append(
            "Skills: "
            + ", ".join(
                profile["skills"]
            )
        )


    if profile.get("education"):

        sections.append(
            "Education: "
            + profile["education"]
        )


    if profile.get("experience"):

        sections.append(
            "Experience: "
            + profile["experience"]
        )


    if profile.get("projects"):

        sections.append(
            "Projects: "
            + profile["projects"]
        )


    return "\n\n".join(
        sections
    )


# ============================================================
# FETCH LATEST JOBS
# ============================================================

def fetch_lever_jobs():

    url = (
        f"https://api.lever.co/v0/"
        f"postings/{COMPANY_SLUG}"
        f"?mode=xml"
    )


    response = requests.get(
        url,
        timeout=30
    )


    response.raise_for_status()


    root = ET.fromstring(
        response.content
    )


    jobs = []


    for job in root.findall(
        "job"
    ):

        def get_text(tag):

            element = job.find(
                tag
            )


            if (
                element is not None
                and element.text
            ):

                return (
                    element.text.strip()
                )


            return ""


        jobs.append({

            "job_id": get_text(
                "id"
            ),

            "title": get_text(
                "position"
            ),

            "company": get_text(
                "employer"
            ),

            "description": get_text(
                "description"
            ),

            "location": get_text(
                "location"
            ),

            "category": get_text(
                "category"
            ),

            "employment_type": get_text(
                "commitment"
            ),

            "apply_url": get_text(
                "apply_url"
            ),

            "post_date": get_text(
                "post_date"
            ),

            "source": "Lever"
        })


    return jobs


# ============================================================
# JOB TEXT
# ============================================================

def build_job_text(job):

    return f"""
Job Title: {job['title']}

Company: {job['company']}

Location: {job['location']}

Category: {job['category']}

Employment Type: {job['employment_type']}

Job Description:
{job['description']}
"""


# ============================================================
# ROLE MATCH
# ============================================================

TARGET_ROLE_PATTERNS = {

    "data analyst": [
        "data analyst",
        "junior data analyst",
        "senior data analyst",
        "business data analyst",
        "reporting analyst"
    ],

    "data scientist": [
        "data scientist",
        "junior data scientist"
    ],

    "machine learning engineer": [
        "machine learning engineer",
        "ml engineer"
    ],

    "ai engineer": [
        "ai engineer",
        "artificial intelligence engineer"
    ],

    "ai analyst": [
        "ai analyst",
        "artificial intelligence analyst"
    ],

    "business analyst": [
        "business analyst"
    ],

    "data analytics": [
        "data analytics",
        "analytics analyst",
        "analytics specialist"
    ],

    "nlp": [
        "nlp engineer",
        "nlp scientist",
        "natural language processing engineer"
    ]
}


def role_match(title):

    title = (
        str(title)
        .lower()
        .strip()
    )


    for patterns in (
        TARGET_ROLE_PATTERNS.values()
    ):

        for pattern in patterns:

            if pattern in title:

                return 1.0


    related_patterns = [

        "machine learning",
        "artificial intelligence",
        "data analytics",
        "analytics",
        "nlp",
        "data science"

    ]


    if any(
        pattern in title
        for pattern in related_patterns
    ):

        return 0.6


    return 0.0


# ============================================================
# EXPERIENCE MATCH
# ============================================================

def experience_match(title):

    title = (
        str(title)
        .lower()
    )


    senior_keywords = [

        "senior",
        "sr.",
        "lead",
        "principal",
        "staff",
        "director",
        "head",
        "manager"

    ]


    if any(
        word in title
        for word in senior_keywords
    ):

        return 0.1


    junior_keywords = [

        "junior",
        "jr.",
        "entry level",
        "entry-level",
        "graduate",
        "trainee",
        "intern"

    ]


    if any(
        word in title
        for word in junior_keywords
    ):

        return 1.0


    return 0.8


# ============================================================
# SKILL MATCH
# ============================================================

SKILL_VOCABULARY = [

    "python",
    "sql",
    "machine learning",
    "deep learning",
    "pytorch",
    "tensorflow",
    "keras",
    "scikit-learn",
    "pandas",
    "numpy",
    "xgboost",
    "nlp",
    "transformers",
    "llm",
    "langchain",
    "langgraph",
    "rag",
    "fastapi",
    "docker",
    "git",
    "github",
    "javascript",
    "html",
    "css",
    "excel",
    "power bi",
    "tableau"

]


def extract_job_skills(description):

    text = (
        str(description)
        .lower()
    )


    return {

        skill

        for skill
        in SKILL_VOCABULARY

        if skill in text
    }


def skill_match(
    description,
    candidate_skills
):

    job_skills = (
        extract_job_skills(
            description
        )
    )


    if not job_skills:

        return 0.5


    candidate = {

        str(skill)
        .lower()
        .strip()

        for skill
        in candidate_skills

    }


    matched = (
        job_skills
        .intersection(
            candidate
        )
    )


    return (
        len(matched)
        /
        len(job_skills)
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def recommendation_label(score):

    if score >= 0.80:

        return "HIGH PRIORITY"


    if score >= 0.70:

        return "GOOD MATCH"


    if score >= 0.60:

        return "CONSIDER"


    return "LOW MATCH"


# ============================================================
# LOAD PREVIOUS RECOMMENDATIONS
# ============================================================

def load_previous_jobs():

    if not RECOMMENDED_JOBS_FILE.exists():

        return pd.DataFrame()


    try:

        return pd.read_csv(
            RECOMMENDED_JOBS_FILE
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# LOAD APPLICATION QUEUE
# ============================================================

def load_application_queue():

    if not APPLICATION_QUEUE_FILE.exists():

        return pd.DataFrame()


    try:

        return pd.read_csv(
            APPLICATION_QUEUE_FILE
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# GET PROCESSED JOB IDS
# ============================================================

def get_processed_job_ids():

    queue = (
        load_application_queue()
    )


    if queue.empty:

        return set()


    if "job_id" not in queue.columns:

        return set()


    if "status" not in queue.columns:

        return set()


    active_statuses = {

        "USER_APPROVED",
        "IN_PROGRESS",
        "FORM_FILLED",
        "READY_FOR_REVIEW",
        "USER_CONFIRMED",
        "SUBMITTED"

    }


    processed = queue[
        queue["status"].isin(
            active_statuses
        )
    ]


    return set(
        processed[
            "job_id"
        ]
        .astype(str)
        .tolist()
    )


# ============================================================
# CLEAN LEGACY QUEUE
# ============================================================

def clean_legacy_queue():

    queue = (
        load_application_queue()
    )


    if queue.empty:

        return 0


    if "status" not in queue.columns:

        return 0


    # Old version of the project inserted
    # every recommendation as PENDING.
    #
    # Remove those old PENDING entries.
    # Preserve everything the user actually
    # interacted with.

    before = len(queue)


    queue = queue[
        queue["status"]
        != "PENDING"
    ].copy()


    removed = (
        before - len(queue)
    )


    if removed > 0:

        queue.to_csv(
            APPLICATION_QUEUE_FILE,
            index=False
        )


    return removed


# ============================================================
# SAVE UPDATE METADATA
# ============================================================

def save_update_metadata(
    total_fetched,
    new_jobs,
    recommendations,
    excluded_jobs,
    removed_legacy,
    matcher="legacy"
):

    metadata = {

        "updated_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "total_jobs_fetched":
            int(total_fetched),

        "new_jobs":
            int(new_jobs),

        "recommendations":
            int(recommendations),

        "excluded_jobs":
            int(excluded_jobs),

        "removed_legacy_queue_entries":
            int(removed_legacy),

        "matcher":
            matcher
    }


    UPDATE_METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2
        )
    )


# ============================================================
# SETTINGS AND SEEN JOBS
# ============================================================

def load_settings():

    from jobagent.config import get_settings

    return get_settings()


def load_seen_jobs(settings):

    from jobagent.matching.ranking import SeenJobs

    return SeenJobs(
        settings.seen_jobs_file
    )


def record_seen_jobs(
    seen_jobs,
    jobs
):

    from jobagent.matching.ranking import job_key

    seen_jobs.record(
        job_key(job)
        for job in jobs
    )

    seen_jobs.save()


def build_matching_embedder(settings):

    from jobagent.matching.pipeline import build_embedder

    return build_embedder(
        settings
    )


# ============================================================
# HYBRID RECOMMENDATIONS
# ============================================================

def generate_hybrid_recommendations(
    latest_jobs,
    processed_ids,
    new_job_ids,
    excluded_jobs,
    seen_jobs,
    settings
):

    from jobagent.matching.pipeline import run_hybrid

    print(
        "\nScoring all fetched jobs "
        "with the hybrid matcher..."
    )

    # Every fetched job is scored and written to scored_jobs.csv.
    # Jobs already in the application workflow are flagged there
    # and left out of the recommendations.
    scored, recommended = run_hybrid(
        latest_jobs,
        settings,
        in_workflow_ids=processed_ids,
        embedder=build_matching_embedder(
            settings
        ),
        seen=seen_jobs
    )

    recommended.to_csv(
        RECOMMENDED_JOBS_FILE,
        index=False
    )

    record_seen_jobs(
        seen_jobs,
        latest_jobs
    )

    save_update_metadata(
        total_fetched=len(latest_jobs),
        new_jobs=len(new_job_ids),
        recommendations=len(recommended),
        excluded_jobs=excluded_jobs,
        removed_legacy=0,
        matcher="hybrid"
    )

    print(
        f"Jobs scored: {len(scored)}"
    )

    print(
        f"Saved: {settings.scored_jobs_file}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TOP JOB RECOMMENDATIONS (HYBRID)"
    )

    print(
        "=" * 70
    )

    for _, job in (
        recommended.iterrows()
    ):

        new_label = (
            " [NEW]"
            if bool(job["is_new"])
            else ""
        )

        print(
            f"{job['title']}"
            f"{new_label} | "
            f"{job['company']} | "
            f"{job['location']} | "
            f"{job['final_score']:.2%} | "
            f"{job['recommendation']}"
        )

    return recommended


# ============================================================
# LEGACY SCORING
# ============================================================

def score_jobs_legacy(
    available_jobs,
    candidate_profile,
    matcher
):

    # The original scorer, moved here unchanged so it can also be run
    # on a frozen snapshot for evaluation.

    candidate_text = (
        build_candidate_text(
            candidate_profile
        )
    )


    # --------------------------------------------------------
    # Candidate embedding
    # --------------------------------------------------------

    candidate_embedding = (
        matcher.encode(
            candidate_text,
            normalize_embeddings=True
        )
    )


    # --------------------------------------------------------
    # Job embeddings
    # --------------------------------------------------------

    job_texts = [

        build_job_text(job)

        for job in available_jobs

    ]


    print(
        f"Encoding "
        f"{len(job_texts)} jobs..."
    )


    job_embeddings = (
        matcher.encode(
            job_texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
    )


    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    similarities = (
        cosine_similarity(
            [candidate_embedding],
            job_embeddings
        )[0]
    )


    jobs_df = pd.DataFrame(
        available_jobs
    )


    jobs_df[
        "similarity_score"
    ] = similarities


    # --------------------------------------------------------
    # Skill
    # --------------------------------------------------------

    jobs_df[
        "skill_match"
    ] = jobs_df[
        "description"
    ].apply(

        lambda description:

        skill_match(
            description,
            candidate_profile[
                "skills"
            ]
        )
    )


    # --------------------------------------------------------
    # Role
    # --------------------------------------------------------

    jobs_df[
        "role_match"
    ] = jobs_df[
        "title"
    ].apply(
        role_match
    )


    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    jobs_df[
        "experience_match"
    ] = jobs_df[
        "title"
    ].apply(
        experience_match
    )


    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    jobs_df[
        "final_score"
    ] = (

        jobs_df[
            "similarity_score"
        ] * 0.50

        +

        jobs_df[
            "skill_match"
        ] * 0.25

        +

        jobs_df[
            "role_match"
        ] * 0.15

        +

        jobs_df[
            "experience_match"
        ] * 0.10
    )


    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    jobs_df[
        "recommendation"
    ] = jobs_df[
        "final_score"
    ].apply(
        recommendation_label
    )


    return jobs_df


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations():

    print(
        "\nFetching latest jobs..."
    )


    latest_jobs = (
        fetch_lever_jobs()
    )


    print(
        f"Jobs fetched: "
        f"{len(latest_jobs)}"
    )


    if not latest_jobs:

        raise RuntimeError(
            "No jobs were returned."
        )


    # --------------------------------------------------------
    # Previous recommendations
    # --------------------------------------------------------

    previous_jobs = (
        load_previous_jobs()
    )


    previous_ids = set()


    if (
        not previous_jobs.empty
        and "job_id" in previous_jobs.columns
    ):

        previous_ids = set(
            previous_jobs[
                "job_id"
            ]
            .astype(str)
            .tolist()
        )


    # --------------------------------------------------------
    # Detect genuinely new jobs
    # --------------------------------------------------------

    current_ids = {

        str(job["job_id"])

        for job in latest_jobs

    }


    # A job is new when it has never been fetched before, not when
    # it was merely absent from the previous recommendation list.
    settings = load_settings()

    seen_jobs = load_seen_jobs(
        settings
    )

    new_job_ids = {
        job_id
        for job_id in current_ids
        if seen_jobs.is_new(
            f"id:{job_id}"
        )
    }


    print(
        f"New jobs detected: "
        f"{len(new_job_ids)}"
    )


    # --------------------------------------------------------
    # Clean old queue
    # --------------------------------------------------------

    # A refresh no longer deletes PENDING queue rows.
    # clean_legacy_queue() is kept for deliberate cleanup only.
    removed_legacy = 0


    if removed_legacy:

        print(
            f"Removed {removed_legacy} "
            "legacy PENDING queue entries."
        )


    # --------------------------------------------------------
    # Jobs already being processed/applied
    # --------------------------------------------------------

    processed_ids = (
        get_processed_job_ids()
    )


    # Do not recommend jobs that are already
    # being handled by the application workflow.

    available_jobs = [

        job

        for job in latest_jobs

        if str(job["job_id"])
        not in processed_ids

    ]


    excluded_jobs = (
        len(latest_jobs)
        - len(available_jobs)
    )


    print(
        f"Jobs excluded because they are "
        f"already in application workflow: "
        f"{excluded_jobs}"
    )


    if not available_jobs:

        raise RuntimeError(
            "All fetched jobs are already "
            "in the application workflow."
        )


    # --------------------------------------------------------
    # Hybrid matcher (JOBAGENT_MATCHER=hybrid)
    # --------------------------------------------------------

    if settings.matcher == "hybrid":

        return generate_hybrid_recommendations(
            latest_jobs=latest_jobs,
            processed_ids=processed_ids,
            new_job_ids=new_job_ids,
            excluded_jobs=excluded_jobs,
            seen_jobs=seen_jobs,
            settings=settings
        )


    # --------------------------------------------------------
    # Candidate
    # --------------------------------------------------------

    candidate_profile = (
        load_candidate_profile()
    )


    candidate_text = (
        build_candidate_text(
            candidate_profile
        )
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_path = (
        find_model_path()
    )


    print(
        "\nLoading trained matching model..."
    )


    print(
        f"Model: {model_path}"
    )


    matcher = SentenceTransformer(
        str(model_path)
    )


    jobs_df = score_jobs_legacy(
        available_jobs,
        candidate_profile,
        matcher
    )


    # --------------------------------------------------------
    # New flag
    # --------------------------------------------------------

    jobs_df[
        "is_new"
    ] = jobs_df[
        "job_id"
    ].astype(str).isin(
        new_job_ids
    )


    # --------------------------------------------------------
    # Updated timestamp
    # --------------------------------------------------------

    updated_at = datetime.now().isoformat(
        timespec="seconds"
    )


    jobs_df[
        "updated_at"
    ] = updated_at


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    jobs_df = (
        jobs_df
        .sort_values(
            [
                "is_new",
                "final_score"
            ],
            ascending=[
                False,
                False
            ]
        )
        .drop_duplicates(
            subset=[
                "job_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Top recommendations
    # --------------------------------------------------------

    recommended = (
        jobs_df
        .head(TOP_JOBS)
        .copy()
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_columns = [

        "job_id",
        "title",
        "company",
        "location",
        "employment_type",
        "description",
        "similarity_score",
        "skill_match",
        "role_match",
        "experience_match",
        "final_score",
        "recommendation",
        "apply_url",
        "source",
        "is_new",
        "updated_at"
    ]


    recommended[
        output_columns
    ].to_csv(
        RECOMMENDED_JOBS_FILE,
        index=False
    )


    record_seen_jobs(
        seen_jobs,
        latest_jobs
    )


    save_update_metadata(
        total_fetched=len(latest_jobs),
        new_jobs=len(new_job_ids),
        recommendations=len(recommended),
        excluded_jobs=excluded_jobs,
        removed_legacy=removed_legacy
    )


    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )


    print(
        "TOP JOB RECOMMENDATIONS"
    )


    print(
        "=" * 70
    )


    for _, job in (
        recommended.iterrows()
    ):

        new_label = (
            " [NEW]"
            if bool(job["is_new"])
            else ""
        )


        print(
            f"{job['title']}"
            f"{new_label} | "
            f"{job['company']} | "
            f"{job['location']} | "
            f"{job['final_score']:.2%}"
        )


    return recommended


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )


    print(
        "AI JOB RECOMMENDATION UPDATER"
    )


    print(
        "=" * 70
    )


    try:

        recommended = (
            generate_recommendations()
        )


        print(
            "\n"
            + "=" * 70
        )


        print(
            "JOB UPDATE COMPLETE"
        )


        print(
            "=" * 70
        )


        print(
            f"\nRecommendations: "
            f"{len(recommended)}"
        )


    except Exception as e:

        print(
            "\n"
            + "=" * 70
        )


        print(
            "JOB UPDATE FAILED"
        )


        print(
            "=" * 70
        )


        print(
            f"\n{type(e).__name__}: {e}"
        )


        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()