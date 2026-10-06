from pathlib import Path
import subprocess

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RECOMMENDED_JOBS_FILE = (
    DATA_DIR / "recommended_jobs.csv"
)

APPLICATION_QUEUE_FILE = (
    DATA_DIR / "application_queue.csv"
)

JOB_UPDATER_FILE = (
    BASE_DIR / "automation" / "job_updater.py"
)


# ============================================================
# APPLICATION QUEUE COLUMNS
# ============================================================

QUEUE_COLUMNS = [
    "job_id",
    "title",
    "company",
    "location",
    "apply_url",
    "match_score",
    "recommendation",
    "status"
]


# ============================================================
# SCORE FORMATTING
# ============================================================

def format_score(value):

    # A component the matcher could not score is empty in the CSV.
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"

    if pd.isna(number):
        return "n/a"

    return f"{number:.1%}"


def has_value(job, column):

    return (
        column in job.index
        and isinstance(job[column], str)
        and job[column].strip() != ""
    )


# ============================================================
# LOAD RECOMMENDED JOBS
# ============================================================

def load_jobs():

    if not RECOMMENDED_JOBS_FILE.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(
            RECOMMENDED_JOBS_FILE
        )

    except Exception as e:

        st.error(
            f"Could not load recommended jobs: {e}"
        )

        return pd.DataFrame()


# ============================================================
# LOAD APPLICATION QUEUE
# ============================================================

def load_application_queue():

    if not APPLICATION_QUEUE_FILE.exists():

        queue = pd.DataFrame(
            columns=QUEUE_COLUMNS
        )

        queue.to_csv(
            APPLICATION_QUEUE_FILE,
            index=False
        )

        return queue

    try:

        queue = pd.read_csv(
            APPLICATION_QUEUE_FILE
        )

    except Exception as e:

        st.error(
            f"Could not load application queue: {e}"
        )

        return pd.DataFrame(
            columns=QUEUE_COLUMNS
        )

    # Make sure all required columns exist.

    for column in QUEUE_COLUMNS:

        if column not in queue.columns:

            queue[column] = ""

    return queue[QUEUE_COLUMNS]


# ============================================================
# SAVE APPLICATION QUEUE
# ============================================================

def save_application_queue(queue):

    queue.to_csv(
        APPLICATION_QUEUE_FILE,
        index=False
    )


# ============================================================
# RUN JOB UPDATER
# ============================================================

def run_job_updater():

    if not JOB_UPDATER_FILE.exists():

        return False, (
            "Job updater not found:\n"
            f"{JOB_UPDATER_FILE}"
        )

    try:

        result = subprocess.run(
            [
                "python",
                str(JOB_UPDATER_FILE)
            ],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:

            error_output = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown updater error."
            )

            return False, error_output

        return True, (
            result.stdout.strip()
            or "Job update completed successfully."
        )

    except subprocess.TimeoutExpired:

        return False, (
            "Job updater timed out after "
            "5 minutes."
        )

    except Exception as e:

        return False, (
            f"Could not run job updater: {e}"
        )


# ============================================================
# APPROVE APPLICATION
# ============================================================

def approve_application(job):

    queue = load_application_queue()

    job_id = str(
        job["job_id"]
    )

    # --------------------------------------------------------
    # Find existing job
    # --------------------------------------------------------

    if (
        not queue.empty
        and "job_id" in queue.columns
    ):

        matching_rows = (
            queue["job_id"]
            .astype(str)
            .eq(job_id)
        )

    else:

        matching_rows = pd.Series(
            [False] * len(queue),
            index=queue.index
        )

    # --------------------------------------------------------
    # Existing job
    # --------------------------------------------------------

    if matching_rows.any():

        row_index = queue.index[
            matching_rows
        ][0]

        current_status = str(
            queue.at[
                row_index,
                "status"
            ]
        ).upper()

        # Already being processed

        if current_status in [
            "IN_PROGRESS",
            "FORM_FILLED",
            "READY_FOR_REVIEW",
            "USER_CONFIRMED",
            "SUBMITTED"
        ]:

            return False, (
                f"Application is already "
                f"in {current_status} state."
            )

        # Already approved

        if current_status == "USER_APPROVED":

            return False, (
                "Application is already "
                "USER_APPROVED."
            )

        # Failed/PENDING/etc.
        # Re-approve it.

        queue.at[
            row_index,
            "status"
        ] = "USER_APPROVED"

        save_application_queue(
            queue
        )

        return True, (
            "Application approved and "
            "added to the queue."
        )

    # --------------------------------------------------------
    # New job
    # --------------------------------------------------------

    new_application = pd.DataFrame(
        [{
            "job_id": job_id,
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "apply_url": job["apply_url"],
            "match_score": job["final_score"],
            "recommendation": job["recommendation"],
            "status": "USER_APPROVED"
        }]
    )

    queue = pd.concat(
        [
            queue,
            new_application
        ],
        ignore_index=True
    )

    save_application_queue(
        queue
    )

    return True, (
        "Application approved and "
        "added to the queue."
    )


# ============================================================
# GET APPLICATION STATUS
# ============================================================

def get_application_status(
    queue,
    job_id
):

    if (
        queue.empty
        or "job_id" not in queue.columns
    ):

        return "NOT IN QUEUE"

    matching = (
        queue["job_id"]
        .astype(str)
        .eq(str(job_id))
    )

    if not matching.any():

        return "NOT IN QUEUE"

    return str(
        queue.loc[
            matching,
            "status"
        ].iloc[0]
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "💼 AI Job Recommendation System"
)

st.write(
    "Review AI-ranked job opportunities and "
    "approve applications for human-in-the-loop "
    "automation."
)


# ============================================================
# LATEST JOB RECOMMENDATIONS
# ============================================================

st.divider()

header_col1, header_col2 = st.columns(
    [4, 1]
)

with header_col1:

    st.header(
        "🔄 Latest Job Recommendations"
    )

    st.caption(
        "Fetch the latest jobs and run the "
        "matching model."
    )


with header_col2:

    if st.button(
        "🔄 Refresh Jobs",
        width="stretch"
    ):

        with st.spinner(
            "Fetching latest jobs and "
            "running the matching model..."
        ):

            success, output = (
                run_job_updater()
            )

        if success:

            st.success(
                "Job update completed successfully."
            )

            st.session_state[
                "refresh_output"
            ] = output

            st.rerun()

        else:

            st.error(
                "Job update failed."
            )

            st.code(
                output,
                language="text"
            )


# ============================================================
# LOAD JOBS
# ============================================================

jobs = load_jobs()


# ============================================================
# LOAD QUEUE
# ============================================================

queue = load_application_queue()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header(
    "Filters"
)


minimum_score = st.sidebar.slider(
    "Minimum Match Score",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.01
)


recommendation_filter = (
    st.sidebar.multiselect(
        "Recommendation",
        [
            "HIGH PRIORITY",
            "GOOD MATCH",
            "CONSIDER",
            "LOW MATCH"
        ],
        default=[
            "HIGH PRIORITY",
            "GOOD MATCH",
            "CONSIDER"
        ]
    )
)


# ============================================================
# VALIDATE JOB DATA
# ============================================================

if jobs.empty:

    st.warning(
        "No recommended jobs found."
    )

else:

    required_job_columns = [
        "job_id",
        "title",
        "company",
        "location",
        "apply_url",
        "final_score",
        "recommendation",
        "skill_match",
        "role_match",
        "experience_match"
    ]

    missing_columns = [
        column
        for column in required_job_columns
        if column not in jobs.columns
    ]

    if missing_columns:

        st.error(
            "recommended_jobs.csv is missing "
            f"columns: {missing_columns}"
        )

        st.stop()


# ============================================================
# FILTER JOBS
# ============================================================

if not jobs.empty:

    filtered_jobs = jobs[
        (
            jobs["final_score"]
            >= minimum_score
        )
        &
        (
            jobs["recommendation"].isin(
                recommendation_filter
            )
        )
    ].copy()

else:

    filtered_jobs = pd.DataFrame()


# ============================================================
# RECOMMENDED JOBS HEADER
# ============================================================

st.header(
    f"🎯 {len(filtered_jobs)} Recommended Jobs"
)


# ============================================================
# JOB CARDS
# ============================================================

if filtered_jobs.empty:

    st.info(
        "No jobs match the current filters."
    )

else:

    for index, job in (
        filtered_jobs.iterrows()
    ):

        st.divider()

        # ----------------------------------------------------
        # CURRENT APPLICATION STATUS
        # ----------------------------------------------------

        job_id = str(
            job["job_id"]
        )

        current_status = (
            get_application_status(
                queue,
                job_id
            )
        )

        # ----------------------------------------------------
        # JOB LAYOUT
        # ----------------------------------------------------

        col1, col2 = st.columns(
            [4, 1]
        )

        # ----------------------------------------------------
        # JOB INFORMATION
        # ----------------------------------------------------

        with col1:

            st.subheader(
                job["title"]
            )

            st.write(
                f"**{job['company']}** • "
                f"{job['location']}"
            )

            st.write(
                f"🎯 **Match Score: "
                f"{float(job['final_score']):.1%}**"
            )

            st.write(
                f"Recommendation: "
                f"**{job['recommendation']}**"
            )

            st.link_button(
                "🔗 View Job Posting",
                job["apply_url"]
            )

        # ----------------------------------------------------
        # SCORE BREAKDOWN
        # ----------------------------------------------------

        with col2:

            st.metric(
                "Skill Match",
                format_score(job["skill_match"])
            )

            st.metric(
                "Role Match",
                format_score(job["role_match"])
            )

            st.metric(
                "Experience",
                format_score(job["experience_match"])
            )

            # Present only when the hybrid matcher produced the list.
            if "semantic_score" in job.index:

                st.metric(
                    "Semantic",
                    format_score(job["semantic_score"])
                )

                st.metric(
                    "Preference",
                    format_score(job["preference_score"])
                )

        # ----------------------------------------------------
        # SCORE EXPLANATION (hybrid matcher)
        # ----------------------------------------------------

        if has_value(job, "explanation"):

            with st.expander(
                "Why this score"
            ):

                st.text(
                    job["explanation"]
                )


        # ----------------------------------------------------
        # APPLICATION STATUS
        # ----------------------------------------------------

        if current_status != "NOT IN QUEUE":

            st.caption(
                f"Application status: "
                f"**{current_status}**"
            )

        # ----------------------------------------------------
        # APPROVAL BUTTON
        # ----------------------------------------------------

        col_apply, col_skip = st.columns(
            2
        )

        application_locked = (
            current_status in [
                "USER_APPROVED",
                "IN_PROGRESS",
                "FORM_FILLED",
                "READY_FOR_REVIEW",
                "USER_CONFIRMED",
                "SUBMITTED"
            ]
        )

        with col_apply:

            if application_locked:

                button_label = (
                    f"✓ {current_status}"
                )

            else:

                button_label = (
                    "🚀 APPROVE & APPLY"
                )

            if st.button(
                button_label,
                key=f"apply_{job_id}_{index}",
                width="stretch",
                disabled=application_locked
            ):

                success, message = (
                    approve_application(
                        job
                    )
                )

                if success:

                    st.success(
                        message
                    )

                    st.info(
                        "The Mac application worker "
                        "will detect this approved job "
                        "and open the browser."
                    )

                    st.rerun()

                else:

                    st.warning(
                        message
                    )

        # ----------------------------------------------------
        # SKIP
        # ----------------------------------------------------

        with col_skip:

            if st.button(
                "❌ SKIP",
                key=f"skip_{job_id}_{index}",
                width="stretch"
            ):

                st.info(
                    f"Skipped: {job['title']}"
                )


# ============================================================
# APPLICATION QUEUE
# ============================================================

st.divider()

st.header(
    "📋 Application Queue"
)


queue = load_application_queue()


if queue.empty:

    st.info(
        "No applications in the queue."
    )

else:

    # --------------------------------------------------------
    # STATUS COUNTS
    # --------------------------------------------------------

    total = len(queue)

    pending = (
        queue["status"]
        .astype(str)
        .str.upper()
        .eq("PENDING")
        .sum()
    )

    approved = (
        queue["status"]
        .astype(str)
        .str.upper()
        .eq("USER_APPROVED")
        .sum()
    )

    in_progress = (
        queue["status"]
        .astype(str)
        .str.upper()
        .eq("IN_PROGRESS")
        .sum()
    )

    ready = (
        queue["status"]
        .astype(str)
        .str.upper()
        .eq("READY_FOR_REVIEW")
        .sum()
    )

    submitted = (
        queue["status"]
        .astype(str)
        .str.upper()
        .eq("SUBMITTED")
        .sum()
    )

    failed = (
        queue["status"]
        .astype(str)
        .str.upper()
        .eq("FAILED")
        .sum()
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metric1, metric2, metric3, metric4, metric5, metric6 = (
        st.columns(6)
    )

    with metric1:

        st.metric(
            "Total",
            total
        )

    with metric2:

        st.metric(
            "Pending",
            pending
        )

    with metric3:

        st.metric(
            "Approved",
            approved
        )

    with metric4:

        st.metric(
            "In Progress",
            in_progress
        )

    with metric5:

        st.metric(
            "Ready",
            ready
        )

    with metric6:

        st.metric(
            "Submitted",
            submitted
        )

    # --------------------------------------------------------
    # FAILED COUNT
    # --------------------------------------------------------

    if failed > 0:

        st.warning(
            f"{failed} application(s) failed."
        )

    # --------------------------------------------------------
    # QUEUE TABLE
    # --------------------------------------------------------

    display_columns = [
        "title",
        "company",
        "location",
        "match_score",
        "recommendation",
        "status"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in queue.columns
    ]

    st.dataframe(
        queue[available_columns],
        width="stretch",
        hide_index=False
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.divider()

st.header(
    "⚙️ System Status"
)

status_col1, status_col2 = st.columns(
    2
)

with status_col1:

    if RECOMMENDED_JOBS_FILE.exists():

        st.success(
            "✓ Recommended jobs file available"
        )

    else:

        st.error(
            "✗ Recommended jobs file missing"
        )

    if APPLICATION_QUEUE_FILE.exists():

        st.success(
            "✓ Application queue available"
        )

    else:

        st.error(
            "✗ Application queue missing"
        )


with status_col2:

    if JOB_UPDATER_FILE.exists():

        st.success(
            "✓ Job updater available"
        )

    else:

        st.error(
            "✗ Job updater missing"
        )

    st.info(
        "🖥️ Browser automation runs on the "
        "Mac host, not inside Docker."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Human-in-the-loop system • "
    "AI recommends and autofills • "
    "User reviews CAPTCHA, questions, and submits."
)