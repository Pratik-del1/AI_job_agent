from pathlib import Path

import pandas as pd


# ============================================================
# APPLICATION STATES
# ============================================================

USER_APPROVED = "USER_APPROVED"

IN_PROGRESS = "IN_PROGRESS"

FORM_FILLED = "FORM_FILLED"

READY_FOR_REVIEW = "READY_FOR_REVIEW"

USER_CONFIRMED = "USER_CONFIRMED"

SUBMITTING = "SUBMITTING"

SUBMITTED = "SUBMITTED"

FAILED = "FAILED"


# ============================================================
# IDENTIFIER COLUMN
# ============================================================

IDENTIFIER_COLUMN = "job_id"


# ============================================================
# LOAD QUEUE
# ============================================================

def load_application_queue(queue_file):

    queue_file = Path(queue_file)

    if not queue_file.exists():

        raise FileNotFoundError(
            f"Application queue not found:\n"
            f"{queue_file}"
        )

    return pd.read_csv(queue_file)


# ============================================================
# SAVE QUEUE
# ============================================================

def save_application_queue(
    queue,
    queue_file,
):

    queue_file = Path(queue_file)

    queue.to_csv(
        queue_file,
        index=False,
    )


# ============================================================
# FIND APPLICATION
# ============================================================

def find_application_index(
    queue,
    job_id,
):

    if IDENTIFIER_COLUMN in queue.columns:

        matches = queue.index[
            queue[IDENTIFIER_COLUMN]
            .astype(str)
            .eq(str(job_id))
        ]

        if len(matches) > 0:

            return matches[0]

    # --------------------------------------------------------
    # Fallback: pandas index
    # --------------------------------------------------------

    try:

        numeric_job_id = int(job_id)

        if numeric_job_id in queue.index:

            return numeric_job_id

    except (
        ValueError,
        TypeError,
    ):

        pass

    return None


# ============================================================
# UPDATE APPLICATION STATUS
# ============================================================

def update_application_status(
    queue_file,
    job_id,
    status,
):

    queue = load_application_queue(
        queue_file
    )

    index = find_application_index(
        queue,
        job_id,
    )

    if index is None:

        raise ValueError(
            f"Could not find application "
            f"with job_id: {job_id}"
        )

    queue.at[
        index,
        "status",
    ] = status

    save_application_queue(
        queue,
        queue_file,
    )

    print(
        f"\nAPPLICATION STATE UPDATED"
    )

    print(
        f"Job ID:  {job_id}"
    )

    print(
        f"Status:  {status}"
    )

    return True


# ============================================================
# UPDATE BY INDEX
# ============================================================

def update_by_index(
    queue_file,
    index,
    status,
):

    queue = load_application_queue(
        queue_file
    )

    if index not in queue.index:

        raise IndexError(
            f"Queue index {index} "
            f"does not exist."
        )

    queue.at[
        index,
        "status",
    ] = status

    save_application_queue(
        queue,
        queue_file,
    )

    print(
        f"\nAPPLICATION STATE UPDATED"
    )

    print(
        f"Index:   {index}"
    )

    print(
        f"Status:  {status}"
    )

    return True


# ============================================================
# GET APPLICATION STATUS
# ============================================================

def get_application_status(
    queue_file,
    job_id,
):

    queue = load_application_queue(
        queue_file
    )

    index = find_application_index(
        queue,
        job_id,
    )

    if index is None:

        raise ValueError(
            f"Could not find application "
            f"with job_id: {job_id}"
        )

    return str(
        queue.at[
            index,
            "status",
        ]
    )


# ============================================================
# GET APPLICATION
# ============================================================

def get_application(
    queue_file,
    job_id,
):

    queue = load_application_queue(
        queue_file
    )

    index = find_application_index(
        queue,
        job_id,
    )

    if index is None:

        raise ValueError(
            f"Could not find application "
            f"with job_id: {job_id}"
        )

    return queue.loc[index]


# ============================================================
# CHECK STATE
# ============================================================

def is_status(
    queue_file,
    job_id,
    expected_status,
):

    current_status = (
        get_application_status(
            queue_file,
            job_id,
        )
    )

    return (
        current_status
        == expected_status
    )