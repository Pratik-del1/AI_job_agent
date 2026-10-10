import json
import time
from pathlib import Path

import pandas as pd

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from application_state import (
    mark_submitted,
    update_application_status,
)


from form_handler import (
    handle_application_form,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

QUEUE_FILE = DATA_DIR / "application_queue.csv"

CANDIDATE_PROFILE_FILE = (
    DATA_DIR / "candidate_profile.json"
)

SCREENSHOT_BEFORE = (
    DATA_DIR / "application_form_before.png"
)

SCREENSHOT_FILLED = (
    DATA_DIR / "application_form_filled.png"
)


# ============================================================
# APPLICATION STATES
# ============================================================

USER_APPROVED = "USER_APPROVED"

IN_PROGRESS = "IN_PROGRESS"

FORM_FILLED = "FORM_FILLED"

READY_FOR_REVIEW = "READY_FOR_REVIEW"

USER_CANCELLED = "USER_CANCELLED"

FAILED = "FAILED"


# ============================================================
# WORKER SETTINGS
# ============================================================

POLL_INTERVAL = 5

BROWSER_CLOSE_DELAY = 30


# ============================================================
# ATS DETECTION
# ============================================================

def detect_ats(url):

    from urllib.parse import urlparse

    parsed = urlparse(url)

    domain = (
        parsed.netloc
        .lower()
        .replace("www.", "")
    )

    if "jobs.lever.co" in domain:

        return {
            "ats": "lever",
            "confidence": 1.0,
            "domain": domain,
        }

    if "greenhouse.io" in domain:

        return {
            "ats": "greenhouse",
            "confidence": 1.0,
            "domain": domain,
        }

    if "myworkdayjobs.com" in domain:

        return {
            "ats": "workday",
            "confidence": 1.0,
            "domain": domain,
        }

    return {
        "ats": "unknown",
        "confidence": 0.0,
        "domain": domain,
    }


# ============================================================
# LOAD APPLICATION QUEUE
# ============================================================

def load_queue():

    if not QUEUE_FILE.exists():

        return pd.DataFrame()

    queue = pd.read_csv(
        QUEUE_FILE
    )

    if queue.empty:

        return queue

    required_columns = [
        "job_id",
        "title",
        "company",
        "location",
        "apply_url",
        "match_score",
        "status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in queue.columns
    ]

    if missing_columns:

        raise ValueError(
            "Application queue is missing "
            f"columns: {missing_columns}"
        )

    return queue


# ============================================================
# GET NEXT APPROVED APPLICATION
# ============================================================

def get_next_application(queue):

    if queue.empty:

        return None

    approved = queue[
        queue["status"]
        .astype(str)
        .str.upper()
        .eq(USER_APPROVED)
    ]

    if approved.empty:

        return None

    return approved.iloc[0]


# ============================================================
# UPDATE APPLICATION STATUS
# ============================================================

def update_status(
    job_id,
    status,
):

    try:

        update_application_status(
            QUEUE_FILE,
            job_id,
            status,
        )

        print(
            "\nAPPLICATION STATE UPDATED"
        )

        print(
            f"Job ID:  {job_id}"
        )

        print(
            f"Status:  {status}"
        )

        return True

    except Exception as e:

        print(
            "\nWARNING: Could not update "
            f"application status to {status}:"
        )

        print(e)

        return False


# ============================================================
# LOAD CANDIDATE PROFILE
# ============================================================

def load_candidate_profile():

    if not CANDIDATE_PROFILE_FILE.exists():

        raise FileNotFoundError(
            "\nCandidate profile not found:\n"
            f"{CANDIDATE_PROFILE_FILE}"
        )

    with open(
        CANDIDATE_PROFILE_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        candidate_profile = json.load(
            file
        )

    return candidate_profile


# ============================================================
# INSPECT APPLICATION FORM
# ============================================================

def inspect_form(page):

    print("\n")
    print("=" * 70)
    print("APPLICATION FORM INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    inputs = page.locator(
        "input"
    )

    input_count = inputs.count()

    print(
        f"\nINPUTS FOUND: {input_count}"
    )

    for i in range(
        input_count
    ):

        element = inputs.nth(i)

        try:

            field_type = (
                element.get_attribute(
                    "type"
                )
            )

            name = (
                element.get_attribute(
                    "name"
                )
            )

            field_id = (
                element.get_attribute(
                    "id"
                )
            )

            print(
                f"  [{i}] "
                f"type={field_type} "
                f"name={name} "
                f"id={field_id}"
            )

        except Exception as e:

            print(
                f"  Could not inspect "
                f"input {i}: {e}"
            )

    # --------------------------------------------------------
    # TEXTAREAS
    # --------------------------------------------------------

    textareas = page.locator(
        "textarea"
    )

    print(
        f"\nTEXTAREAS FOUND: "
        f"{textareas.count()}"
    )

    for i in range(
        textareas.count()
    ):

        element = textareas.nth(i)

        try:

            name = (
                element.get_attribute(
                    "name"
                )
            )

            field_id = (
                element.get_attribute(
                    "id"
                )
            )

            print(
                f"  [{i}] "
                f"name={name} "
                f"id={field_id}"
            )

        except Exception as e:

            print(
                f"  Could not inspect "
                f"textarea {i}: {e}"
            )

    # --------------------------------------------------------
    # SELECT BOXES
    # --------------------------------------------------------

    selects = page.locator(
        "select"
    )

    select_count = selects.count()

    print(
        f"\nSELECT BOXES FOUND: "
        f"{select_count}"
    )

    for i in range(
        select_count
    ):

        element = selects.nth(i)

        try:

            name = (
                element.get_attribute(
                    "name"
                )
            )

            field_id = (
                element.get_attribute(
                    "id"
                )
            )

            print(
                f"  [{i}] "
                f"name={name} "
                f"id={field_id}"
            )

        except Exception as e:

            print(
                f"  Could not inspect "
                f"select {i}: {e}"
            )


# ============================================================
# SAVE SCREENSHOT
# ============================================================

def save_screenshot(
    page,
    path,
):

    try:

        page.screenshot(
            path=str(path),
            full_page=True,
        )

        print(
            f"\nScreenshot saved:\n"
            f"{path}"
        )

    except Exception as e:

        print(
            f"\nCould not save screenshot: "
            f"{e}"
        )


# ============================================================
# PRINT FIELD RESULTS
# ============================================================

def print_field_results(
    results,
):

    print("\n")
    print("=" * 70)
    print("FIELD RESULTS")
    print("=" * 70)

    if not results:

        print(
            "\nNo field results returned."
        )

        return

    if isinstance(
        results,
        dict,
    ):

        successful = 0

        total = len(results)

        for field, result in (
            results.items()
        ):

            if isinstance(
                result,
                bool,
            ):

                success = result

            elif isinstance(
                result,
                dict,
            ):

                success = result.get(
                    "success",
                    False,
                )

            else:

                success = bool(
                    result
                )

            if success:

                successful += 1

                print(
                    f"  ✓ {field}"
                )

            else:

                print(
                    f"  - {field}"
                )

        print(
            f"\nHandled: "
            f"{successful}/{total}"
        )

    else:

        print(results)


# ============================================================
# HUMAN REVIEW
# ============================================================

def human_review(
    page,
    job,
):

    print("\n")
    print("=" * 70)
    print("HUMAN REVIEW")
    print("=" * 70)

    print(
        "\nThe application form has been "
        "autofilled where possible."
    )

    print(
        "\nPlease complete the remaining "
        "fields manually."
    )

    print(
        "\nComplete location autocomplete, "
        "CAPTCHA, and any job-specific "
        "questions manually."
    )

    print(
        "\nReview all information carefully."
    )

    print(
        "\nThe automation will NOT click "
        "the Submit button."
    )

    print(
        "\nYou can submit the application "
        "yourself in the browser."
    )

    print(
        "\nThe browser will remain open."
    )

    print(
        "Press ENTER in this terminal "
        "when you are finished."
    )

    try:

        input(
            "\nPress ENTER after you finish "
            "reviewing/submitting..."
        )

    except (
        EOFError,
        KeyboardInterrupt,
    ):

        pass

    return True


# ============================================================
# SUBMISSION OUTCOME
# ============================================================

def confirm_submission(job_id):
    """
    Ask whether the user submitted the application themselves.

    Only an explicit yes records SUBMITTED. Any other answer, or
    leaving the prompt, keeps the application at READY_FOR_REVIEW.
    """

    try:

        answer = input(
            "\nDid you submit this application "
            "yourself in the browser? [y/N]: "
        )

    except (
        EOFError,
        KeyboardInterrupt,
    ):

        answer = ""

    if answer.strip().lower() not in ("y", "yes"):

        print(
            "\nNot recorded as submitted. "
            f"The application stays {READY_FOR_REVIEW}."
        )

        print(
            "You can mark it as submitted "
            "later from the dashboard."
        )

        return False

    try:

        changed, message = mark_submitted(
            QUEUE_FILE,
            job_id,
        )

    except Exception as e:

        print(
            "\nWARNING: Could not record "
            "the submission:"
        )

        print(e)

        return False

    print(
        f"\n{message}"
    )

    return changed


# ============================================================
# PROCESS ONE APPLICATION
# ============================================================

def process_application(job):

    job_id = job["job_id"]

    print("\n")
    print("=" * 70)
    print("NEXT APPLICATION")
    print("=" * 70)

    print(
        f"Job ID:      {job['job_id']}"
    )

    print(
        f"Title:       {job['title']}"
    )

    print(
        f"Company:     {job['company']}"
    )

    print(
        f"Location:    {job['location']}"
    )

    try:

        print(
            f"Match Score: "
            f"{float(job['match_score']):.2%}"
        )

    except Exception:

        print(
            f"Match Score: "
            f"{job['match_score']}"
        )

    print(
        f"Status:      {job['status']}"
    )

    print(
        f"URL:         {job['apply_url']}"
    )

    # ========================================================
    # UPDATE → IN_PROGRESS
    # ========================================================

    update_status(
        job_id,
        IN_PROGRESS,
    )

    # ========================================================
    # ATS DETECTION
    # ========================================================

    ats_info = detect_ats(
        job["apply_url"]
    )

    ats = ats_info["ats"]

    print("\n")
    print("=" * 70)
    print("ATS DETECTION")
    print("=" * 70)

    print(
        f"Domain:     "
        f"{ats_info['domain']}"
    )

    print(
        f"ATS:        "
        f"{ats}"
    )

    print(
        f"Confidence: "
        f"{ats_info['confidence']:.2f}"
    )

    # ========================================================
    # UNKNOWN ATS
    # ========================================================

    if ats == "unknown":

        print(
            "\nUnknown ATS."
        )

        print(
            "No form handler is available "
            "for this website."
        )

        update_status(
            job_id,
            FAILED,
        )

        return

    # ========================================================
    # LOAD CANDIDATE PROFILE
    # ========================================================

    try:

        candidate_profile = (
            load_candidate_profile()
        )

    except Exception as e:

        print(
            "\nERROR loading "
            "candidate profile:"
        )

        print(e)

        update_status(
            job_id,
            FAILED,
        )

        return

    # ========================================================
    # PLAYWRIGHT
    # ========================================================

    with sync_playwright() as p:

        browser = None

        context = None

        page = None

        try:

            # ------------------------------------------------
            # Visible browser
            #
            # IMPORTANT:
            # This worker runs on the Mac host.
            # Do NOT run it inside Docker.
            # ------------------------------------------------

            browser = p.chromium.launch(
                headless=False
            )

            context = browser.new_context(
                viewport={
                    "width": 1440,
                    "height": 1000,
                }
            )

            page = context.new_page()

            # ------------------------------------------------
            # Open application
            # ------------------------------------------------

            print(
                "\nOpening application page..."
            )

            try:

                page.goto(
                    job["apply_url"],
                    wait_until=(
                        "domcontentloaded"
                    ),
                    timeout=60000,
                )

            except PlaywrightTimeoutError:

                print(
                    "\nPage load timed out,"
                )

                print(
                    "but the page may "
                    "still be usable."
                )

            # ------------------------------------------------
            # Dynamic content
            # ------------------------------------------------

            page.wait_for_timeout(
                5000
            )

            print(
                "\nPage loaded."
            )

            print(
                "URL:",
                page.url,
            )

            print(
                "Title:",
                page.title(),
            )

            # ------------------------------------------------
            # Screenshot BEFORE
            # ------------------------------------------------

            save_screenshot(
                page,
                SCREENSHOT_BEFORE,
            )

            # ------------------------------------------------
            # Inspect form
            # ------------------------------------------------

            inspect_form(
                page
            )

            # ------------------------------------------------
            # FORM HANDLER
            # ------------------------------------------------

            print("\n")
            print("=" * 70)
            print("FORM HANDLER")
            print("=" * 70)

            results = (
                handle_application_form(
                    page,
                    ats,
                    candidate_profile,
                )
            )

            # ------------------------------------------------
            # Allow dynamic updates
            # ------------------------------------------------

            page.wait_for_timeout(
                2000
            )

            # ------------------------------------------------
            # Print results
            # ------------------------------------------------

            print_field_results(
                results
            )

            # ------------------------------------------------
            # Screenshot AFTER
            # ------------------------------------------------

            save_screenshot(
                page,
                SCREENSHOT_FILLED,
            )

            # ------------------------------------------------
            # FORM_FILLED
            # ------------------------------------------------

            update_status(
                job_id,
                FORM_FILLED,
            )

            # ------------------------------------------------
            # READY_FOR_REVIEW
            # ------------------------------------------------

            update_status(
                job_id,
                READY_FOR_REVIEW,
            )

            # ------------------------------------------------
            # HUMAN REVIEW
            # ------------------------------------------------

            human_review(
                page,
                job,
            )

            # ------------------------------------------------
            # SUBMISSION OUTCOME
            # ------------------------------------------------

            confirm_submission(
                job_id
            )

            # ------------------------------------------------
            # IMPORTANT
            # ------------------------------------------------

            print("\n")
            print("=" * 70)
            print("AUTOMATION COMPLETE")
            print("=" * 70)

            print(
                "\nForm autofill is complete."
            )

            print(
                "The browser was left for "
                "manual completion."
            )

            print(
                "\nThe automation did NOT:"
            )

            print(
                "  - complete CAPTCHA"
            )

            print(
                "  - click Submit"
            )

            print(
                "  - attempt to submit "
                "the application"
            )

            print(
                "\nYou can now finish the "
                "application manually."
            )

            print(
                "\nClosing browser in "
                f"{BROWSER_CLOSE_DELAY} seconds..."
            )

            try:

                page.wait_for_timeout(
                    BROWSER_CLOSE_DELAY * 1000
                )

            except Exception:

                pass

        except KeyboardInterrupt:

            print(
                "\n\nAutomation interrupted "
                "by user."
            )

        except Exception as e:

            print("\n")
            print("=" * 70)
            print("AUTOMATION ERROR")
            print("=" * 70)

            print(
                f"\n{type(e).__name__}: {e}"
            )

            update_status(
                job_id,
                FAILED,
            )

        finally:

            # ------------------------------------------------
            # Close page
            # ------------------------------------------------

            try:

                if page is not None:

                    page.close()

            except Exception:

                pass

            # ------------------------------------------------
            # Close context
            # ------------------------------------------------

            try:

                if context is not None:

                    context.close()

            except Exception:

                pass

            # ------------------------------------------------
            # Close browser
            # ------------------------------------------------

            try:

                if browser is not None:

                    browser.close()

            except Exception:

                pass

    print(
        "\nBrowser automation finished."
    )


# ============================================================
# CONTINUOUS WORKER
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("AI JOB APPLICATION WORKER")
    print("=" * 70)

    print(
        "\nWorker is running on the Mac host."
    )

    print(
        "Waiting for USER_APPROVED "
        "applications..."
    )

    print(
        f"Checking queue every "
        f"{POLL_INTERVAL} seconds."
    )

    print(
        "\nPress CTRL+C to stop the worker."
    )

    while True:

        try:

            queue = load_queue()

            if queue.empty:

                print(
                    "\nQueue is empty. "
                    "Waiting..."
                )

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            job = get_next_application(
                queue
            )

            if job is None:

                print(
                    "\nNo USER_APPROVED "
                    "applications."
                )

                print(
                    "Waiting for approval..."
                )

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            process_application(
                job
            )

            print(
                "\nReturning to queue "
                "monitor..."
            )

            time.sleep(2)

        except KeyboardInterrupt:

            print(
                "\n\n"
                + "=" * 70
            )

            print(
                "APPLICATION WORKER STOPPED"
            )

            print(
                "=" * 70
            )

            break

        except Exception as e:

            print(
                "\n"
                + "=" * 70
            )

            print(
                "WORKER ERROR"
            )

            print(
                "=" * 70
            )

            print(
                f"\n{type(e).__name__}: {e}"
            )

            print(
                "\nWorker will continue "
                "monitoring the queue."
            )

            time.sleep(
                POLL_INTERVAL
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()