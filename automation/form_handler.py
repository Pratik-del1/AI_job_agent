from pathlib import Path

from playwright.sync_api import Page


# ============================================================
# GENERIC HELPERS
# ============================================================

def clean_value(value):
    """
    Convert a value into a clean string.

    Empty / None / NaN values become "".
    """

    if value is None:
        return ""

    try:
        if str(value).lower() == "nan":
            return ""
    except Exception:
        pass

    return str(value).strip()


def field_exists(
    page: Page,
    selector: str
):
    """
    Check whether a field exists on the page.
    """

    try:

        return page.locator(
            selector
        ).count() > 0

    except Exception:

        return False


# ============================================================
# GENERIC TEXT FIELD
# ============================================================

def fill_text_field(
    page: Page,
    selector: str,
    value,
    label: str
):
    """
    Fill a normal text-like input.
    """

    value = clean_value(value)

    if not value:

        print(
            f"SKIPPED {label}: "
            "no value provided"
        )

        return False

    if not field_exists(
        page,
        selector
    ):

        print(
            f"SKIPPED {label}: "
            "field not found"
        )

        return False

    try:

        field = page.locator(
            selector
        ).first

        if not field.is_visible():

            print(
                f"SKIPPED {label}: "
                "field not visible"
            )

            return False

        field.fill(value)

        print(
            f"FILLED {label}: {value}"
        )

        return True

    except Exception as e:

        print(
            f"FAILED {label}: {e}"
        )

        return False


# ============================================================
# FILE UPLOAD
# ============================================================

def upload_file(
    page: Page,
    selector: str,
    file_path,
    label: str
):
    """
    Upload a file to a file input.
    """

    file_path = clean_value(
        file_path
    )

    if not file_path:

        print(
            f"SKIPPED {label}: "
            "no file provided"
        )

        return False

    path = Path(
        file_path
    )

    if not path.exists():

        print(
            f"FAILED {label}: "
            f"file does not exist:\n{path}"
        )

        return False

    if not field_exists(
        page,
        selector
    ):

        print(
            f"SKIPPED {label}: "
            "file input not found"
        )

        return False

    try:

        page.locator(
            selector
        ).first.set_input_files(
            str(path)
        )

        print(
            f"UPLOADED {label}: "
            f"{path.name}"
        )

        return True

    except Exception as e:

        print(
            f"FAILED {label}: {e}"
        )

        return False


# ============================================================
# LEVER HANDLER
# ============================================================

def fill_lever_form(
    page: Page,
    candidate_profile
):
    """
    Handle common Lever application fields.

    This function deliberately does NOT submit
    the application.
    """

    print("\n")
    print("=" * 70)
    print("LEVER FORM HANDLER")
    print("=" * 70)

    results = {}

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    results["resume"] = upload_file(
        page,
        'input[type="file"][name="resume"]',
        candidate_profile.get(
            "resume_path"
        ),
        "Resume"
    )

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    results["name"] = fill_text_field(
        page,
        'input[name="name"]',
        candidate_profile.get(
            "name"
        ),
        "Name"
    )

    # --------------------------------------------------------
    # Email
    # --------------------------------------------------------

    results["email"] = fill_text_field(
        page,
        'input[name="email"]',
        candidate_profile.get(
            "email"
        ),
        "Email"
    )

    # --------------------------------------------------------
    # Phone
    # --------------------------------------------------------

    results["phone"] = fill_text_field(
        page,
        'input[name="phone"]',
        candidate_profile.get(
            "phone"
        ),
        "Phone"
    )

    # --------------------------------------------------------
    # Location
    #
    # Intentionally skipped for now because Lever
    # uses an autocomplete component rather than
    # a simple text field.
    # --------------------------------------------------------

    print(
        "SKIPPED Location: "
        "autocomplete handler not implemented yet"
    )

    results["location"] = False

    # --------------------------------------------------------
    # Organization
    # --------------------------------------------------------

    results["organization"] = fill_text_field(
        page,
        'input[name="org"]',
        candidate_profile.get(
            "org"
        ),
        "Organization"
    )

    # --------------------------------------------------------
    # LinkedIn
    # --------------------------------------------------------

    results["linkedin"] = fill_text_field(
        page,
        'input[name="urls[LinkedIn]"]',
        candidate_profile.get(
            "linkedin_url"
        ),
        "LinkedIn"
    )

    # --------------------------------------------------------
    # GitHub
    # --------------------------------------------------------

    results["github"] = fill_text_field(
        page,
        'input[name="urls[GitHub]"]',
        candidate_profile.get(
            "github_url"
        ),
        "GitHub"
    )

    # --------------------------------------------------------
    # Portfolio
    # --------------------------------------------------------

    results["portfolio"] = fill_text_field(
        page,
        'input[name="urls[Portfolio]"]',
        candidate_profile.get(
            "portfolio_url"
        ),
        "Portfolio"
    )

    # --------------------------------------------------------
    # Twitter
    # --------------------------------------------------------

    results["twitter"] = fill_text_field(
        page,
        'input[name="urls[Twitter]"]',
        candidate_profile.get(
            "twitter_url"
        ),
        "Twitter"
    )

    # --------------------------------------------------------
    # Other
    # --------------------------------------------------------

    results["other"] = fill_text_field(
        page,
        'input[name="urls[Other]"]',
        candidate_profile.get(
            "other_url"
        ),
        "Other URL"
    )

    return results


# ============================================================
# GREENHOUSE PLACEHOLDER
# ============================================================

def fill_greenhouse_form(
    page: Page,
    candidate_profile
):
    """
    Greenhouse handler.

    We will implement this after inspecting
    an actual Greenhouse application form.
    """

    print("\n")
    print("=" * 70)
    print("GREENHOUSE FORM HANDLER")
    print("=" * 70)

    print(
        "Greenhouse handler not implemented yet."
    )

    return {}


# ============================================================
# WORKDAY PLACEHOLDER
# ============================================================

def fill_workday_form(
    page: Page,
    candidate_profile
):
    """
    Workday handler.

    We will implement this after inspecting
    an actual Workday application form.
    """

    print("\n")
    print("=" * 70)
    print("WORKDAY FORM HANDLER")
    print("=" * 70)

    print(
        "Workday handler not implemented yet."
    )

    return {}


# ============================================================
# UNKNOWN ATS
# ============================================================

def handle_unknown_form(
    page: Page,
    candidate_profile
):
    """
    Generic fallback for unknown application systems.
    """

    print("\n")
    print("=" * 70)
    print("UNKNOWN ATS")
    print("=" * 70)

    print(
        "No ATS-specific handler is available."
    )

    print(
        "Form will remain untouched."
    )

    return {}


# ============================================================
# ROUTER
# ============================================================

def handle_application_form(
    page: Page,
    ats: str,
    candidate_profile
):
    """
    Route the application to the correct ATS handler.
    """

    ats = clean_value(
        ats
    ).lower()

    if ats == "lever":

        return fill_lever_form(
            page,
            candidate_profile
        )

    if ats == "greenhouse":

        return fill_greenhouse_form(
            page,
            candidate_profile
        )

    if ats == "workday":

        return fill_workday_form(
            page,
            candidate_profile
        )

    return handle_unknown_form(
        page,
        candidate_profile
    )