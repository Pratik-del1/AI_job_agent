def build_application_data(candidate_profile):
    """
    Build deterministic application data from the
    structured candidate profile.

    Only uses information explicitly present in
    the candidate profile.
    """

    application_data = {
        "name": candidate_profile.get(
            "name",
            ""
        ),

        "email": candidate_profile.get(
            "email",
            ""
        ),

        "phone": candidate_profile.get(
            "phone",
            ""
        ),

        "location": candidate_profile.get(
            "location",
            ""
        ),

        "organization": candidate_profile.get(
            "organization",
            ""
        ),

        "linkedin": candidate_profile.get(
            "linkedin_url",
            ""
        ),

        "github": candidate_profile.get(
            "github_url",
            ""
        ),

        "portfolio": candidate_profile.get(
            "portfolio_url",
            ""
        ),

        "twitter": candidate_profile.get(
            "twitter_url",
            ""
        ),

        "other": candidate_profile.get(
            "other_url",
            ""
        ),

        "resume_path": candidate_profile.get(
            "resume_path",
            ""
        ),
    }

    return application_data


def print_application_data(application_data):

    print("\n")
    print("=" * 70)
    print("APPLICATION DATA")
    print("=" * 70)

    for field, value in application_data.items():

        if value:
            print(
                f"{field:15} → {value}"
            )

        else:
            print(
                f"{field:15} → [NOT PROVIDED]"
            )