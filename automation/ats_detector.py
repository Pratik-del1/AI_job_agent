from urllib.parse import urlparse


# ============================================================
# ATS DOMAIN RULES
# ============================================================

ATS_RULES = {
    "lever": [
        "jobs.lever.co",
        "jobs.eu.lever.co",
    ],

    "greenhouse": [
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
    ],

    "workday": [
        "myworkdayjobs.com",
    ],
}


# ============================================================
# DETECT ATS
# ============================================================

def detect_ats(url):
    """
    Detect the ATS/platform from a job application URL.

    Returns:
        {
            "ats": str,
            "confidence": float,
            "domain": str
        }
    """

    if not url:

        return {
            "ats": "unknown",
            "confidence": 0.0,
            "domain": "",
        }

    parsed = urlparse(
        str(url).strip()
    )

    domain = parsed.netloc.lower()

    # Remove www.
    if domain.startswith("www."):
        domain = domain[4:]

    # --------------------------------------------------------
    # Check known ATS platforms
    # --------------------------------------------------------

    for ats, domains in ATS_RULES.items():

        for known_domain in domains:

            if (
                domain == known_domain
                or domain.endswith(
                    "." + known_domain
                )
            ):

                return {
                    "ats": ats,
                    "confidence": 1.0,
                    "domain": domain,
                }

    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    return {
        "ats": "unknown",
        "confidence": 0.0,
        "domain": domain,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_urls = [

        "https://jobs.lever.co/company/job-id",

        "https://boards.greenhouse.io/company/jobs/123",

        "https://company.myworkdayjobs.com/en-US/careers",

        "https://example.com/careers/job",
    ]

    for url in test_urls:

        result = detect_ats(url)

        print(
            f"\nURL: {url}"
        )

        print(
            f"ATS: {result['ats']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']}"
        )

        print(
            f"Domain: {result['domain']}"
        )