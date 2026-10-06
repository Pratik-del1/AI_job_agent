"""Parse the resume and write the structured profile.

Run from the project root::

    python -m jobagent.resume.cli
"""

import argparse
import logging
from pathlib import Path

from jobagent.config import get_settings
from jobagent.resume.parser import parse_resume_file
from jobagent.resume.preferences import (
    apply_user_preferences,
    load_user_preferences,
)


def main(argv=None) -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Extract a structured profile from a resume PDF."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=settings.resume_file,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.resume_profile_file,
    )
    parser.add_argument(
        "--preferences",
        type=Path,
        default=settings.preferences_file,
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use regex parsing only.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    profile = parse_resume_file(
        args.resume,
        settings=settings,
        use_llm=not args.no_llm,
    )

    profile = apply_user_preferences(
        profile,
        load_user_preferences(args.preferences),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        profile.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(f"Parsed with: {profile.metadata.method}")
    if profile.metadata.fallback_reason:
        print(f"Fallback reason: {profile.metadata.fallback_reason}")
    print(f"Skills: {len(profile.skills)}")
    print(f"Experience entries: {len(profile.experience)}")
    print(f"Education entries: {len(profile.education)}")
    print(f"Projects: {len(profile.projects)}")
    print(f"Technologies: {len(profile.technologies)}")
    print(
        "Preferred roles: "
        + (
            ", ".join(
                item.value
                for item in profile.preferences.preferred_roles
            )
            or "none"
        )
    )
    print(f"Saved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
