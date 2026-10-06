"""User-written job preferences, which override anything read from the resume.

``data/preferences.json`` is optional and looks like::

    {
      "preferred_roles": ["Machine Learning Engineer", "AI Engineer"],
      "preferred_locations": ["Bengaluru", "Remote"],
      "work_modes": ["hybrid"],
      "excluded_roles": ["Sales"]
    }
"""

import json
from pathlib import Path

from jobagent.resume.schema import JobPreferences, Preference, ResumeProfile

PREFERENCE_FIELDS = tuple(JobPreferences.model_fields)


def load_user_preferences(path: Path) -> dict[str, list[str]]:
    path = Path(path)

    if not path.exists():
        return {}

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name} must contain a JSON object"
        )

    unknown = sorted(set(data) - set(PREFERENCE_FIELDS))
    if unknown:
        raise ValueError(
            f"Unknown keys in {path.name}: {unknown}. "
            f"Allowed: {list(PREFERENCE_FIELDS)}"
        )

    for key, values in data.items():
        if (
            not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
        ):
            raise ValueError(
                f"{path.name}: '{key}' must be a list of strings"
            )

    return data


def apply_user_preferences(
    profile: ResumeProfile,
    overrides: dict[str, list[str]],
) -> ResumeProfile:
    """Replace each preference field the user supplied; leave the rest."""

    if not overrides:
        return profile

    preferences = profile.preferences.model_copy(
        update={
            key: [
                Preference(value=value.strip(), source="user")
                for value in values
                if value.strip()
            ]
            for key, values in overrides.items()
        }
    )

    return profile.model_copy(
        update={"preferences": preferences}
    )
