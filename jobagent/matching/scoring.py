"""Combines component scores into an overall score, label and explanation."""

from typing import Optional

from pydantic import BaseModel, Field

COMPONENTS = ("skill", "role", "experience", "semantic", "preference")

COMPONENT_LABELS = {
    "skill": "Skill Match",
    "role": "Role Match",
    "experience": "Experience Match",
    "semantic": "Semantic Match",
    "preference": "Preference Match",
}

# Best first. A cap can only move a label later in this list.
LABELS = ("HIGH PRIORITY", "GOOD MATCH", "CONSIDER", "LOW MATCH")


class JobMatch(BaseModel):
    """One job's score with everything needed to explain it."""

    overall_score: float
    skill_score: Optional[float] = None
    role_score: Optional[float] = None
    experience_score: Optional[float] = None
    semantic_score: Optional[float] = None
    preference_score: Optional[float] = None

    matched_skills: list[str] = Field(default_factory=list)
    missing_required_skills: list[str] = Field(default_factory=list)
    missing_nice_to_have_skills: list[str] = Field(default_factory=list)
    # Missing skills from text that does not say required or optional.
    missing_other_skills: list[str] = Field(default_factory=list)

    role_family: Optional[str] = None
    role_match_type: str = "unavailable"
    experience_required: Optional[float] = None
    experience_required_source: Optional[str] = None
    experience_candidate: Optional[float] = None

    recommendation: str
    label_note: Optional[str] = None

    # Effective weights after redistribution; they sum to 1.
    weights_used: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""

    def component_scores(self) -> dict[str, Optional[float]]:
        return {
            name: getattr(self, f"{name}_score")
            for name in COMPONENTS
        }

    def contributions(self) -> dict[str, float]:
        """Points each component adds to the overall score."""

        scores = self.component_scores()
        return {
            name: weight * scores[name]
            for name, weight in self.weights_used.items()
        }


def combine(
    scores: dict[str, Optional[float]],
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Weighted mean over the components that have a score.

    An unavailable component's weight is spread over the others in
    proportion to their own weights. Returns the overall score and the
    effective weights.
    """

    available = {
        name: weights[name]
        for name in COMPONENTS
        if scores.get(name) is not None and weights.get(name, 0) > 0
    }

    total = sum(available.values())
    if total <= 0:
        return 0.0, {}

    effective = {name: weight / total for name, weight in available.items()}

    overall = sum(
        effective[name] * scores[name]
        for name in effective
    )

    return float(overall), effective


def label_for(
    score: float,
    high_priority: float,
    good_match: float,
    consider: float,
) -> str:
    if score >= high_priority:
        return "HIGH PRIORITY"
    if score >= good_match:
        return "GOOD MATCH"
    if score >= consider:
        return "CONSIDER"
    return "LOW MATCH"


def cap_label(label: str, cap: Optional[str]) -> str:
    """The weaker of ``label`` and ``cap``. Unknown or empty caps do nothing."""

    if not cap or cap not in LABELS:
        return label

    return LABELS[max(LABELS.index(label), LABELS.index(cap))]


def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{round(value * 100)}%"


def _years(value: float) -> str:
    return f"{value:g}"


def build_explanation(match: JobMatch, notes: dict[str, str]) -> str:
    """Plain-text breakdown built only from the computed fields."""

    lines = [f"Overall Match: {_percent(match.overall_score)}"]

    scores = match.component_scores()
    for name in COMPONENTS:
        line = f"- {COMPONENT_LABELS[name]}: {_percent(scores[name])}"
        if notes.get(name):
            line += f" ({notes[name]})"
        lines.append(line)

    def section(title, items):
        if items:
            lines.append("")
            lines.append(f"{title}:")
            lines.extend(f"- {item}" for item in items)

    section("Matched Skills", match.matched_skills)
    section("Missing Required Skills", match.missing_required_skills)
    section("Missing Nice-to-Have Skills", match.missing_nice_to_have_skills)
    section("Missing Skills (requirement level not stated)",
            match.missing_other_skills)

    unavailable = [
        COMPONENT_LABELS[name]
        for name in COMPONENTS
        if scores[name] is None
    ]
    if unavailable:
        lines.append("")
        lines.append(
            "Not scored (weight shared among the other components): "
            + ", ".join(unavailable)
        )

    if match.label_note:
        lines.append("")
        lines.append(f"Label: {match.recommendation}. {match.label_note}")

    return "\n".join(lines)
