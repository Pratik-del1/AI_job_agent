"""Role matching through normalised role families."""

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from jobagent.matching.taxonomy import Taxonomy


def normalize_title(title: str, taxonomy: Taxonomy) -> str:
    """Lowercased title without parentheticals, seniority words or dashes."""

    text = re.sub(r"\([^)]*\)", " ", str(title or "").lower())

    for level in taxonomy.seniority_levels:
        text = level.pattern.sub(" ", text)

    text = re.sub(r"[—–\-,|:.&]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def title_families(title: str, taxonomy: Taxonomy) -> list[str]:
    """Role families named in a title, in order of appearance."""

    text = normalize_title(title, taxonomy)

    hits = []
    for family in taxonomy.role_families:
        match = family.pattern.search(text)
        if match:
            # Earlier wins; at the same position the longer alias wins.
            hits.append((match.start(), -len(match.group(0)), family.id))

    return [family_id for _, _, family_id in sorted(hits)]


@dataclass
class RoleResult:
    score: Optional[float]
    family: Optional[str] = None
    other_families: list[str] = field(default_factory=list)
    match_type: str = "unavailable"
    target: Optional[str] = None


def score_role(
    title: str,
    target_roles: list[str],
    taxonomy: Taxonomy,
    title_similarity: Optional[Callable[[str, list[str]], Optional[float]]] = None,
    fallback_cap: float = 0.7,
) -> RoleResult:
    """Score a job title against the roles the candidate wants.

    - Same family as a target role: 1.0 ("exact").
    - A related family: the affinity from the taxonomy ("related").
    - A second family in the title ("Backend Engineer - AI Agents") pulls
      the score halfway toward its own affinity when that is higher.
    - No family recognised on either side: similarity of the title text to
      the target role names, capped at ``fallback_cap`` ("semantic").
    """

    if not target_roles:
        return RoleResult(score=None)

    families = title_families(title, taxonomy)

    targets = []
    unmapped_targets = []
    for role in target_roles:
        role_families = title_families(role, taxonomy)
        if role_families:
            targets.append((role, role_families[0]))
        else:
            unmapped_targets.append(role)

    if families and targets:
        def best(family):
            return max(
                (taxonomy.role_affinity(target_family, family), role)
                for role, target_family in targets
            )

        score, target = best(families[0])

        if len(families) > 1:
            other_score, other_target = max(
                best(family) for family in families[1:]
            )
            if other_score > score:
                score = (score + other_score) / 2
                target = other_target

        if score >= 1.0:
            match_type = "exact"
        elif score > taxonomy.default_affinity:
            match_type = "related"
        else:
            match_type = "unrelated"

        return RoleResult(
            score=score,
            family=families[0],
            other_families=families[1:],
            match_type=match_type,
            target=target,
        )

    # Unknown title, or target roles the taxonomy does not know.
    if title_similarity is None:
        return RoleResult(
            score=None,
            family=families[0] if families else None,
            other_families=families[1:],
        )

    similarity = title_similarity(
        normalize_title(title, taxonomy),
        [normalize_title(role, taxonomy) for role in target_roles],
    )

    if similarity is None:
        return RoleResult(
            score=None,
            family=families[0] if families else None,
            other_families=families[1:],
        )

    return RoleResult(
        score=min(similarity, fallback_cap),
        family=families[0] if families else None,
        other_families=families[1:],
        match_type="semantic",
    )
