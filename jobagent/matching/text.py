"""Job description text: HTML cleaning, section splitting and chunking."""

import html
import re
from dataclasses import dataclass, field

from jobagent.matching.taxonomy import Taxonomy

REQUIRED = "required"
NICE_TO_HAVE = "nice_to_have"
NEUTRAL = "neutral"
IGNORE = "ignore"

# Checked in this order, so "Preferred qualifications" is nice-to-have
# rather than required.
_HEADING_ORDER = (NICE_TO_HAVE, IGNORE, REQUIRED, NEUTRAL)

_BLOCK_TAGS = re.compile(
    r"</?(?:p|div|br|ul|ol|h[1-6]|tr|td|table|section|article|hr)\b[^>]*>",
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(r"<li\b[^>]*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")

BULLET = "- "


def clean_html(raw: str) -> str:
    """HTML to plain text, one block or list item per line."""

    if not raw:
        return ""

    text = _LIST_ITEM.sub(f"\n{BULLET}", str(raw))
    text = _BLOCK_TAGS.sub("\n", text)
    text = re.sub(r"</li\s*>", "\n", text, flags=re.IGNORECASE)
    text = _ANY_TAG.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("​", "")

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.split("\n")
    ]

    return "\n".join(
        line
        for line in lines
        if line and line != BULLET.strip()
    )


def _heading_kind(line: str, taxonomy: Taxonomy):
    """Section kind if ``line`` looks like a known heading, else None."""

    if (
        line.startswith(BULLET)
        or len(line) > 90
        or len(line.split()) > 12
        or line.rstrip().endswith(".")
    ):
        return None

    for kind in _HEADING_ORDER:
        for pattern in taxonomy.section_headings.get(kind, []):
            if pattern.search(line):
                return kind

    return None


@dataclass
class JobText:
    """A job description split by how binding each part is."""

    lines: dict[str, list[str]] = field(
        default_factory=lambda: {
            REQUIRED: [],
            NICE_TO_HAVE: [],
            NEUTRAL: [],
        }
    )
    has_sections: bool = False

    def text(self, kind: str) -> str:
        return "\n".join(self.lines[kind])

    @property
    def relevant_lines(self) -> list[str]:
        """Everything that describes the job, without company boilerplate."""

        return (
            self.lines[REQUIRED]
            + self.lines[NEUTRAL]
            + self.lines[NICE_TO_HAVE]
        )

    @property
    def relevant_text(self) -> str:
        return "\n".join(self.relevant_lines)


def split_sections(text: str, taxonomy: Taxonomy) -> JobText:
    """Assign each line to required, nice-to-have or neutral.

    Text under an ignored heading (company blurb, perks) is dropped, and so
    is the introduction before the first heading. When no heading is
    recognised the description does not distinguish requirements, so every
    line is neutral.
    """

    job_text = JobText()
    lines = [line for line in text.split("\n") if line.strip()]

    kinds = [_heading_kind(line, taxonomy) for line in lines]
    job_text.has_sections = any(
        kind in (REQUIRED, NICE_TO_HAVE, NEUTRAL)
        for kind in kinds
    )

    current = IGNORE if job_text.has_sections else NEUTRAL

    for line, kind in zip(lines, kinds):
        if kind is not None:
            current = kind

        target = current

        if (
            target in (REQUIRED, NEUTRAL)
            and kind is None
            and taxonomy.inline_nice_to_have is not None
            and taxonomy.inline_nice_to_have.search(line)
        ):
            target = NICE_TO_HAVE

        if target != IGNORE:
            job_text.lines[target].append(line)

    return job_text


def chunk_lines(
    lines: list[str],
    max_words: int,
    overlap: int = 0,
) -> list[str]:
    """Pack lines into chunks of at most ``max_words`` words.

    ``overlap`` words are repeated between consecutive chunks so a
    requirement split across a boundary is still seen whole once.
    """

    words: list[str] = []
    for line in lines:
        words.extend(line.split())

    if not words:
        return []

    step = max(1, max_words - overlap)
    chunks = []

    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break

    return chunks
