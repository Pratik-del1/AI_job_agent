"""Resume text to a validated ``ResumeProfile``.

The LLM fills ``ResumeExtraction``. Invalid output is retried once with the
validation error attached; anything else falls back to regex parsing so a
profile is always returned.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from jobagent.config import Settings, get_settings
from jobagent.llm import LLMNotConfigured, get_structured_llm
from jobagent.resume.enrich import add_usage_evidence
from jobagent.resume.fallback import fallback_extraction, split_sections
from jobagent.resume.loader import load_resume_text
from jobagent.resume.schema import (
    ParseMetadata,
    ResumeExtraction,
    ResumeProfile,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2

SYSTEM_PROMPT = """\
You extract structured data from a candidate's resume.

General rules:
- Use only information present in the resume. Never invent employers, dates,
  degrees, skills or metrics. Leave a field empty when the resume does not
  give it.
- Copy names, dates and numbers as written.

Experience and projects:
- highlights: every bullet point of the entry, as written. Do not drop or
  merge bullets.
- A project's description is one sentence saying what the project is.
- technologies: every tool, library, framework or language the entry names,
  whether in a tech-stack line or inside a bullet.
- outcomes: each measured result or scale figure in the bullets, as a short
  phrase that keeps the number and what it measures: accuracy and other
  metrics, dataset sizes, number of classes, users, volumes.

Skills:
- One entry per skill the candidate lists as a skill, with a category.
- A technology that appears only inside a role or project belongs in that
  entry's technologies, not in skills.
- evidence: 'skills section' if listed there, plus the employer name of
  every role and the name of every project that names the skill or clearly
  applies it, including under another wording (for example 'EDA' and
  'exploratory data analysis').

Education:
- institution: the full name as written. Keep a campus or city that the
  resume gives alongside the name as part of it, and also put it in
  location.

Experience level:
- total_years_experience counts professional work only, not education or
  personal projects.
- seniority is the level of role the candidate should now apply for. Use
  'intern' only for someone still studying. A candidate who has finished
  their degree and has only internship experience is 'entry'.

Preferences:
- A role the candidate uses to describe themselves in a summary, objective
  or headline is a preferred role with source 'stated'. So is any role,
  location or work mode the resume says they are seeking.
- Otherwise infer preferred_roles from the candidate's skills and background
  and mark them 'inferred'.
- Do not guess locations, work modes or industries that the resume gives no
  basis for.
"""


def _build_messages(text: str) -> list[tuple[str, str]]:
    return [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            f"Today's date: {date.today():%B %Y}\n\n"
            f"Resume:\n\n{text}",
        ),
    ]


def _extract_with_llm(
    extractor: Any,
    text: str,
) -> ResumeExtraction:
    messages = _build_messages(text)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = extractor.invoke(messages)

            if isinstance(result, ResumeExtraction):
                result = result.model_dump()

            return ResumeExtraction.model_validate(result)

        # Schema violations and unparseable output (langchain raises a
        # ValueError subclass) are worth one retry; other errors are not.
        except (ValidationError, ValueError) as error:
            logger.warning(
                "Resume extraction attempt %d/%d was invalid: %s",
                attempt,
                MAX_ATTEMPTS,
                error,
            )
            if attempt == MAX_ATTEMPTS:
                raise
            messages = _build_messages(text) + [
                (
                    "human",
                    "Your previous answer did not match the schema:\n"
                    f"{error}\n"
                    "Return the extraction again with these problems fixed.",
                )
            ]

    raise AssertionError("unreachable")


def parse_resume_text(
    text: str,
    *,
    settings: Optional[Settings] = None,
    extractor: Any = None,
    use_llm: bool = True,
    source_file: Optional[str] = None,
) -> ResumeProfile:
    """Parse resume text.

    ``extractor`` is anything with ``invoke(messages)`` returning a
    ``ResumeExtraction`` or a dict; it defaults to the configured LLM.
    """

    settings = settings or get_settings()

    extraction = None
    fallback_reason = None

    if not use_llm:
        fallback_reason = "LLM disabled"
    else:
        try:
            if extractor is None:
                extractor = get_structured_llm(
                    ResumeExtraction,
                    settings,
                )
            extraction = _extract_with_llm(extractor, text)

        except LLMNotConfigured as error:
            fallback_reason = str(error)

        except Exception as error:
            fallback_reason = f"{type(error).__name__}: {error}"
            logger.warning(
                "LLM resume extraction failed, using regex fallback: %s",
                fallback_reason,
            )

    if extraction is None:
        extraction = fallback_extraction(text)

    extraction = add_usage_evidence(extraction)

    return ResumeProfile(
        **extraction.model_dump(),
        raw_sections=split_sections(text),
        metadata=ParseMetadata(
            method="fallback" if fallback_reason else "llm",
            model=None if fallback_reason else settings.llm_model,
            parsed_at=datetime.now().replace(microsecond=0),
            source_file=source_file,
            fallback_reason=fallback_reason,
        ),
    )


def parse_resume_file(
    path: Optional[Path] = None,
    *,
    settings: Optional[Settings] = None,
    extractor: Any = None,
    use_llm: bool = True,
) -> ResumeProfile:
    settings = settings or get_settings()

    path = Path(path) if path else settings.resume_file

    return parse_resume_text(
        load_resume_text(path),
        settings=settings,
        extractor=extractor,
        use_llm=use_llm,
        source_file=path.name,
    )
