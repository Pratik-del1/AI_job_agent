"""Single place where an LLM client is created, so the provider stays a setting."""

import logging
import time
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ValidationError

from jobagent.config import Settings

logger = logging.getLogger(__name__)

# HTTP statuses worth another attempt: timeout, rate limit, server errors.
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class LLMNotConfigured(RuntimeError):
    """Raised when the configured provider cannot be used (e.g. no API key)."""


def _status_code(error: BaseException) -> Optional[int]:
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value

    value = getattr(
        getattr(error, "response", None),
        "status_code",
        None,
    )
    return value if isinstance(value, int) else None


def is_transient(error: BaseException) -> bool:
    """True for overload, rate-limit, timeout and connection errors.

    Configuration and request errors (bad key, unknown model, rejected
    schema) and invalid model output are not transient.
    """

    seen = set()

    while error is not None and id(error) not in seen:
        seen.add(id(error))

        if isinstance(error, (LLMNotConfigured, ValidationError)):
            return False

        status = _status_code(error)
        if status is not None:
            return status in TRANSIENT_STATUS_CODES

        name = type(error).__name__
        if (
            isinstance(error, (TimeoutError, ConnectionError))
            or "Timeout" in name
            or "RateLimit" in name
            or "ConnectionError" in name
        ):
            return True

        # Provider libraries often wrap the HTTP error.
        error = error.__cause__

    return False


class RetryingLLM:
    """Wraps a runnable and retries transient failures with exponential backoff."""

    def __init__(
        self,
        runnable: Any,
        *,
        max_retries: int,
        backoff_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.runnable = runnable
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep

    def invoke(self, messages: Any) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return self.runnable.invoke(messages)

            except Exception as error:
                if (
                    attempt == self.max_retries
                    or not is_transient(error)
                ):
                    raise

                delay = self.backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Transient LLM error (%s); retry %d/%d in %.0fs",
                    type(error).__name__,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                self.sleep(delay)

        raise AssertionError("unreachable")


# The clients' own retries are switched off below so RetryingLLM is the
# only retry layer.

def _google(settings: Settings, schema: Type[BaseModel]) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0,
        google_api_key=settings.google_api_key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
        # 1 means a single attempt; 0 would mean the Google SDK default.
        max_retries=1,
    )

    # Gemini's native structured output.
    return llm.with_structured_output(
        schema,
        method="json_schema",
    )


def _openai(settings: Settings, schema: Type[BaseModel]) -> Any:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )

    # Tool calling tolerates optional fields and defaults in the schema.
    return llm.with_structured_output(
        schema,
        method="function_calling",
    )


# provider -> (settings field holding the key, env var to name in errors,
#              pip package, builder)
PROVIDERS = {
    "google": (
        "google_api_key",
        "GOOGLE_API_KEY",
        "langchain-google-genai",
        _google,
    ),
    "openai": (
        "openai_api_key",
        "OPENAI_API_KEY",
        "langchain-openai",
        _openai,
    ),
}


def get_structured_llm(
    schema: Type[BaseModel],
    settings: Settings,
) -> Any:
    """Return a runnable whose ``invoke(messages)`` yields a ``schema`` instance."""

    provider = settings.llm_provider.strip().lower()

    if provider not in PROVIDERS:
        raise LLMNotConfigured(
            f"Unsupported LLM provider: {settings.llm_provider!r}. "
            f"Supported: {sorted(PROVIDERS)}"
        )

    key_field, key_env, package, build = PROVIDERS[provider]

    if getattr(settings, key_field) is None:
        raise LLMNotConfigured(
            f"{key_env} is not set"
        )

    if not settings.llm_model:
        raise LLMNotConfigured(
            "JOBAGENT_LLM_MODEL is not set"
        )

    # Provider packages are imported lazily so only the one in use is needed.
    try:
        runnable = build(settings, schema)
    except ImportError as error:
        raise LLMNotConfigured(
            f"Provider package is not installed: pip install {package}"
        ) from error

    return RetryingLLM(
        runnable,
        max_retries=settings.llm_max_retries,
        backoff_seconds=settings.llm_retry_backoff_seconds,
    )
