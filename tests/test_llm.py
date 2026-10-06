import pytest
from pydantic import ValidationError

from jobagent.config import BASE_DIR, Settings
from jobagent.llm import (
    LLMNotConfigured,
    RetryingLLM,
    get_structured_llm,
    is_transient,
)
from jobagent.resume.parser import parse_resume_text
from jobagent.resume.schema import ResumeExtraction


class ApiError(Exception):
    """Shaped like a provider HTTP error."""

    def __init__(self, code, message="error"):
        super().__init__(f"{code} {message}")
        self.code = code


class ReadTimeout(Exception):
    pass


class Flaky:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def retrying(runnable, max_retries=2):
    delays = []
    llm = RetryingLLM(
        runnable,
        max_retries=max_retries,
        backoff_seconds=2.0,
        sleep=delays.append,
    )
    return llm, delays


def validation_error():
    try:
        ResumeExtraction.model_validate({"seniority": "wizard"})
    except ValidationError as error:
        return error


# ------------------------------------------------------------
# Classification
# ------------------------------------------------------------

@pytest.mark.parametrize("code", [429, 500, 503, 504])
def test_overload_and_rate_limit_statuses_are_transient(code):
    assert is_transient(ApiError(code))


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_request_and_auth_statuses_are_not_transient(code):
    assert not is_transient(ApiError(code))


def test_timeouts_and_connection_errors_are_transient():
    assert is_transient(ReadTimeout())
    assert is_transient(TimeoutError())
    assert is_transient(ConnectionError())


def test_wrapped_transient_error_is_detected():
    try:
        try:
            raise ApiError(503)
        except ApiError as inner:
            raise RuntimeError("provider call failed") from inner
    except RuntimeError as outer:
        assert is_transient(outer)


def test_config_and_validation_errors_are_not_transient():
    assert not is_transient(LLMNotConfigured("GOOGLE_API_KEY is not set"))
    assert not is_transient(validation_error())
    assert not is_transient(ValueError("could not parse output"))


# ------------------------------------------------------------
# Retry behaviour
# ------------------------------------------------------------

def test_transient_error_is_retried_with_exponential_backoff():
    runnable = Flaky(ApiError(503), ApiError(429), "ok")
    llm, delays = retrying(runnable)

    assert llm.invoke([]) == "ok"
    assert runnable.calls == 3
    assert delays == [2.0, 4.0]


def test_retries_are_capped():
    runnable = Flaky(ApiError(503), ApiError(503), ApiError(503), "ok")
    llm, delays = retrying(runnable)

    with pytest.raises(ApiError):
        llm.invoke([])

    assert runnable.calls == 3
    assert delays == [2.0, 4.0]


@pytest.mark.parametrize(
    "error",
    [ApiError(404, "model not found"), ApiError(400, "bad key")],
)
def test_configuration_errors_are_not_retried(error):
    runnable = Flaky(error, "ok")
    llm, delays = retrying(runnable)

    with pytest.raises(ApiError):
        llm.invoke([])

    assert runnable.calls == 1
    assert delays == []


def test_validation_errors_are_not_retried():
    runnable = Flaky(validation_error(), "ok")
    llm, delays = retrying(runnable)

    with pytest.raises(ValidationError):
        llm.invoke([])

    assert runnable.calls == 1


# ------------------------------------------------------------
# Through the parser
# ------------------------------------------------------------

def test_exhausted_retries_fall_back_with_the_real_reason(
    sample_text, settings
):
    runnable = Flaky(
        ApiError(503, "UNAVAILABLE high demand"),
        ApiError(503, "UNAVAILABLE high demand"),
        ApiError(503, "UNAVAILABLE high demand"),
    )
    llm, delays = retrying(runnable)

    profile = parse_resume_text(
        sample_text, settings=settings, extractor=llm
    )

    assert profile.metadata.method == "fallback"
    assert "503 UNAVAILABLE" in profile.metadata.fallback_reason
    assert runnable.calls == 3
    assert profile.contact.email == "asha.rao@example.com"


def test_recovery_after_a_transient_error_uses_the_llm(
    sample_text, settings, extraction_payload
):
    llm, _ = retrying(Flaky(ApiError(503), extraction_payload))

    profile = parse_resume_text(
        sample_text, settings=settings, extractor=llm
    )

    assert profile.metadata.method == "llm"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

def test_get_structured_llm_applies_retry_settings():
    settings = Settings(
        _env_file=None,
        llm_provider="google",
        llm_model="test-model",
        google_api_key="not-a-real-key",
        llm_max_retries=1,
        llm_retry_backoff_seconds=0.5,
    )

    llm = get_structured_llm(ResumeExtraction, settings)

    assert isinstance(llm, RetryingLLM)
    assert llm.max_retries == 1
    assert llm.backoff_seconds == 0.5


def test_missing_model_name_is_a_configuration_error():
    settings = Settings(
        _env_file=None,
        llm_provider="google",
        llm_model=None,
        google_api_key="not-a-real-key",
    )

    with pytest.raises(LLMNotConfigured, match="JOBAGENT_LLM_MODEL"):
        get_structured_llm(ResumeExtraction, settings)


def test_relative_paths_resolve_against_the_project_root():
    settings = Settings(
        _env_file=None,
        resume_file="data/some-resume.pdf",
    )

    assert settings.resume_file == BASE_DIR / "data" / "some-resume.pdf"
