"""Structured Gemini/Groq generation with explicit provider fallback."""

import asyncio
import json
from typing import Any, Literal, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from incident_copilot.config import Settings


class TriageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    signals: list[str]


class DiagnosisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause_code: Literal[
        "database_pool_saturation",
        "compute_saturation",
        "elevated_error_rate",
        "unknown",
    ]
    likely_cause: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[str]


class RunbookResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation: str
    steps: list[str]
    risk: str


class ReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_supported: bool
    unsupported_claims: list[str]
    notes: list[str]


class ProviderError(RuntimeError):
    """Raised when a provider request or its structured response fails."""


class LLMServiceError(RuntimeError):
    """Raised when configured LLM providers cannot return a valid response."""


def _request_error_category(exc: Exception) -> str:
    """Return a safe provider failure category without exposing response details."""

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int):
        status_code = getattr(exc, "code", None)

    if status_code == 400:
        detail = _safe_bad_request_detail(exc)
        return detail or "request rejected"
    if status_code == 401:
        return "authentication failed"
    if status_code == 403:
        return "permission denied"
    if status_code == 404:
        return "model or endpoint not found"
    if status_code == 429:
        return "quota or rate limit exceeded"
    if isinstance(status_code, int) and status_code >= 500:
        return "provider unavailable"
    if isinstance(exc, httpx.TimeoutException):
        return "request timed out"
    if isinstance(exc, httpx.NetworkError):
        return "network connection failed"
    return "request failed"


def _safe_bad_request_detail(exc: Exception) -> str | None:
    """Extract only allowlisted error codes or parameter names from provider errors."""

    code_categories = {
        "API_KEY_INVALID": "authentication failed",
        "INVALID_API_KEY": "authentication failed",
        "UNAUTHENTICATED": "authentication failed",
        "PERMISSION_DENIED": "permission denied",
        "MODEL_NOT_FOUND": "model or endpoint not found",
        "NOT_FOUND": "model or endpoint not found",
        "RESOURCE_EXHAUSTED": "quota or rate limit exceeded",
        "UNSUPPORTED_PARAMETER": "unsupported request parameter",
        "UNSUPPORTED_VALUE": "unsupported request value",
        "INVALID_ARGUMENT": "invalid request",
        "INVALID_REQUEST": "invalid request",
        "INVALID_REQUEST_ERROR": "invalid request",
    }
    allowed_parameters = {
        "contents",
        "generation_config",
        "max_completion_tokens",
        "max_output_tokens",
        "max_tokens",
        "mime_type",
        "model",
        "response_format",
        "response_mime_type",
        "response_schema",
        "schema",
        "system_instruction",
        "temperature",
    }

    body = getattr(exc, "body", None)
    generic_category = None
    message = getattr(exc, "message", None)
    if isinstance(body, dict):
        error_body = body.get("error", body)
        if isinstance(error_body, dict):
            for field in ("status", "code"):
                value = error_body.get(field)
                if isinstance(value, str):
                    category = code_categories.get(value.upper())
                    if category is not None:
                        if category != "invalid request":
                            return category
                        generic_category = category

            parameter = error_body.get("param") or error_body.get("parameter")
            if isinstance(parameter, str) and parameter in allowed_parameters:
                return f"request rejected (parameter: {parameter})"

            if isinstance(error_body.get("message"), str):
                message = error_body["message"]
            value = error_body.get("type")
            if isinstance(value, str):
                category = code_categories.get(value.upper())
                if category is not None:
                    if category != "invalid request":
                        return category
                    generic_category = category

    for field in ("status", "code", "type"):
        value = getattr(exc, field, None)
        if isinstance(value, str):
            category = code_categories.get(value.upper())
            if category is not None:
                if category != "invalid request":
                    return category
                generic_category = category

    if isinstance(message, str):
        normalized_message = message.lower()
        if "api key not valid" in normalized_message or "invalid api key" in normalized_message:
            return "authentication failed"
        for parameter in sorted(allowed_parameters, key=len, reverse=True):
            if parameter in normalized_message:
                return f"request rejected (parameter: {parameter})"
        if "response schema" in normalized_message or "json schema" in normalized_message:
            return "request rejected (response schema)"
        if "unsupported parameter" in normalized_message:
            return "request rejected (unsupported parameter)"

    return generic_category


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class StructuredLLM(Protocol):
    async def generate(
        self,
        response_model: type[ResponseModel],
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> tuple[ResponseModel, str]: ...


class JSONProvider(Protocol):
    name: str

    def generate(
        self,
        response_model: type[BaseModel],
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> BaseModel: ...


def _validate_response(response_model: type[ResponseModel], parsed: Any) -> ResponseModel:
    try:
        if isinstance(parsed, response_model):
            return parsed
        return response_model.model_validate(parsed)
    except ValidationError as exc:
        raise ProviderError("LLM response did not match the requested schema.") from exc


class GeminiProvider:
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.name = f"Gemini ({model})"

    def generate(
        self,
        response_model: type[ResponseModel],
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> ResponseModel:
        from google.genai.errors import APIError

        try:
            response = self._client.interactions.create(
                model=self._model,
                input=json.dumps(user_payload, ensure_ascii=False),
                system_instruction=(
                    f"{system_prompt}\n"
                    "Return only a valid JSON object matching the response schema."
                ),
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": response_model.model_json_schema(),
                },
                store=False,
            )
        except (APIError, httpx.HTTPError) as exc:
            category = _request_error_category(exc)
            raise ProviderError(f"request failed: {category}.") from exc

        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("returned an empty response.")
        try:
            return response_model.model_validate_json(text)
        except ValidationError as exc:
            raise ProviderError("returned invalid structured output.") from exc


class GroqProvider:
    def __init__(self, api_key: str, model: str) -> None:
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model
        self.name = f"Groq ({model})"

    def generate(
        self,
        response_model: type[ResponseModel],
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> ResponseModel:
        from groq import APIError

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{system_prompt}\n"
                            "Return only a valid JSON object matching the response schema."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__.lower(),
                        "strict": True,
                        "schema": response_model.model_json_schema(),
                    },
                },
                temperature=0,
                max_completion_tokens=512,
            )
        except (APIError, httpx.HTTPError) as exc:
            category = _request_error_category(exc)
            raise ProviderError(f"request failed: {category}.") from exc

        choices = getattr(response, "choices", [])
        if not choices:
            raise ProviderError("returned no completion choices.")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("returned an empty response.")
        try:
            return response_model.model_validate_json(content)
        except ValidationError as exc:
            raise ProviderError("returned invalid structured output.") from exc


class LLMClient:
    """Validate structured model output and use the configured fallback on failure."""

    def __init__(self, primary: JSONProvider, fallback: JSONProvider | None = None) -> None:
        self._primary = primary
        self._fallback = fallback

    async def generate(
        self,
        response_model: type[ResponseModel],
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> tuple[ResponseModel, str]:
        providers = [self._primary]
        if self._fallback is not None:
            providers.append(self._fallback)

        errors: list[tuple[str, ProviderError]] = []
        for provider in providers:
            try:
                result = await asyncio.to_thread(
                    provider.generate,
                    response_model,
                    system_prompt,
                    user_payload,
                )
                return _validate_response(response_model, result), provider.name
            except ProviderError as exc:
                errors.append((provider.name, exc))

        if len(errors) == 1:
            provider_name, error = errors[0]
            raise LLMServiceError(
                f"{provider_name} failed: {error}"
            ) from error

        primary_name, primary_error = errors[0]
        fallback_name, fallback_error = errors[1]
        raise LLMServiceError(
            f"{primary_name} failed: {primary_error} "
            f"{fallback_name} failed: {fallback_error}"
        ) from fallback_error


def build_llm_client(settings: Settings) -> LLMClient:
    """Build Gemini-first/Groq-fallback clients from local settings."""

    if not settings.gemini_api_key and not settings.groq_api_key:
        raise ValueError("Set GEMINI_API_KEY or GROQ_API_KEY in your local .env file.")

    groq_provider = (
        GroqProvider(settings.groq_api_key, settings.groq_model)
        if settings.groq_api_key
        else None
    )
    if settings.gemini_api_key:
        return LLMClient(
            GeminiProvider(settings.gemini_api_key, settings.gemini_model),
            fallback=groq_provider,
        )
    if groq_provider is None:
        raise ValueError("No LLM provider is configured.")
    return LLMClient(groq_provider)
