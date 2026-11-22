"""
Gemini access for all agents — one place for configuration, retries and validation.

Rules enforced here:
    * The API key only comes from the environment (settings.google_api_key).
    * In DEMO MODE (no key) `llm_available()` is False and NO network call is made.
    * Every response is parsed into a Pydantic schema; invalid output -> None.
    * Callers receive an `LLMCall` record so they can honestly report whether
      the LLM was actually used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import get_logger, settings

log = get_logger("llm")
T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMCall:
    ok: bool
    model: str | None
    attempts: int
    duration_ms: float
    error: str | None = None


def llm_available() -> bool:
    return not settings.demo_mode


@lru_cache(maxsize=1)
def get_chat_model():
    """Create the Gemini chat model lazily (import cost and key only when needed)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,  # deterministic, factual extraction
        max_retries=0,  # we handle retries ourselves below
        timeout=45,
    )


def invoke_structured(prompt: str, schema: type[T], retries: int = 2, model=None) -> tuple[T | None, LLMCall]:
    """Call Gemini and parse the answer into `schema`.

    Retries transient failures with exponential backoff (1s, 2s, ...).
    Returns (parsed_object_or_None, call_record). Never raises.
    """
    if model is None and not llm_available():
        return None, LLMCall(False, None, 0, 0.0, "LLM disabled (demo mode)")

    start = time.perf_counter()
    model = model or get_chat_model()
    structured = model.with_structured_output(schema)
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            result = structured.invoke(prompt)
            if isinstance(result, dict):
                result = schema.model_validate(result)
            if not isinstance(result, schema):
                raise TypeError(f"LLM returned {type(result).__name__}, expected {schema.__name__}")
            return result, LLMCall(True, settings.gemini_model, attempt, (time.perf_counter() - start) * 1000)
        except Exception as exc:  # network error, quota, invalid JSON, schema mismatch ...
            last_error = f"{exc.__class__.__name__}: {str(exc)[:200]}"
            log.warning("LLM attempt %s failed: %s", attempt, last_error)
            if attempt <= retries:
                time.sleep(2 ** (attempt - 1))
    return None, LLMCall(False, settings.gemini_model, retries + 1, (time.perf_counter() - start) * 1000, last_error)
