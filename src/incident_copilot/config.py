"""Runtime settings for the LLM providers."""

from dataclasses import dataclass, field
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None = field(repr=False)
    groq_api_key: str | None = field(repr=False)
    gemini_model: str
    groq_model: str


def get_settings() -> Settings:
    """Load local environment settings without overriding existing variables."""

    load_dotenv()
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    )
