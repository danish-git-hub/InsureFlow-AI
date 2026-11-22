"""
Central configuration for InsureFlow AI.

Every module (dashboard, API, agents, database) reads settings from here, so
there is exactly one place that knows about environment variables.

Beginner note:
    Secrets such as GOOGLE_API_KEY are read from the environment (or a local
    `.env` file loaded by python-dotenv). They are never written in code. If
    no API key is present, `settings.demo_mode` is True and every agent uses
    deterministic, rule-based logic instead of calling Gemini.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains this `app/` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load `.env` from the project root (silently ignored if it does not exist).
load_dotenv(PROJECT_ROOT / ".env")


def _normalize_database_url(url: str) -> str:
    """Normalize database URLs for SQLite and PostgreSQL."""
    prefix = "sqlite:///"

    if url.startswith(prefix) and not url.startswith(prefix + "/"):
        relative = url[len(prefix):]
        if relative and relative != ":memory:":
            return prefix + str((PROJECT_ROOT / relative).resolve())

    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)

    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)

    return url


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded once at start-up."""

    app_name: str = "InsureFlow AI"
    tagline: str = "Intelligent Claims. Trusted Decisions."
    version: str = "0.1.0"

    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", "").strip())
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_embedding_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    )

    database_url: str = field(
        default_factory=lambda: _normalize_database_url(
            os.getenv("DATABASE_URL", "sqlite:///data/insureflow.db")
        )
    )
    api_base_url: str = field(default_factory=lambda: os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "10")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    # Folder layout
    data_dir: Path = PROJECT_ROOT / "data"
    policies_dir: Path = PROJECT_ROOT / "data" / "policies"
    claims_dir: Path = PROJECT_ROOT / "data" / "claims"
    synthetic_dir: Path = PROJECT_ROOT / "data" / "synthetic"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    vectorstore_dir: Path = PROJECT_ROOT / "data" / "vectorstore"

    @property
    def demo_mode(self) -> bool:
        """True when no Gemini key is configured -> rule-based demo mode."""
        return not self.google_api_key or self.google_api_key == "your_api_key_here"

    @property
    def mode_label(self) -> str:
        return "DEMO MODE (rule-based, no LLM calls)" if self.demo_mode else f"LIVE (Gemini: {self.gemini_model})"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format. Never log secrets."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
