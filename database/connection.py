"""
Database engine and session management.

Beginner note:
    * The *engine* knows how to talk to the database (SQLite file by default).
    * A *session* is a short-lived "unit of work": you add/query objects, then
      commit (save everything) or roll back (undo everything) as one transaction.
    * `session_scope()` wraps that pattern so a failure never leaves half-saved data.

Switching to PostgreSQL only requires changing DATABASE_URL in `.env`
(e.g. postgresql+psycopg://user:pass@host/insureflow) and installing a driver.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings


def make_engine(url: str | None = None, echo: bool = False) -> Engine:
    """Create an engine. SQLite gets foreign-key enforcement switched on."""
    url = url or settings.database_url
    is_sqlite = url.startswith("sqlite")
    engine = create_engine(
        url,
        echo=echo,
        future=True,
        # Streamlit and FastAPI use worker threads; SQLite needs this flag for that.
        connect_args={"check_same_thread": False} if is_sqlite else {},
    )
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _enable_sqlite_fk(dbapi_conn, _record):  # pragma: no cover - driver hook
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


# Default application engine + session factory.
engine: Engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker = SessionLocal) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request (used in Phase 9)."""
    with session_scope() as session:
        yield session
