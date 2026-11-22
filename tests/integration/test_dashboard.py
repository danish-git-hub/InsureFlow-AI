"""Phase 10 — every dashboard page renders without errors against an isolated, processed database."""

import pytest
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from app.config import PROJECT_ROOT
from database.repository import ClaimRepository
from tests.conftest import REFERENCE_DATE
from workflows.graph import process_claim

PAGES = ["dashboard", "system_status", "new_claim", "processing_center", "claim_management", "claim_details",
         "policy_management", "human_review", "reports"]


@pytest.fixture
def isolated_db(engine, seeded_session, policy_index, monkeypatch):
    claims = ClaimRepository(seeded_session).list()
    for claim in claims[:4]:  # a few processed claims so detail/review pages have content
        process_claim(seeded_session, claim, rag_index=policy_index, today=REFERENCE_DATE, use_llm=False)
    import app.components.backend
    import database.connection

    monkeypatch.setattr(database.connection, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(app.components.backend, "api_online", lambda: False)
    return claims[0].claim_number


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(isolated_db, page):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "main.py"), default_timeout=60)
    at.session_state["selected_claim"] = isolated_db
    at.run()
    at.switch_page(f"pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("if-header" in m.value for m in at.markdown)


def test_dashboard_kpis_come_from_database(isolated_db):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "main.py"), default_timeout=60)
    at.run()
    html = " ".join(m.value for m in at.markdown)
    assert ">33<" in html  # total claims in the isolated database, not a hard-coded number
    assert "Avg Processing Time" in html


def test_new_claim_form_shows_validation_errors(isolated_db):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "main.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/new_claim.py").run()
    submit = next(b for b in at.button if b.label == "Submit claim")
    submit.click().run()
    errors = " ".join(e.value for e in at.error)
    assert "Customer Name" in errors and "Policy Number" in errors
