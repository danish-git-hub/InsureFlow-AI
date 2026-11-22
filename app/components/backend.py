"""
Dashboard <-> backend integration.

Write operations (submit, process, review, policy upload) go through the
FastAPI backend when it is running, so the dashboard exercises the same API
an insurer's other systems would use. If the API is not reachable, the same
service functions are called in-process, and the UI states which path was
used ("via FastAPI" / "in-process") — nothing is hidden.

Read-only pages query the database directly through the repositories, which
keeps charts fast and still uses only real records.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import httpx

from app.config import settings
from database.connection import SessionLocal
from services.claim_service import ServiceError
from services.schemas import ClaimSubmission, ReviewSubmission, UploadedFile

API = settings.api_base_url.rstrip("/") + "/api/v1"
TIMEOUT = httpx.Timeout(120.0, connect=1.5)

API_HEADERS = (
    {"X-API-Token": settings.api_access_token}
    if settings.api_access_token
    else {}
)


def api_online() -> bool:
    try:
        return httpx.get(settings.api_base_url.rstrip("/") + "/health", timeout=1.5).status_code == 200
    except httpx.HTTPError:
        return False


@contextmanager
def db_session():
    """Session for in-process writes: commit on success, roll back on any error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _raise_api_error(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        err = resp.json().get("error", {})
    except ValueError:
        err = {}
    raise ServiceError(err.get("message") or f"API error {resp.status_code}", resp.status_code, err.get("details"))


# --------------------------------------------------------------------------- #
def submit_claim(form: ClaimSubmission, files: list[UploadedFile], process_now: bool) -> tuple[dict, str]:
    if api_online():
        data = {k: (v.isoformat() if hasattr(v, "isoformat") else str(v)) for k, v in
                form.model_dump(mode="json").items() if v is not None}
        data["process_now"] = str(process_now).lower()
        data["document_types"] = ",".join(f.declared_type or "OTHER" for f in files)
        multipart = [("files", (f.filename, f.content, "application/pdf")) for f in files]
        resp = httpx.post(f"{API}/claims", data=data, files=multipart, headers=API_HEADERS, timeout=TIMEOUT)
        _raise_api_error(resp)
        return resp.json(), "FastAPI"

    from services import claim_service

    with db_session() as s:
        claim, warnings = claim_service.submit_claim(s, form, files)
        number = claim.claim_number
        s.commit()
        result: dict[str, Any] = {"claim_number": number, "warnings": warnings, "documents": len(files)}
        if process_now:
            state = claim_service.process(s, number)
            result["processing"] = _state_summary(state)
    return result, "in-process"


def process_claim(claim_number: str) -> tuple[dict, str]:
    if api_online():
        resp = httpx.post(f"{API}/claims/{claim_number}/process", headers=API_HEADERS, timeout=TIMEOUT)
        _raise_api_error(resp)
        return resp.json(), "FastAPI"

    from services import claim_service

    with db_session() as s:
        return _state_summary(claim_service.process(s, claim_number)), "in-process"


def record_review(claim_number: str, review: ReviewSubmission) -> tuple[dict, str]:
    if api_online():
        resp = httpx.post(f"{API}/claims/{claim_number}/reviews", json=review.model_dump(mode="json"), headers=API_HEADERS, timeout=TIMEOUT)
        _raise_api_error(resp)
        return resp.json(), "FastAPI"

    from services import claim_service

    with db_session() as s:
        r = claim_service.record_review(s, claim_number, review)
        return {"claim_number": claim_number, "decision": r.decision,
                "claim_status": r.claim.claim_status}, "in-process"


def upload_policy_document(filename: str, content: bytes) -> tuple[dict, str]:
    if api_online():
        resp = httpx.post(f"{API}/policies/documents", files={"file": (filename, content, "application/pdf")}, headers=API_HEADERS,
                          timeout=TIMEOUT)
        _raise_api_error(resp)
        return resp.json(), "FastAPI"

    from services import claim_service

    with db_session() as s:
        return claim_service.ingest_policy_document(s, filename, content), "in-process"


def _state_summary(state: dict) -> dict:
    decision = state.get("decision_support") or {}
    return {
        "claim_number": state.get("claim_number"),
        "run_id": state.get("run_id"),
        "processing_status": state.get("processing_status"),
        "recommendation": decision.get("recommendation"),
        "confidence": decision.get("confidence"),
        "path": [r["node"] for r in state.get("execution_history") or []],
        "skipped_agents": state.get("skipped_agents") or [],
        "errors": state.get("error_information") or [],
    }
